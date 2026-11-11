"""Team and player rating features that carry across seasons, from earlier games only.

  elo     Elo with margin-of-victory multiplier and 75% carry-over between seasons (K 20, home 100)
  apm     game-level adjusted plus-minus: a ridge regression of each game's home margin on who played
          how many minutes (minutes / 48, home positive, away negative, plus a home-court term),
          time-decayed (half-life HALF_LIFE_DAYS) and refitted every REFIT_DAYS on games before the
          refit date. A team's strength is the minutes-weighted rating of its expected rotation:
            full      everyone in the rotation (last 10 games), minutes as averaged when they played
            last_game only players who played the team's previous game
            injury    the full rotation minus players listed Out or Doubtful on the official injury
                      report (Questionable counts fully: chosen on TRAIN); missing where no report was matched
          Remaining minutes are rescaled to 240, so a missing star's minutes go to his teammates.
          With a box-score prior (default), each player's rating is c * (his decayed game score per
          minute, shrunk by BOX_SHRINK_MINUTES and centred) plus a ridge correction; c is fitted jointly.

Ratings for a game use only games on earlier dates (refits happen before the refit date's games);
the injury report is the one collected at or before the game's cutoff.
"""
from collections import defaultdict
from datetime import date, timedelta
import math
import re
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

ELO_K, ELO_HOME, ELO_CARRY, ELO_MEAN = 20.0, 100.0, 0.75, 1505.0
HALF_LIFE_DAYS = 365.0
REFIT_DAYS = 7
RIDGE = 15.0
ROTATION_GAMES = 10
BOX_SHRINK_MINUTES = 100.0
STATUS_WEIGHT = {"Out": 0.0, "Doubtful": 0.0, "Questionable": 1.0, "Probable": 1.0, "Available": 1.0}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
NAME_ROW = re.compile(r"([A-Z][\w'.\-]*(?: [A-Z][\w'.\-]*)*), ([A-Z][\w'.\-]*(?: [A-Z][\w'.\-]*)?) "
                      r"(Out|Doubtful|Questionable|Probable|Available)\b")

PER_TEAM = ("elo", "apm_full", "apm_last_game", "apm_injury", "injury_value_out", "injury_matched")
GAME = ("elo_prob", "apm_margin_full", "apm_margin_last_game", "apm_margin_injury", "apm_home_court")
FAMILIES = {"elo": ("elo", "elo_prob"),
            "apm": ("apm_full", "apm_last_game", "apm_margin_full", "apm_margin_last_game", "apm_home_court"),
            "apm_injury": ("apm_injury", "injury_value_out", "injury_matched", "apm_margin_injury")}
FEATURE_NAMES = tuple(f"{side}_{n}" for side in ("home", "away", "diff") for n in PER_TEAM) + GAME


def feature_names(families):
    keep = {n for f in families for n in FAMILIES[f]}
    return [n for n in FEATURE_NAMES if (n.split("_", 1)[1] if n.split("_", 1)[0] in ("home", "away", "diff") else n) in keep]


def _day(value):
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def normalize(name):
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    words = [w for w in re.sub(r"[^a-z ]", " ", name.replace("-", " ")).split() if w not in SUFFIXES]
    return " ".join(words)


def report_rows(segment):
    """(last-name words, first name, status) for every player row in one team's injury-report segment."""
    return [(m.group(1).split(" "), m.group(2), m.group(3)) for m in NAME_ROW.finditer(segment)]


def match(rows, roster, names):
    """{player_id: status} for report rows matched to this team's roster (by first+last, else unique last name)."""
    by_full, by_last = {}, defaultdict(list)
    for pid in roster:
        n = names.get(pid)
        if n:
            by_full[normalize(f"{n['first']} {n['last']}")] = pid
            by_last[normalize(n["last"])].append(pid)
    out = {}
    for last_words, first, status in rows:
        for k in range(len(last_words)):          # tolerate reason words glued before the name
            last = " ".join(last_words[k:])
            pid = by_full.get(normalize(f"{first} {last}"))
            if pid is None and len(by_last.get(normalize(last), [])) == 1:
                pid = by_last[normalize(last)][0]
            if pid is not None:
                out[pid] = status
                break
    return out


def _segments(report_row):
    text = report_row["injury_report"]["text"]
    i_home, i_away = text.find(report_row["home_team"]), text.find(report_row["away_team"])
    if i_home < 0 and i_away < 0:
        return None
    if i_home >= 0 and i_away >= 0:
        return ({"home": text[i_home:i_away], "away": text[i_away:]} if i_home < i_away
                else {"away": text[i_away:i_home], "home": text[i_home:]})
    return {"home": text[i_home:]} if i_home >= 0 else {"away": text[i_away:]}


def elo_features(games):
    """game_id -> (home_elo, away_elo, home_win_probability) before the game."""
    rating, last_season, out = defaultdict(lambda: ELO_MEAN), {}, {}
    by_date = defaultdict(list)
    for g in games:
        by_date[_day(g["game_date_et"])].append(g)
    for day in sorted(by_date):
        todays = by_date[day]
        for g in todays:
            for team in (g["home_abbr"], g["away_abbr"]):
                if last_season.get(team) not in (None, g["season"]):
                    rating[team] = ELO_CARRY * rating[team] + (1 - ELO_CARRY) * ELO_MEAN
                last_season[team] = g["season"]
            h, a = rating[g["home_abbr"]], rating[g["away_abbr"]]
            out[g["game_id"]] = (h, a, 1 / (1 + 10 ** (-(h + ELO_HOME - a) / 400)))
        for g in todays:
            h, a = rating[g["home_abbr"]], rating[g["away_abbr"]]
            expected = 1 / (1 + 10 ** (-(h + ELO_HOME - a) / 400))
            margin = g["home_score"] - g["away_score"]
            won = margin > 0
            diff = (h + ELO_HOME - a) * (1 if won else -1)
            multiplier = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * diff)
            shift = ELO_K * multiplier * (won - expected)
            rating[g["home_abbr"]] += shift
            rating[g["away_abbr"]] -= shift
    return out


class Apm:
    """Rolling ridge ratings; `at(day)` returns (ratings dict, home-court) fitted on games before `day`."""

    def __init__(self, games, minutes, ridge=RIDGE, half_life=HALF_LIFE_DAYS, refit_days=REFIT_DAYS, box=None):
        rows = [g for g in games if (g["game_id"], "home") in minutes and (g["game_id"], "away") in minutes]
        self.players = sorted({p for g in rows for side in ("home", "away") for p in minutes[g["game_id"], side]})
        index = {p: i for i, p in enumerate(self.players)}
        data, cols, ptr = [], [], [0]
        for g in rows:
            for side, sign in (("home", 1.0), ("away", -1.0)):
                for p, m in minutes[g["game_id"], side].items():
                    data.append(sign * m / 48.0); cols.append(index[p])
            data.append(1.0); cols.append(len(self.players))        # home-court term
            ptr.append(len(data))
        self.x = csr_matrix((data, cols, ptr), shape=(len(rows), len(self.players) + 1))
        self.y = np.array([g["home_score"] - g["away_score"] for g in rows], float)
        self.days = np.array([_day(g["game_date_et"]).toordinal() for g in rows])
        self.ridge, self.half_life, self.refit_days = ridge, half_life, refit_days
        self._fits = {}
        self.box = None
        if box:
            pg = [(r, index[p], m, box[g["game_id"], side].get(p, 0.0)) for r, g in enumerate(rows)
                  for side in ("home", "away") for p, m in minutes[g["game_id"], side].items() if (g["game_id"], side) in box]
            self.box = tuple(np.array(c) for c in zip(*pg))   # row, player, minutes, game score

    def at(self, day):
        anchor = day.toordinal() - (day.toordinal() % self.refit_days)
        if anchor not in self._fits:
            mask = self.days < anchor
            if mask.sum() < 200:
                self._fits[anchor] = ({}, 0.0)
            else:
                decay = 0.5 ** ((anchor - self.days[mask]) / self.half_life)
                w = np.sqrt(decay)
                x = self.x[mask]
                prior = None
                if self.box is not None:
                    rows, players, mins, score = self.box
                    keep = self.days[rows] < anchor
                    pw = 0.5 ** ((anchor - self.days[rows[keep]]) / self.half_life)
                    n = len(self.players)
                    m_sum = np.bincount(players[keep], pw * mins[keep], n)
                    s_sum = np.bincount(players[keep], pw * score[keep], n)
                    league = s_sum.sum() / max(m_sum.sum(), 1e-9)
                    prior = (s_sum + BOX_SHRINK_MINUTES * league) / (m_sum + BOX_SHRINK_MINUTES) - league
                    z = x[:, :n] @ prior                       # minutes-weighted prior strength per game
                    x = hstack([x, csr_matrix(z[:, None])]).tocsr()   # columns: players, home court, prior scale
                beta = lsqr(x.multiply(w[:, None]).tocsr(), self.y[mask] * w, damp=math.sqrt(self.ridge))[0]
                n = len(self.players)
                ratings = beta[:n] + (beta[n + 1] * prior if prior is not None else 0.0)
                self._fits[anchor] = (dict(zip(self.players, ratings)), float(beta[n]))
        return self._fits[anchor]


def build(games, minutes, injuries=None, names=None, ridge=RIDGE, box=None):
    """game_id -> {feature: value}. minutes: (game_id, side) -> {player_id: minutes};
    box: (game_id, side) -> {player_id: game score} for the box-score prior (None: pure plus-minus)."""
    injuries, names = injuries or {}, names or {}
    elo = elo_features(games)
    apm = Apm(games, minutes, ridge=ridge, half_life=HALF_LIFE_DAYS, box=box)
    log = defaultdict(list)                     # (season, team) -> [(day, {player: minutes})]
    by_date = defaultdict(list)
    for g in games:
        by_date[_day(g["game_date_et"])].append(g)
    out = {}
    for day in sorted(by_date):
        ratings, home_court = apm.at(day)
        for g in by_date[day]:
            row = {"apm_home_court": home_court}
            h, a, prob = elo[g["game_id"]]
            row.update(home_elo=h, away_elo=a, elo_prob=prob)
            report = injuries.get(g["game_id"])
            segments = _segments(report) if report and report.get("injury_report") else None
            for side in ("home", "away"):
                team = g[f"{side}_abbr"]
                recent = log[g["season"], team][-ROTATION_GAMES:]
                row.update(_team_strength(side, recent, ratings, segments, names))
            for name in PER_TEAM:
                row[f"diff_{name}"] = row[f"home_{name}"] - row[f"away_{name}"]
            for kind in ("full", "last_game", "injury"):
                row[f"apm_margin_{kind}"] = home_court + row[f"diff_apm_{kind}"]
            out[g["game_id"]] = row
        for g in by_date[day]:
            for side in ("home", "away"):
                played = minutes.get((g["game_id"], side))
                if played:
                    log[g["season"], g[f"{side}_abbr"]].append((day, played))
    return out


def _strength(expected, ratings):
    total = sum(expected.values())
    if total <= 0:
        return math.nan
    return sum(m * 240 / total / 48 * ratings.get(p, 0.0) for p, m in expected.items())


def _team_strength(side, recent, ratings, segments, names):
    out = {f"{side}_{k}": math.nan for k in ("apm_full", "apm_last_game", "apm_injury", "injury_value_out")}
    out[f"{side}_injury_matched"] = math.nan
    if not recent:
        return out
    played = defaultdict(list)
    for _, game in recent:
        for p, m in game.items():
            played[p].append(m)
    # Minutes a player averages when he plays; rotation players only.
    expected = {p: float(np.mean(ms)) for p, ms in played.items() if np.mean(ms) >= 5}
    out[f"{side}_apm_full"] = _strength(expected, ratings)
    last = recent[-1][1]
    out[f"{side}_apm_last_game"] = _strength({p: m for p, m in expected.items() if p in last}, ratings)
    if segments is not None and side in segments:
        status = match(report_rows(segments[side]), list(expected), names)
        # Available = played one of the last 3 games (so traded players drop out) or is on the report.
        active = {p for _, game in recent[-3:] for p in game} | set(status)
        adjusted = {p: m * STATUS_WEIGHT.get(status.get(p, "Available"), 1.0) for p, m in expected.items() if p in active}
        out[f"{side}_apm_injury"] = _strength(adjusted, ratings)
        out[f"{side}_injury_value_out"] = sum(m / 48 * max(ratings.get(p, 0.0), 0.0) * (1 - STATUS_WEIGHT.get(status.get(p), 1.0))
                                             for p, m in expected.items() if p in status)
        out[f"{side}_injury_matched"] = float(len(status))
    return out


def from_dataset(dataset, injury_paths=(), names_path=None, ridge=RIDGE, box_prior=True):
    import json
    from collect_pilot import stamp
    games = dataset.history()
    complete = {r["game_id"] for r in dataset.table("box_score_games") if r["players_complete"]}
    minutes, box = defaultdict(dict), defaultdict(dict)
    for r in dataset.table("player_box_scores"):
        if r["game_id"] in complete and r.get("minutes"):
            minutes[r["game_id"], r["side"]][r["player_id"]] = float(r["minutes"])
            if all(r.get(k) is not None for k in ("pts", "fga", "fgm", "fta", "ftm", "oreb", "dreb", "stl", "ast", "blk", "pf", "tov")):
                box[r["game_id"], r["side"]][r["player_id"]] = (r["pts"] + 0.4 * r["fgm"] - 0.7 * r["fga"] - 0.4 * (r["fta"] - r["ftm"])
                    + 0.7 * r["oreb"] + 0.3 * r["dreb"] + r["stl"] + 0.7 * r["ast"] + 0.7 * r["blk"] - 0.4 * r["pf"] - r["tov"])
    cutoff = {g["game_id"]: g["cutoff_utc"] for g in dataset.table("games")}
    injuries = {}
    for path in injury_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            report = row.get("injury_report")
            if report and stamp(report["report_time_utc"]) > stamp(cutoff[row["game_id"]]):
                raise ValueError(f"Injury report after cutoff: {row['game_id']}")
            injuries[row["game_id"]] = row
    names = json.loads(names_path.read_text()) if names_path and names_path.exists() else {}
    return build(games, dict(minutes), injuries, names, ridge=ridge, box=dict(box) if box_prior else None)
