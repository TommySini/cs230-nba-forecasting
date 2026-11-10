"""Situational and availability features for the boosting model, from games on earlier ET dates only.

Families (per team; the model gets home, away and home-minus-away columns):
  recency    exponentially weighted net/offensive/defensive rating (half-life 5 games), plain and
             opponent-adjusted (each game's net rating plus the opponent's pre-game season net rating)
  fatigue    games in the last 4 and 7 days, miles travelled since the previous game and in the
             last 7 days, time-zone shift, altitude visits (Denver, Utah) on short rest
  rematch    same opponent in the team's previous game within 4 days, and that game's margin
  shooting   3-point and effective-FG "luck": last-5 percentage minus season-to-date, for and against,
             and 3-point percentage shrunk toward the league
  lineup     box-score value of the rotation (top 9 by minutes): game score per minute times expected
             minutes, for everyone and for players who played the team's most recent game
  injury     official NBA injury report at or before the cutoff (from collect_injuries.py): players
             listed Out / Doubtful / Questionable for injury, illness or rest (G League excluded)

Nothing here reads the game being predicted or any later game: each ET date's features are computed
before that date's games enter the running state. Shot-quality (xeFG%) and on/off ratings need
tracking or play-by-play data the project does not have; `lineup` is the box-score substitute.
"""
from collections import defaultdict
from datetime import date, timedelta
import json
import math
import re

import numpy as np

HALF_LIFE = 5
ROTATION = 9
SHRINK_ATTEMPTS = 300
# Arena latitude, longitude, elevation (ft), standard UTC offset. Approximate; current arenas.
ARENAS = {
    "ATL": (33.757, -84.396, 1050, -5), "BOS": (42.366, -71.062, 20, -5), "BKN": (40.683, -73.975, 30, -5),
    "CHA": (35.225, -80.839, 750, -5), "CHI": (41.881, -87.674, 600, -6), "CLE": (41.496, -81.688, 650, -5),
    "DAL": (32.790, -96.810, 430, -6), "DEN": (39.749, -105.008, 5280, -7), "DET": (42.341, -83.055, 600, -5),
    "GSW": (37.768, -122.388, 10, -8), "HOU": (29.751, -95.362, 50, -6), "IND": (39.764, -86.155, 715, -5),
    "LAC": (34.043, -118.267, 300, -8), "LAL": (34.043, -118.267, 300, -8), "MEM": (35.138, -90.051, 250, -6),
    "MIA": (25.781, -80.188, 10, -5), "MIL": (43.045, -87.917, 600, -6), "MIN": (44.980, -93.276, 830, -6),
    "NOP": (29.949, -90.082, 10, -6), "NYK": (40.751, -73.993, 30, -5), "OKC": (35.463, -97.515, 1200, -6),
    "ORL": (28.539, -81.384, 100, -5), "PHI": (39.901, -75.172, 30, -5), "PHX": (33.446, -112.071, 1100, -7),
    "POR": (45.532, -122.667, 50, -8), "SAC": (38.580, -121.500, 30, -8), "SAS": (29.427, -98.438, 650, -6),
    "TOR": (43.643, -79.379, 250, -5), "UTA": (40.768, -111.901, 4226, -7), "WAS": (38.898, -77.021, 50, -5),
}
BUBBLE = ((28.370, -81.550, 100, -5), date(2020, 7, 30), date(2020, 10, 15))  # 2019-20 restart, Orlando
ALTITUDE_FEET = 4000
INJURY_STATUSES = ("Out", "Doubtful", "Questionable")
INJURY_ROW = re.compile(r"\b(Out|Doubtful|Questionable|Probable|Available)\s+(Injury/Illness|G League|Not With Team|"
                        r"Personal Reasons?|Rest|Health and Safety Protocols|League Suspension|Concussion Protocol)")

PER_TEAM = ("ewm_net", "ewm_off", "ewm_def", "ewm_adj_net", "season_adj_net", "schedule_strength",
            "games_last_4d", "games_last_7d", "miles_since_last", "miles_last_7d", "tz_shift", "altitude_short_rest",
            "fatigue_score", "rematch", "rematch_prev_margin",
            "fg3_luck", "efg_luck", "opp_fg3_luck", "fg3_shrunk",
            "rotation_value", "available_value", "available_share", "rotation_minutes_missing_last", "top_player_missed_last",
            "injury_out", "injury_doubtful", "injury_questionable", "injury_report_found")
FAMILIES = {
    "recency": ("ewm_net", "ewm_off", "ewm_def", "ewm_adj_net", "season_adj_net", "schedule_strength"),
    "fatigue": ("games_last_4d", "games_last_7d", "miles_since_last", "miles_last_7d", "tz_shift",
                "altitude_short_rest", "fatigue_score"),
    "rematch": ("rematch", "rematch_prev_margin"),
    "shooting": ("fg3_luck", "efg_luck", "opp_fg3_luck", "fg3_shrunk"),
    "lineup": ("rotation_value", "available_value", "available_share", "rotation_minutes_missing_last", "top_player_missed_last"),
    "injury": ("injury_out", "injury_doubtful", "injury_questionable", "injury_report_found"),
}
FEATURE_NAMES = tuple(f"{side}_{name}" for side in ("home", "away", "diff") for name in PER_TEAM)
# Constant by construction: home teams never visit altitude; both teams share one report.
CONSTANT = ("home_altitude_short_rest", "diff_injury_report_found")


def feature_names(families):
    """Model columns for the chosen families (constant columns left out)."""
    keep = {name for family in families for name in FAMILIES[family]}
    return [f"{side}_{name}" for side in ("home", "away", "diff") for name in PER_TEAM
            if name in keep and f"{side}_{name}" not in CONSTANT]


def _day(value):
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def venue(game):
    when = _day(game["game_date_et"])
    if game["season"] == 2019 and BUBBLE[1] <= when <= BUBBLE[2]:
        return BUBBLE[0]
    return ARENAS[game["home_abbr"]]


def miles(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def possessions(box):
    return box["fga"] - box["oreb"] + box["tov"] + 0.44 * box["fta"]


def game_score(p):
    return (p["pts"] + 0.4 * p["fgm"] - 0.7 * p["fga"] - 0.4 * (p["fta"] - p["ftm"]) + 0.7 * p["oreb"]
            + 0.3 * p["dreb"] + p["stl"] + 0.7 * p["ast"] + 0.7 * p["blk"] - 0.4 * p["pf"] - p["tov"])


def ewm(values, half_life=HALF_LIFE):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return math.nan
    w = 0.5 ** (np.arange(len(values))[::-1] / half_life)
    return float(np.dot(w, values) / w.sum())


def _mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else math.nan


def _ratio(made, attempts):
    return made / attempts if attempts else math.nan


def injury_counts(row):
    """(home, away) dicts of injury-report counts, or None if no usable report for the game."""
    report = row.get("injury_report") if row else None
    if not report or report.get("not_yet_submitted"):
        return None
    text = report["text"]
    i_home, i_away = text.find(row["home_team"]), text.find(row["away_team"])
    segments = {"home": "", "away": ""}
    if i_home >= 0 and i_away >= 0:
        if i_home < i_away:
            segments = {"home": text[i_home:i_away], "away": text[i_away:]}
        else:
            segments = {"away": text[i_away:i_home], "home": text[i_home:]}
    elif i_home >= 0:
        segments["home"] = text[i_home:]
    elif i_away >= 0:
        segments["away"] = text[i_away:]
    out = {}
    for side, segment in segments.items():
        counts = dict.fromkeys(INJURY_STATUSES, 0)
        for status, reason in INJURY_ROW.findall(segment):
            if status in counts and reason != "G League":
                counts[status] += 1
        out[side] = counts
    return out


def build(games, team_boxes, player_boxes, injuries=None):
    """game_id -> {feature: value} for every game, from earlier ET dates only.

    games: dicts with game_id, season, game_date_et, home_abbr, away_abbr, home_score, away_score.
    team_boxes: (game_id, side) -> team box (pts, fga, fta, fgm, fg3m, fg3a, oreb, tov).
    player_boxes: (game_id, side) -> list of player boxes (player_id, minutes, box counters).
    injuries: game_id -> collect_injuries.py row (report time already checked against the cutoff).
    """
    injuries = injuries or {}
    log = defaultdict(list)            # (season, team) -> per-game records, chronological
    league = defaultdict(lambda: [0.0, 0.0])  # season -> [fg3m, fg3a] so far
    by_date = defaultdict(list)
    for g in games:
        by_date[_day(g["game_date_et"])].append(g)
    features = {}
    for day in sorted(by_date):
        todays = sorted(by_date[day], key=lambda g: g["game_id"])
        pregame_net = {}
        for g in todays:
            row = {}
            for side, team, opp in (("home", g["home_abbr"], g["away_abbr"]), ("away", g["away_abbr"], g["home_abbr"])):
                past = log[g["season"], team]
                pregame_net[g["game_id"], side] = _mean([r["net"] for r in past])
                row.update({f"{side}_{k}": v for k, v in _team_features(g, day, side, opp, past, league[g["season"]]).items()})
            counts = injury_counts(injuries.get(g["game_id"]))
            for side in ("home", "away"):
                c = counts[side] if counts else None
                row[f"{side}_injury_report_found"] = float(counts is not None)
                for status in INJURY_STATUSES:
                    row[f"{side}_injury_{status.lower()}"] = float(c[status]) if c else math.nan
            for name in PER_TEAM:
                row[f"diff_{name}"] = row[f"home_{name}"] - row[f"away_{name}"]
            features[g["game_id"]] = row
        for g in todays:                # only now do this date's games enter the state
            _record(g, day, team_boxes, player_boxes, pregame_net, log, league)
    return features


def _team_features(g, day, side, opp, past, league_fg3):
    out = dict.fromkeys(PER_TEAM, math.nan)
    here = venue(g)
    boxed = [r for r in past if r["net"] is not None]
    out["ewm_net"] = ewm([r["net"] for r in boxed])
    out["ewm_off"] = ewm([r["off"] for r in boxed])
    out["ewm_def"] = ewm([r["def"] for r in boxed])
    out["ewm_adj_net"] = ewm([r["adj_net"] for r in boxed])
    out["season_adj_net"] = _mean([r["adj_net"] for r in boxed])
    out["schedule_strength"] = _mean([r["opp_pregame_net"] for r in past])
    out["games_last_4d"] = float(sum(day - r["day"] <= timedelta(days=4) for r in past))
    out["games_last_7d"] = float(sum(day - r["day"] <= timedelta(days=7) for r in past))
    if past:
        last = past[-1]
        out["miles_since_last"] = miles(last["venue"], here)
        out["tz_shift"] = float(here[3] - last["venue"][3])
        path = [r["venue"] for r in past if day - r["day"] <= timedelta(days=7)] + [here]
        out["miles_last_7d"] = sum(miles(a, b) for a, b in zip(path, path[1:])) if len(path) > 1 else 0.0
        rest = (day - last["day"]).days
        out["altitude_short_rest"] = float(side == "away" and here[2] >= ALTITUDE_FEET and rest <= 1)
        out["fatigue_score"] = (out["games_last_4d"] + out["games_last_7d"] / 2 + out["miles_last_7d"] / 1000
                                + abs(out["tz_shift"]) / 2 + 2 * out["altitude_short_rest"] + 2 * (rest <= 1))
        out["rematch"] = float(last["opp"] == opp and rest <= 4)
        out["rematch_prev_margin"] = last["margin"] if out["rematch"] else 0.0
    else:
        out.update(miles_since_last=0.0, tz_shift=0.0, miles_last_7d=0.0, altitude_short_rest=0.0,
                   fatigue_score=0.0, rematch=0.0, rematch_prev_margin=0.0)
    shots = [r for r in boxed if r["fg3a"]]
    if shots:
        season = _ratio(sum(r["fg3m"] for r in shots), sum(r["fg3a"] for r in shots))
        recent = shots[-5:]
        out["fg3_luck"] = _ratio(sum(r["fg3m"] for r in recent), sum(r["fg3a"] for r in recent)) - season
        out["opp_fg3_luck"] = (_ratio(sum(r["opp_fg3m"] for r in recent), sum(r["opp_fg3a"] for r in recent))
                               - _ratio(sum(r["opp_fg3m"] for r in shots), sum(r["opp_fg3a"] for r in shots)))
        out["efg_luck"] = _mean([r["efg"] for r in recent]) - _mean([r["efg"] for r in shots])
        league_pct = _ratio(*league_fg3) if league_fg3[1] else season
        out["fg3_shrunk"] = (sum(r["fg3m"] for r in shots) + SHRINK_ATTEMPTS * league_pct) / (
            sum(r["fg3a"] for r in shots) + SHRINK_ATTEMPTS)
    played = [r for r in past if r["players"] is not None]
    if played:
        total_minutes, total_value, appearances, recent_minutes = defaultdict(float), defaultdict(float), defaultdict(int), defaultdict(list)
        for r in played:
            for pid, (minutes, value) in r["players"].items():
                total_minutes[pid] += minutes
                total_value[pid] += value
                appearances[pid] += 1
        for r in played[-10:]:
            for pid, (minutes, _) in r["players"].items():
                recent_minutes[pid].append(minutes)
        rotation = sorted(total_minutes, key=lambda p: (-total_minutes[p], p))[:ROTATION]
        expected = {p: (np.mean(recent_minutes[p]) if recent_minutes[p] else total_minutes[p] / appearances[p]) for p in rotation}
        value = {p: total_value[p] / total_minutes[p] * expected[p] if total_minutes[p] else 0.0 for p in rotation}
        last_players = played[-1]["players"]
        out["rotation_value"] = float(sum(value.values()))
        out["available_value"] = float(sum(v for p, v in value.items() if p in last_players))
        out["available_share"] = out["available_value"] / out["rotation_value"] if out["rotation_value"] > 0 else math.nan
        minutes_total = sum(expected.values())
        out["rotation_minutes_missing_last"] = (sum(m for p, m in expected.items() if p not in last_players) / minutes_total
                                                if minutes_total else math.nan)
        top = max(rotation, key=lambda p: (value[p], p))
        out["top_player_missed_last"] = float(top not in last_players)
    return out


def _record(g, day, team_boxes, player_boxes, pregame_net, log, league):
    boxes = {side: team_boxes.get((g["game_id"], side)) for side in ("home", "away")}
    complete = all(b is not None and b.get("complete") and b.get("fga") for b in boxes.values())
    poss = (possessions(boxes["home"]) + possessions(boxes["away"])) / 2 if complete else None
    for side, team, opp, other in (("home", g["home_abbr"], g["away_abbr"], "away"), ("away", g["away_abbr"], g["home_abbr"], "home")):
        record = {"day": day, "game_id": g["game_id"], "opp": opp, "venue": venue(g),
                  "margin": float(g[f"{side}_score"] - g[f"{other}_score"]),
                  "opp_pregame_net": pregame_net[g["game_id"], other], "net": None, "players": None}
        if complete and poss:
            mine, theirs = boxes[side], boxes[other]
            record.update(off=100 * mine["pts"] / poss, **{"def": 100 * theirs["pts"] / poss})
            record["net"] = record["off"] - record["def"]
            opp_strength = record["opp_pregame_net"]
            record["adj_net"] = record["net"] + (opp_strength if math.isfinite(opp_strength) else 0.0)
            record.update(fg3m=mine["fg3m"], fg3a=mine["fg3a"], opp_fg3m=theirs["fg3m"], opp_fg3a=theirs["fg3a"],
                          efg=_ratio(mine["fgm"] + 0.5 * mine["fg3m"], mine["fga"]))
            league[g["season"]][0] += mine["fg3m"]
            league[g["season"]][1] += mine["fg3a"]
        players = player_boxes.get((g["game_id"], side))
        if players:
            record["players"] = {p["player_id"]: (float(p["minutes"]), game_score(p)) for p in players
                                 if p.get("minutes") and all(p.get(k) is not None for k in ("pts", "fga", "tov"))}
        log[g["season"], g[f"{side}_abbr"]].append(record)


def from_dataset(dataset, injury_paths=()):
    """Build features for every franchise game in the shared dataset, plus collect_injuries.py outputs if given."""
    from collect_pilot import stamp
    games = dataset.history()
    team_boxes = {(r["game_id"], r["side"]): r for r in dataset.table("team_box_scores")}
    complete = {r["game_id"] for r in dataset.table("box_score_games") if r["players_complete"]}
    player_boxes = defaultdict(list)
    for r in dataset.table("player_box_scores"):
        if r["game_id"] in complete:
            player_boxes[r["game_id"], r["side"]].append(r)
    injuries = {}
    cutoff = {g["game_id"]: g["cutoff_utc"] for g in dataset.table("games")}
    for path in injury_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            report = row.get("injury_report")
            if report and stamp(report["report_time_utc"]) > stamp(cutoff[row["game_id"]]):
                raise ValueError(f"Injury report after cutoff: {row['game_id']}")
            injuries[row["game_id"]] = row
    return build(games, team_boxes, player_boxes, injuries)
