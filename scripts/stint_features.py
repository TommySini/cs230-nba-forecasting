"""Lineup stints from ESPN play-by-play (sportsdataverse season files), for stint-level player ratings.

A stint is a stretch of one game with the same ten players on the court. Play-by-play does not
state who starts each period, so the five per team are inferred: players who act or are substituted
out before being substituted in, topped up from the players with the most box-score minutes when
fewer than five are seen. Stints are postgame data: like box scores, a game's stints may only feed
ratings for later games.

stints(events, home_minutes, away_minutes) -> [(seconds, home_margin_change, home_five, away_five)]

from_dataset keeps a game only when its stints add up to the final margin. `validate` compares the
plus-minus the stints imply with ESPN's own box-score plus-minus (collect_plus_minus.py).
"""
from collections import defaultdict
from pathlib import Path

PERIOD_SECONDS = 720
OVERTIME_SECONDS = 300
PBP_DIR = Path(__file__).resolve().parents[1] / "data/raw/sportsdataverse_pbp"
LAST_SEASON = 2024      # DEV; the TEST season's play-by-play is not downloaded


def period_length(period):
    return PERIOD_SECONDS if period <= 4 else OVERTIME_SECONDS


def _starters(period_events, roster, minutes):
    """Infer one team's five at the start of a period from that period's events."""
    on, seen_in = [], set()
    for e in period_events:
        if e["sub"]:
            player_in, player_out = e["a1"], e["a2"]
            if player_out in roster and player_out not in seen_in and player_out not in on:
                on.append(player_out)
            if player_in in roster:
                seen_in.add(player_in)
        else:
            for p in (e["a1"], e["a2"], e["a3"]):
                if p in roster and p not in seen_in and p not in on:
                    on.append(p)
    on = on[:5]
    if len(on) < 5:                       # silent players: fill with the highest-minute players not yet placed
        for p in sorted(roster, key=lambda q: -minutes.get(q, 0.0)):
            if p not in on and p not in seen_in:
                on.append(p)
            if len(on) == 5:
                break
    return on


def stints(events, home_minutes, away_minutes):
    """events: play-by-play rows of one game (dicts with period, seconds_left, home_score, away_score,
    sub, order, a1, a2, a3), any order. *_minutes: {player_id: box-score minutes} per side.
    Returns None when ten players cannot be placed."""
    rosters = {"home": set(home_minutes), "away": set(away_minutes)}
    minutes = {"home": home_minutes, "away": away_minutes}
    by_period = defaultdict(list)
    for e in events:
        by_period[e["period"]].append(e)
    out, last_score = [], (0, 0)
    for period in sorted(by_period):
        rows = sorted(by_period[period], key=lambda e: (-e["seconds_left"], e["order"]))
        on = {side: _starters(rows, rosters[side], minutes[side]) for side in ("home", "away")}
        if any(len(on[side]) < 5 for side in on):
            return None                    # cannot place ten players: caller falls back to the game equation
        clock, score = float(period_length(period)), last_score

        def close(until, new_score):
            nonlocal clock, score
            if clock - until > 0 or new_score != score:
                margin = (new_score[0] - score[0]) - (new_score[1] - score[1])
                out.append((clock - until, margin, tuple(on["home"]), tuple(on["away"])))
            clock, score = until, new_score

        current = score
        for e in rows:
            if e["home_score"] is not None and e["away_score"] is not None:
                current = (max(current[0], e["home_score"]), max(current[1], e["away_score"]))
            if e["sub"]:
                for side in ("home", "away"):
                    if e["a2"] in on[side] and e["a1"] in rosters[side] and e["a1"] not in on[side]:
                        close(e["seconds_left"], current)
                        on[side][on[side].index(e["a2"])] = e["a1"]
        close(0.0, current)
        last_score = current
    return out


def player_plus_minus(game_stints):
    """{player_id: on-court margin} implied by the stints (home players positive for home margin)."""
    pm = defaultdict(float)
    for _, margin, home, away in game_stints:
        for p in home:
            pm[p] += margin
        for p in away:
            pm[p] -= margin
    return dict(pm)


def load_events(path):
    """Regular-season events from one sportsdataverse season file, grouped by ESPN game ID."""
    import pyarrow.parquet as pq
    columns = ["game_id", "season_type", "period_number", "clock_minutes", "clock_seconds", "home_score", "away_score",
               "type_text", "athlete_id_1", "athlete_id_2", "athlete_id_3", "game_play_number"]
    table = pq.read_table(path, columns=columns).to_pydict()
    games = defaultdict(list)
    ident = lambda v: None if v is None else str(int(v))
    for i in range(len(table["game_id"])):
        if table["season_type"][i] != 2 or table["period_number"][i] is None:
            continue
        minutes_left, seconds_left = table["clock_minutes"][i], table["clock_seconds"][i]
        if minutes_left is None or seconds_left is None:
            continue
        games[str(table["game_id"][i])].append({
            "period": int(table["period_number"][i]), "seconds_left": 60.0 * float(minutes_left) + float(seconds_left),
            "home_score": table["home_score"][i], "away_score": table["away_score"][i],
            "sub": table["type_text"][i] == "Substitution", "order": table["game_play_number"][i] or 0,
            "a1": ident(table["athlete_id_1"][i]), "a2": ident(table["athlete_id_2"][i]), "a3": ident(table["athlete_id_3"][i])})
    return games


def box_minutes(dataset):
    minutes = defaultdict(dict)
    for r in dataset.table("player_box_scores"):
        if r.get("minutes"):
            minutes[r["game_id"], r["side"]][r["player_id"]] = float(r["minutes"])
    return minutes


def from_dataset(dataset, pbp_dir=PBP_DIR, last_season=LAST_SEASON, log=None):
    """game_id -> stints for every game through `last_season` whose stints add up to the final margin.
    The TEST season is never read: its play-by-play file is not part of the frozen download."""
    minutes, by_season, out = box_minutes(dataset), defaultdict(list), {}
    for g in dataset.history():
        if g["season"] <= last_season:
            by_season[g["season"]].append(g)
    for season in sorted(by_season):
        events = load_events(pbp_dir / f"play_by_play_{season + 1}.parquet")
        kept = 0
        for g in by_season[season]:
            rows, key = events.get(str(g["espn_id"])), g["game_id"]
            if not rows or (key, "home") not in minutes or (key, "away") not in minutes:
                continue
            found = stints(rows, minutes[key, "home"], minutes[key, "away"])
            if found is not None and sum(x[1] for x in found) == g["home_score"] - g["away_score"]:
                out[key] = found
                kept += 1
        if log:
            log(f"{season}: {kept}/{len(by_season[season])} games with valid stints")
    return out


def validate(dataset, plus_minus_path):
    """Share of player-games whose stint-implied plus-minus is within 2 points of ESPN's, and the
    mean absolute gap between stint minutes and box-score minutes."""
    import json
    all_stints, minutes = from_dataset(dataset, log=print), box_minutes(dataset)
    close = total = 0
    minute_gap = []
    for line in plus_minus_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["game_id"] not in all_stints or not row.get("plus_minus"):
            continue
        implied, played = player_plus_minus(all_stints[row["game_id"]]), defaultdict(float)
        for seconds, _, home, away in all_stints[row["game_id"]]:
            for p in home + away:
                played[p] += seconds / 60.0
        for side in ("home", "away"):
            for player, value in row["plus_minus"][side].items():
                if player in minutes[row["game_id"], side]:
                    total += 1
                    close += abs(implied.get(player, 0.0) - value) <= 2
                    minute_gap.append(abs(played[player] - minutes[row["game_id"], side][player]))
    print(f"games {len(all_stints)}; plus-minus within 2 points: {close / total:.1%} of {total} player-games; "
          f"mean absolute minutes gap {sum(minute_gap) / len(minute_gap):.2f}")
    if close / total < 0.9:
        raise SystemExit("Stint reconstruction disagrees with ESPN plus-minus")


if __name__ == "__main__":
    from collect_pilot import ROOT
    from nba_dataset import Dataset
    validate(Dataset(), ROOT / "data/processed/plus_minus/plus_minus.jsonl")
