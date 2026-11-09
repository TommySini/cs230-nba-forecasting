"""Collect official NBA injury reports for every game in the boosting evaluation splits.

Reuses the pilot's timestamp-checked matcher (report time must be at or before
the game's cutoff and within 6 hours of it). Only injury reports are fetched.
Output: data/processed/injury_full/injuries.jsonl, one row per game, checkpointed
so interrupted runs resume from cached raw responses.
"""
import argparse
import csv
import json
from datetime import timedelta

from collect_pilot import ROOT, Cache, get_injuries, iso, stamp, write_jsonl

OUT = ROOT / "data/processed/injury_full"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["test", "validation"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cutoff-minutes", type=int, default=60)
    args = parser.parse_args()
    games = {}
    with open(ROOT / "data/processed/seasonal/game_history.jsonl", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            games[row["game_id"]] = row
    cache = Cache(offline=args.offline)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.splits:
        with open(ROOT / f"data/processed/gradient_boosting/{split}_predictions.csv", encoding="utf-8", newline="") as stream:
            ids = [r["game_id"] for r in csv.DictReader(stream)]
        if args.limit:
            ids = ids[:args.limit]
        for i, game_id in enumerate(ids, 1):
            game = games[game_id]
            cutoff = stamp(game["scheduled_start_utc"]) - timedelta(minutes=args.cutoff_minutes)
            injuries, attempts = get_injuries(cache, game, cutoff)
            rows.append({"game_id": game_id, "split": split, "game_date_et": game["game_date_et"],
                         "home_team": game["home_team"], "away_team": game["away_team"],
                         "home_abbr": game["home_abbr"], "away_abbr": game["away_abbr"],
                         "cutoff_utc": iso(cutoff), "injury_report": injuries, "attempts": len(attempts)})
            if i % 25 == 0 or i == len(ids):
                write_jsonl(OUT / "injuries.jsonl", rows)
                matched = sum(r["injury_report"] is not None for r in rows)
                print(f"{split} {i}/{len(ids)}  matched so far {matched}/{len(rows)}", flush=True)
    write_jsonl(OUT / "injuries.jsonl", rows)


if __name__ == "__main__":
    main()
