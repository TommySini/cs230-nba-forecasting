"""Collect official NBA injury reports for every game with a boosting forecast in the chosen splits.

Reuses the pilot's timestamp-checked matcher (report time must be at or before
the game's cutoff and within 6 hours of it). Only injury reports are fetched.
Games come from data/processed/gradient_boosting/predictions.jsonl; schedules from
the histories listed in config/gradient_boosting.json.
Output: data/processed/injury_full/injuries.jsonl, one row per game, checkpointed
so interrupted runs resume from cached raw responses.
"""
import argparse
import json
from datetime import timedelta

from collect_pilot import ROOT, Cache, get_injuries, iso, stamp, write_jsonl
from evaluate_orthogonality import SPLITS, boosting_forecasts, read_rows

OUT = ROOT / "data/processed/injury_full"
BOOSTING_CONFIG = ROOT / "config/gradient_boosting.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=SPLITS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads(BOOSTING_CONFIG.read_text())
    games = {g["game_id"]: g for source in config["histories"] for g in read_rows(ROOT / source["path"])}
    forecasts = boosting_forecasts(args.splits)
    cache = Cache(offline=args.offline)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.splits:
        ids = sorted((i for i, f in forecasts.items() if f["split"] == split), key=lambda i: (games[i]["scheduled_start_utc"], i))
        if args.limit:
            ids = ids[:args.limit]
        for i, game_id in enumerate(ids, 1):
            game = games[game_id]
            cutoff = stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"])
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
