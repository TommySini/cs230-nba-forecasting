"""Collect official NBA injury reports for every game with a boosting forecast in the chosen splits.

Reuses the pilot's timestamp-checked matcher (report time must be at or before
the game's cutoff and within 6 hours of it). Only injury reports are fetched.
Games and cutoffs come from the shared games table; the game list from the frozen
boosting forecasts. No outcome table is opened.
Official reports are published from the 2018-19 season (--first-season 2018).
Output: data/processed/injury_full/injuries.jsonl, one row per game, checkpointed
so interrupted runs resume from cached raw responses.
"""
import argparse

from collect_pilot import ROOT, Cache, get_injuries, stamp, write_jsonl
from nba_dataset import SPLITS, Dataset

OUT = ROOT / "data/processed/injury_full"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=SPLITS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--first-season", type=int, help="skip earlier seasons (official reports start in 2018-19)")
    parser.add_argument("--last-season", type=int)
    parser.add_argument("--output", default="injuries.jsonl", help="file name inside data/processed/injury_full")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    dataset = Dataset()
    games = {g["game_id"]: g for g in dataset.table("games")}
    forecast_ids = {r["game_id"]: r["split"] for r in dataset.predictions("gradient_boosting", stage="baseline")}
    cache = Cache(offline=args.offline)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.splits:
        ids = sorted((i for i, s in forecast_ids.items() if s == split
                      and (args.first_season is None or games[i]["season"] >= args.first_season)
                      and (args.last_season is None or games[i]["season"] <= args.last_season)),
                     key=lambda i: games[i]["schedule_index"])
        if args.limit:
            ids = ids[:args.limit]
        for i, game_id in enumerate(ids, 1):
            game = games[game_id]
            injuries, attempts = get_injuries(cache, game, stamp(game["cutoff_utc"]))
            rows.append({"game_id": game_id, "split": split, "game_date_et": game["game_date_et"],
                         "home_team": game["home_team"], "away_team": game["away_team"],
                         "home_abbr": game["home_abbr"], "away_abbr": game["away_abbr"],
                         "cutoff_utc": game["cutoff_utc"], "injury_report": injuries, "attempts": len(attempts)})
            if i % 25 == 0 or i == len(ids):
                write_jsonl(OUT / args.output, rows)
                matched = sum(r["injury_report"] is not None for r in rows)
                print(f"{split} {i}/{len(ids)}  matched so far {matched}/{len(rows)}", flush=True)
    write_jsonl(OUT / args.output, rows)


if __name__ == "__main__":
    main()
