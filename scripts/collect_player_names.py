"""Collect display names for the box-score player IDs, to match names on official injury reports.

Fetches ESPN's public athlete record for each ESPN player ID in the shared player box scores
(seasons from --first-season), keeping raw responses in the download cache. Only the name fields
are kept: no ESPN response reaches the model. Output: data/processed/injury_full/player_names.json
({player_id: {"full": ..., "first": ..., "last": ...}}), rewritten as it goes so runs resume.
"""
import argparse
import json

from collect_pilot import ROOT, Cache, write_json
from nba_dataset import Dataset

OUT = ROOT / "data/processed/injury_full/player_names.json"
URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/{}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=2018)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    dataset = Dataset()
    season = {g["game_id"]: g["season"] for g in dataset.table("games")}
    ids = sorted({r["player_id"] for r in dataset.table("player_box_scores") if season[r["game_id"]] >= args.first_season}, key=int)
    names = json.loads(OUT.read_text()) if OUT.exists() else {}
    cache = Cache(offline=args.offline)
    for i, player_id in enumerate(ids, 1):
        if player_id not in names:
            body, meta = cache.get(URL.format(player_id), "espn_athletes", ".json")
            if body is not None:
                record = json.loads(body)
                if str(record.get("id")) != player_id:
                    raise ValueError(f"Athlete ID mismatch for {player_id}")
                names[player_id] = {"full": record.get("fullName"), "first": record.get("firstName"), "last": record.get("lastName")}
        if i % 100 == 0 or i == len(ids):
            write_json(OUT, names)
            print(f"{i}/{len(ids)} players, {len(names)} named", flush=True)


if __name__ == "__main__":
    main()
