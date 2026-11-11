"""Collect each player's game plus-minus from ESPN's public summary endpoint.

The shared player box scores do not keep plus-minus. This fetches the summary for every franchise
game and keeps only {player_id: plus_minus} per side, with the response's URL, SHA-256 and retrieval
time (the 370 KB bodies are not stored). The summary's event ID, teams and final score must match the
shared schedule. Plus-minus is a postgame box counter: like every box statistic it may only feed
features for later games. Output: data/processed/plus_minus/plus_minus.jsonl, appended as it goes,
so interrupted runs resume.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import hashlib
import json
import time
import urllib.request

from collect_pilot import ABBR, ROOT
from nba_dataset import Dataset

OUT = ROOT / "data/processed/plus_minus/plus_minus.jsonl"
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"


def parse(document, game):
    """{"home": {player_id: plus_minus}, "away": {...}} after checking identity and score."""
    if str(document.get("header", {}).get("id")) != str(game["espn_id"]):
        raise ValueError("event ID mismatch")
    competitors = document["header"]["competitions"][0]["competitors"]
    side_of = {}
    for c in competitors:
        abbr = ABBR.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
        side = c["homeAway"]
        if abbr != game[f"{side}_abbr"] or int(c["score"]) != game[f"{side}_score"]:
            raise ValueError("team or score mismatch")
        side_of[abbr] = side
    out = {"home": {}, "away": {}}
    for group in document.get("boxscore", {}).get("players", []):
        abbr = ABBR.get(group["team"]["abbreviation"], group["team"]["abbreviation"])
        for block in group.get("statistics", []):
            keys = block.get("keys", [])
            if "plusMinus" not in keys:
                continue
            k = keys.index("plusMinus")
            for athlete in block.get("athletes", []):
                stats = athlete.get("stats") or []
                if len(stats) > k and stats[k] not in ("", "--", None):
                    out[side_of[abbr]][str(athlete["athlete"]["id"])] = int(stats[k])
    return out


def fetch(game):
    url = URL.format(game["espn_id"])
    row = {"game_id": game["game_id"], "url": url}
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CS230-academic-data-pilot/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
            row.update(retrieved_at_utc=datetime.now(UTC).isoformat(), sha256=hashlib.sha256(body).hexdigest(),
                       plus_minus=parse(json.loads(body), game))
            return row
        except Exception as exc:  # network errors retry; parse errors are recorded
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (ValueError, KeyError)):
                return row
            time.sleep(2 * (attempt + 1))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    games = Dataset().history()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["game_id"] for line in OUT.read_text(encoding="utf-8").splitlines() if '"plus_minus"' in line}
    todo = [g for g in games if g["game_id"] not in done][:args.limit]
    print(f"{len(done)} done, {len(todo)} to fetch", flush=True)
    with ThreadPoolExecutor(args.workers) as pool, OUT.open("a", encoding="utf-8") as stream:
        for i, row in enumerate(pool.map(fetch, todo), 1):
            stream.write(json.dumps(row) + "\n")
            if i % 250 == 0 or i == len(todo):
                stream.flush()
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
