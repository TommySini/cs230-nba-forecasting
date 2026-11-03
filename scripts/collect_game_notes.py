"""Schedule-defined rich-note discovery; verified archives remain content candidates."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from urllib.parse import urlencode
import time

from pypdf import PdfReader
from audit_game_notes import archive_lookup, fetch_archive_version, decode_archive_body
from collect_pilot import Cache, ROOT, iso, stamp, write_json, write_jsonl

CONFIG = ROOT / "config/game_notes_collection.json"
DATA = ROOT / "data/processed/game_notes_collection"
REPORT = ROOT / "reports/game_notes_collection"


class RetryingCache(Cache):
    def get(self, url, kind, suffix):
        for attempt in range(3):
            body, source = super().get(url, kind, suffix)
            if self.offline or source.get("status") not in ("network_error", 429, 500, 502, 503, 504):
                return body, source
            failed = ROOT / "data/raw/collection_failed_attempts" / (hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest() + ".json")
            write_json(failed, source)
            time.sleep(.5 * (attempt + 1))
        return body, source


def identify(game):
    return {k: game[k] for k in ("game_id", "game_date_et", "scheduled_start_utc", "season", "home_abbr", "away_abbr")}


def cutoff(game, config):
    return iso(stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"]))


def discover_dated(game, cache, config):
    day = datetime.fromisoformat(game["game_date_et"])
    pair = (game["away_abbr"] + game["home_abbr"]).lower()
    url = f"https://cdn.nba.com/teams/uploads/sites/1610612739/{day:%Y/%m/%y%m%d}-{pair}-notes.pdf"
    body, source = cache.get(url, "team_notes", ".pdf")
    row = dict(identify(game), team="CLE", kind="dated", url=url, source=source, cutoff_utc=cutoff(game, config))
    row["archive_check"] = archive_lookup(cache, url, row["cutoff_utc"])
    if row["archive_check"]["decision"] == "closest_snapshot_after_cutoff":
        # Availability is nearest, not necessarily nearest BEFORE the cutoff.
        # Search the bounded CDX interval before concluding no eligible capture.
        query = "https://web.archive.org/cdx/search/cdx?" + urlencode({
            "url": url, "output": "json", "filter": "statuscode:200",
            "from": (stamp(row["cutoff_utc"]) - timedelta(days=2)).strftime("%Y%m%d%H%M%S"),
            "to": stamp(row["cutoff_utc"]).strftime("%Y%m%d%H%M%S"),
            "collapse": "digest", "fl": "timestamp,original,digest"})
        indexed, meta = cache.get(query, "archive_indexes", ".json")
        row["before_cutoff_index"] = {"source": meta}
        if indexed:
            parsed = json.loads(indexed)
            captures = [dict(zip(parsed[0], r)) for r in parsed[1:]] if parsed else []
            captures = [c for c in captures if c["timestamp"] <= stamp(row["cutoff_utc"]).strftime("%Y%m%d%H%M%S") and c["original"] == url]
            row["before_cutoff_index"]["captures"] = len(captures)
            if captures:
                chosen = max(captures, key=lambda c: c["timestamp"])
                row["nearest_availability_check"] = row["archive_check"]
                row["archive_check"] = {"source": meta, "decision": "pre_cutoff_snapshot_requires_bytes_review", "snapshot": {
                    "url": f"https://web.archive.org/web/{chosen['timestamp']}/{url}", "timestamp": chosen["timestamp"], "status": "200", "available": True}}
    return row


def discover_rolling(item, games, cache, config):
    team, slug = item
    url = f"https://www.nba.com/gamenotes/{slug}.pdf"
    query = "https://web.archive.org/cdx/search/cdx?" + urlencode({
        "url": url, "output": "json", "filter": "statuscode:200", "from": config["archive_from"],
        "to": config["archive_to"], "collapse": "digest", "fl": "timestamp,original,digest"})
    body, source = cache.get(query, "archive_indexes", ".json")
    report = {"team": team, "url": url, "source": source, "captures": 0, "candidate_games": 0}
    rows = []
    if body is None:
        return rows, report
    parsed = json.loads(body)
    captures = [dict(zip(parsed[0], r)) for r in parsed[1:]] if parsed else []
    report["captures"] = len(captures)
    # The earliest capture of each distinct version is retained by CDX collapse.
    # Match only to the next scheduled team game, before its cutoff and within 36h.
    # A document-header review still has to confirm this is that game's packet.
    team_games = sorted([g for g in games if team in (g["home_abbr"], g["away_abbr"])], key=lambda g: g["scheduled_start_utc"])
    for capture in captures:
        when = datetime.strptime(capture["timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        eligible = [g for g in team_games if timedelta(0) <= stamp(cutoff(g, config)) - when <= timedelta(hours=config["rolling_max_age_hours"])]
        if not eligible:
            continue
        game = eligible[0]
        original = capture["original"]
        snapshot = {"url": f"https://web.archive.org/web/{capture['timestamp']}/{original}", "timestamp": capture["timestamp"], "status": "200", "available": True}
        rows.append(dict(identify(game), team=team, kind="rolling_archive", url=original,
                         source={"sha256": None}, cutoff_utc=cutoff(game, config),
                         archive_check={"source": source, "decision": "pre_cutoff_snapshot_requires_bytes_review", "snapshot": snapshot}))
    report["candidate_games"] = len({r["game_id"] for r in rows})
    return rows, report


def fetch_candidate(row, cache, games):
    row = dict(row)
    fetch_row = dict(row, source={"sha256": None, **row["source"]})
    try:
        row["archive_version"] = fetch_archive_version(cache, fetch_row, ".pdf", games[row["game_id"]])
    except Exception as exc:
        row.update(decision="archive_replay_parse_failed", error=str(exc))
        return row
    archived = row["archive_version"]
    row["decision"] = "no_verified_pre_cutoff_archive"
    if archived.get("publication_time_verified"):
        source = archived["source"]
        body = decode_archive_body((ROOT / source["raw_path"]).read_bytes())
        try:
            reader = PdfReader(io.BytesIO(body))
            # Keep the exact front-page text. No raw feeds, labels or synthesized facts.
            text = reader.pages[0].extract_text() or ""
            path = DATA / "candidates" / f"{row['game_id'].split(':')[-1]}-{row['team']}-{source['sha256'][:12]}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            row.update(decision="pre_cutoff_archive_requires_header_and_content_review", pages=len(reader.pages),
                       text_path=str(path.relative_to(ROOT)), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       words=len(text.split()))
        except Exception as exc:
            row.update(decision="pdf_extraction_failed", error=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--stage", choices=("discover", "fetch", "all"), default="all")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    games = [json.loads(x) for x in (ROOT / config["history_path"]).read_text().splitlines()]
    by_id = {g["game_id"]: g for g in games}
    cache = RetryingCache(offline=args.offline)
    DATA.mkdir(parents=True, exist_ok=True)
    if args.stage in ("discover", "all"):
        rows, indexes = [], []
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            dated = [g for g in games if "CLE" in (g["home_abbr"], g["away_abbr"])]
            for i, row in enumerate(pool.map(lambda g: discover_dated(g, cache, config), dated), 1):
                rows.append(row)
                if i % 20 == 0:
                    print(f"Dated discovery {i}/{len(dated)}", flush=True)
            for found, report in pool.map(lambda item: discover_rolling(item, games, cache, config), config["rolling_teams"].items()):
                rows.extend(found)
                indexes.append(report)
                print(f"{report['team']}: {report['captures']} distinct archived versions, {report['candidate_games']} possible schedule matches", flush=True)
        write_jsonl(DATA / "discovery.jsonl", rows)
        write_json(REPORT / "rolling_indexes.json", indexes)
    else:
        rows = [json.loads(x) for x in (DATA / "discovery.jsonl").read_text().splitlines()]
    if args.stage in ("fetch", "all"):
        results = []
        # Discovery assigns each rolling capture only to the next team game.
        rows.sort(key=lambda r: (r["game_date_et"], r["game_id"], r["team"], r["url"]))
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            for i, row in enumerate(pool.map(lambda r: fetch_candidate(r, cache, by_id), rows), 1):
                results.append(row)
                if i % 20 == 0 or i == len(rows):
                    write_jsonl(DATA / "audit.jsonl", results)
                    print(f"Archive fetch {i}/{len(rows)}: {dict(Counter(r['decision'] for r in results))}", flush=True)
        summary = {"scope": "All cached regular-season schedules, dated CLE adapter and archive indexes for 30 teams; coverage is source-selected, not representative.",
                   "candidates": len(results), "decisions": dict(Counter(r["decision"] for r in results)),
                   "verified_by_publisher": dict(Counter(r["team"] for r in results if r.get("text_path"))),
                   "verified_unique_games": len({r["game_id"] for r in results if r.get("text_path")}),
                   "approved_for_model": 0}
        write_json(REPORT / "coverage.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
