"""Resumable, free-source NBA data feasibility audit. No model training."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class Cache:
    """Retain source bytes, URL, retrieval time, HTTP status and SHA-256."""
    def __init__(self, offline=False, delay=0.35):
        self.offline = offline
        self.delay = delay

    def get(self, url, kind, suffix):
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        base = ROOT / "data/raw" / kind / key
        meta_path = base.with_suffix(".meta.json")
        body_path = base.with_suffix(suffix)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["status"] == 200 and body_path.exists():
                body = body_path.read_bytes()
                if hashlib.sha256(body).hexdigest() != meta["sha256"]:
                    raise ValueError(f"Cache checksum mismatch: {body_path}")
                return body, meta
            if meta["status"] in (400, 403, 404, 410) or self.offline:
                return None, meta
        if self.offline:
            return None, {"url": url, "status": "not_cached"}
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {"url": url, "retrieved_at_utc": iso(datetime.now(UTC))}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CS230-academic-data-pilot/0.1"})
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read()
                meta.update(status=response.status, content_type=response.headers.get("Content-Type"),
                            last_modified=response.headers.get("Last-Modified"),
                            sha256=hashlib.sha256(body).hexdigest(),
                            bytes=len(body), raw_path=str(body_path.relative_to(ROOT)))
                body_path.write_bytes(body)
        except urllib.error.HTTPError as exc:
            body = None
            meta.update(status=exc.code, error=str(exc))
        except (OSError, TimeoutError) as exc:
            body = None
            meta.update(status="network_error", error=str(exc))
        write_json(meta_path, meta)
        time.sleep(self.delay)
        return body, meta


def normalize_game(event, source):
    if event.get("season", {}).get("type") != 2:
        return None
    comp = event["competitions"][0]
    if not comp.get("status", event.get("status", {})).get("type", {}).get("completed"):
        return None
    sides = {x["homeAway"]: x for x in comp["competitors"]}
    if set(sides) != {"home", "away"} or not comp.get("timeValid", True):
        return None
    game = {"game_id": "espn:" + event["id"], "espn_id": event["id"],
            "season": event["season"]["year"] - 1,
            "scheduled_start_utc": iso(stamp(comp["date"])),
            "game_date_et": stamp(comp["date"]).astimezone(ET).date().isoformat(),
            "schedule_source": source}
    for side, item in sides.items():
        team = item["team"]
        game[side + "_team"] = team["displayName"]
        game[side + "_abbr"] = ABBR.get(team["abbreviation"], team["abbreviation"])
        game[side + "_score"] = int(item["score"])
    if game["home_score"] == game["away_score"]:
        return None
    return game


def sample_windows(config):
    """Resolve season-relative windows without confusing January with season start."""
    windows = []
    for season in config["seasons"]:
        year = season["start_year"]
        specs = config.get("sample_windows", [{
            "name": "early_season", "year_offset": 0,
            "start_month_day": config.get("sample_start_month_day"),
            "end_month_day": config.get("sample_end_month_day"),
            "sample_size": season.get("sample_size"),
        }])
        for spec in specs:
            calendar_year = year + spec["year_offset"]
            start = f'{calendar_year}-{spec["start_month_day"]}'
            end = f'{calendar_year}-{spec["end_month_day"]}'
            if not (f"{year}-10-01" <= start <= end < f"{year + 1}-10-01"):
                raise ValueError(f"Invalid window for season {year}: {start} through {end}")
            if spec["sample_size"] <= 0:
                raise ValueError("Sample sizes must be positive")
            windows.append(dict(season=year, name=spec["name"], start=start, end=end,
                                sample_size=spec["sample_size"]))
    for i, left in enumerate(windows):
        for right in windows[i + 1:]:
            if left["season"] == right["season"] and left["start"] <= right["end"] and right["start"] <= left["end"]:
                raise ValueError("Sample windows must not overlap")
    return windows


def output_path(config):
    name = config.get("output_name", "")
    if name and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("output_name must be a simple lowercase directory name")
    return ROOT / "data/processed" / name


def collect_games(cache, config):
    games, failures = {}, []
    windows = sample_windows(config)
    for season in config["seasons"]:
        year = season["start_year"]
        day = datetime.fromisoformat(f'{year}-{config["history_start_month_day"]}')
        end = datetime.fromisoformat(max(w["end"] for w in windows if w["season"] == year))
        while day <= end:
            url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=" + day.strftime("%Y%m%d") + "&limit=100"
            body, meta = cache.get(url, "scoreboards", ".json")
            if body is None:
                failures.append(meta)
            else:
                for event in json.loads(body).get("events", []):
                    game = normalize_game(event, meta["raw_path"])
                    if game:
                        games[game["game_id"]] = game
            if day.day % 10 == 0:
                print(f"Schedules through {day.date()}: {len(games)} completed regular-season games", flush=True)
            day += timedelta(days=1)
    failure_path = output_path(config) / "schedule_failures.json"
    if failures:
        write_json(failure_path, failures)
        raise RuntimeError(f"Incomplete schedule history; resolve {failure_path} before deriving features")
    failure_path.unlink(missing_ok=True)
    return sorted(games.values(), key=lambda x: (x["scheduled_start_utc"], x["game_id"]))


def select_sample(games, config):
    rng = random.Random(config["seed"])
    selected = []
    for window in sample_windows(config):
        candidates = [g for g in games if g["season"] == window["season"] and
                      window["start"] <= g["game_date_et"] <= window["end"]]
        if len(candidates) < window["sample_size"]:
            raise ValueError(f"Insufficient completed games for {window}: {len(candidates)}")
        selected.extend(rng.sample(candidates, window["sample_size"]))
    return sorted(selected, key=lambda x: (x["scheduled_start_utc"], x["game_id"]))


def prior_features(game, games, side):
    # Earlier ET calendar dates only: a prior game's scheduled start alone
    # does not establish that it finished before this game's cutoff.
    team = game[side + "_abbr"]
    history = [g for g in games if g["season"] == game["season"] and
               g["game_date_et"] < game["game_date_et"] and team in (g["home_abbr"], g["away_abbr"])]
    margins = [(g["home_score"] - g["away_score"]) * (1 if g["home_abbr"] == team else -1) for g in history]
    days = (datetime.fromisoformat(game["game_date_et"]) - datetime.fromisoformat(history[-1]["game_date_et"])).days if history else None
    return {"previous_games": len(history), "prior_game_ids": [g["game_id"] for g in history],
            "days_since_previous_game": days, "back_to_back": days == 1 if days is not None else None,
            "season_win_rate": sum(m > 0 for m in margins) / len(margins) if margins else None,
            "last_5_games": len(margins[-5:]),
            "last_5_win_rate": sum(m > 0 for m in margins[-5:]) / len(margins[-5:]) if margins else None,
            "last_5_point_differential": sum(margins[-5:]) / len(margins[-5:]) if margins else None}


REPORT_TIME = r"Injury\s+Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s*(AM|PM)"
GAME_HEADER = re.compile(r"(?:(\d{2}/\d{2}/\d{4})\s+)?(?:(\d{2}:\d{2})\s*\(ET\)\s*)?([A-Z]{2,3}@[A-Z]{2,3})")


def parse_injury(body):
    reader = PdfReader(io.BytesIO(body))
    pages = [" ".join((p.extract_text() or "").split()) for p in reader.pages]
    return parse_injury_pages(pages)


def parse_injury_pages(pages):
    found = re.search(REPORT_TIME, pages[0])
    if not found:
        raise ValueError("Missing internal injury-report timestamp")
    report_time = datetime.strptime(" ".join(found.groups()), "%m/%d/%y %I:%M %p").replace(tzinfo=ET)
    cleaned = []
    for page in pages:
        page = re.sub(REPORT_TIME, "", page)
        page = re.sub(r"Page\s+\d+\s+of\s+\d+", "", page)
        page = re.sub(r"Game\s+Date\s+Game\s+Time\s+Matchup\s+Team\s+Player\s+Name\s+Current\s+Status\s+Reason", "", page)
        cleaned.append(page)
    text = " ".join(cleaned)
    headers = list(GAME_HEADER.finditer(text))
    date = None
    blocks = []
    for i, match in enumerate(headers):
        if match.group(1):
            date = datetime.strptime(match.group(1), "%m/%d/%Y").date().isoformat()
        blocks.append({"game_date_et": date, "matchup": match.group(3),
                       "text": " ".join(text[match.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)].split())})
    return iso(report_time), blocks


def injury_urls(cutoff):
    local = cutoff.astimezone(ET).replace(minute=0, second=0, microsecond=0)
    # Probe a bounded window. Legacy filenames can contain a report 30 minutes
    # later than the filename suggests, so every candidate is checked internally.
    for hours_back in range(4):
        t = local - timedelta(hours=hours_back)
        for fmt in ("%I%p", "%I_00%p", "%I_30%p"):
            yield f'https://ak-static.cms.nba.com/referee/injury/Injury-Report_{t:%Y-%m-%d}_{t.strftime(fmt)}.pdf'


def get_injuries(cache, game, cutoff):
    attempts, eligible = [], []
    expected = game["away_abbr"] + "@" + game["home_abbr"]
    for url in injury_urls(cutoff):
        body, meta = cache.get(url, "injuries", ".pdf")
        attempt = {"url": url, "status": meta["status"]}
        if body is not None:
            try:
                report_time, blocks = parse_injury(body)
                attempt["report_time_utc"] = report_time
                if stamp(report_time) > cutoff:
                    attempt["decision"] = "after_cutoff"
                elif cutoff - stamp(report_time) > timedelta(hours=6):
                    attempt["decision"] = "stale"
                else:
                    matches = [b for b in blocks if b["game_date_et"] == game["game_date_et"] and b["matchup"] == expected]
                    if len(matches) == 1 and matches[0]["text"]:
                        block = matches[0]
                        attempt["decision"] = "eligible"
                        eligible.append(dict(block, report_time_utc=report_time, source=meta,
                                             not_yet_submitted="NOT YET SUBMITTED" in block["text"].upper(),
                                             teams_with_rows=[game[s + "_team"] for s in ("home", "away") if game[s + "_team"] in block["text"]]))
                    else:
                        attempt["decision"] = "matchup_not_found_or_ambiguous"
            except Exception as exc:
                attempt["decision"] = "parse_error"
                attempt["error"] = str(exc)
        attempts.append(attempt)
        # A valid candidate within 60 minutes of cutoff is sufficient for this
        # audit. We explicitly do NOT claim it is the latest possible snapshot.
        if eligible and cutoff - stamp(eligible[-1]["report_time_utc"]) <= timedelta(hours=1):
            break
    if eligible:
        return max(eligible, key=lambda x: x["report_time_utc"]), attempts
    return None, attempts


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def preview_decision(article, game, cutoff):
    if not article:
        return "missing_article"
    if article.get("type", "").lower() != "preview":
        return "rejected_not_preview"
    if str(article.get("gameId", "")) != game["espn_id"]:
        return "rejected_game_mismatch"
    for field in ("published", "lastModified"):
        if not article.get(field):
            return "rejected_missing_timestamp"
        try:
            dt = stamp(article[field])
            if dt.tzinfo is None:
                return "rejected_missing_timezone"
            if dt > cutoff:
                return "rejected_after_cutoff"
        except ValueError:
            return "rejected_invalid_timestamp"
    if not article.get("story"):
        return "rejected_empty_text"
    return "candidate_requires_content_audit"


def get_preview(cache, game, cutoff):
    url = f'https://cdn.espn.com/core/nba/preview?xhr=1&gameId={game["espn_id"]}'
    body, meta = cache.get(url, "previews", ".json")
    result = {"source": meta, "decision": "fetch_failed", "included_in_inputs": False}
    if body:
        article = json.loads(body).get("gamepackageJSON", {}).get("article", {})
        result.update(decision=preview_decision(article, game, cutoff),
                      article_type=article.get("type"), published=article.get("published"),
                      last_modified=article.get("lastModified"))
        if result["decision"] == "candidate_requires_content_audit":
            parser = TextOnly()
            parser.feed(article["story"])
            result["candidate_text"] = " ".join(" ".join(parser.parts).split())
    return result


def get_notes(cache, game, processed_dir=None):
    # One team's known dated archive pattern, explicitly not a leaguewide audit.
    if "CLE" not in (game["home_abbr"], game["away_abbr"]):
        return {"decision": "not_attempted_no_adapter", "included_in_inputs": False}
    day = datetime.fromisoformat(game["game_date_et"])
    pair = (game["away_abbr"] + game["home_abbr"]).lower()
    url = f'https://cdn.nba.com/teams/uploads/sites/1610612739/{day:%Y/%m/%y%m%d}-{pair}-notes.pdf'
    body, meta = cache.get(url, "team_notes", ".pdf")
    result = {"source": meta, "decision": "fetch_failed", "included_in_inputs": False}
    if body:
        try:
            reader = PdfReader(io.BytesIO(body))
            text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            path = (processed_dir or ROOT / "data/processed") / "narrative_candidates" / (game["espn_id"] + "-CLE.txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            result.update(decision="candidate_requires_matchup_and_publication_audit", pages=len(reader.pages),
                          text_path=str(path.relative_to(ROOT)),
                          pdf_metadata={str(k): str(v) for k, v in (reader.metadata or {}).items()})
        except Exception as exc:
            result.update(decision="parse_error", error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Rebuild entirely from saved source responses")
    parser.add_argument("--stage", choices=("schedules", "audit"), default="audit")
    parser.add_argument("--limit", type=int, help="Audit only the first N sampled games for debugging")
    parser.add_argument("--config", default="config/pilot.json", help="Config path relative to repository root")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    processed_dir = output_path(config)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(offline=args.offline)
    games = collect_games(cache, config)
    selected = select_sample(games, config)
    write_jsonl(processed_dir / "game_history.jsonl", games)
    write_jsonl(processed_dir / "sample_manifest.jsonl", selected)
    if args.stage == "schedules":
        print(f"Saved {len(selected)} sampled games and {len(games)} historical results", flush=True)
        return
    if args.limit:
        selected = selected[:args.limit]
    inputs, labels, audit = [], [], []
    for i, game in enumerate(selected, 1):
        cutoff = stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"])
        injuries, attempts = get_injuries(cache, game, cutoff)
        preview = get_preview(cache, game, cutoff)
        notes = get_notes(cache, game, processed_dir)
        record = {k: game[k] for k in ("game_id", "season", "scheduled_start_utc", "game_date_et", "home_team", "away_team", "home_abbr", "away_abbr")}
        record.update(cutoff_utc=iso(cutoff),
                      home_features=prior_features(game, games, "home"),
                      away_features=prior_features(game, games, "away"),
                      injury_report=injuries, narrative_text=None,
                      model_contamination_status="not_checked_model_not_selected")
        inputs.append(record)
        labels.append({"game_id": game["game_id"], "home_win": int(game["home_score"] > game["away_score"]),
                       "home_score": game["home_score"], "away_score": game["away_score"]})
        audit.append({"game_id": game["game_id"], "season": game["season"],
                      "injury_available": injuries is not None,
                      "injury_not_yet_submitted": injuries["not_yet_submitted"] if injuries else None,
                      "injury_attempts": attempts, "preview": preview, "team_notes": notes})
        # Checkpoint after every game; interrupted runs reuse all raw responses.
        write_jsonl(processed_dir / "inputs.jsonl", inputs)
        write_jsonl(processed_dir / "labels.jsonl", labels)
        write_jsonl(processed_dir / "audit.jsonl", audit)
        print(f'[{i}/{len(selected)}] {game["game_date_et"]} {game["away_abbr"]}@{game["home_abbr"]}: injury={injuries is not None}; preview={preview["decision"]}; notes={notes["decision"]}', flush=True)
    summary = {"games": len(inputs), "seasons": dict(Counter(str(g["season"]) for g in inputs)),
               "teams": len({g[s + "_abbr"] for g in inputs for s in ("home", "away")}),
               "injury_reports_matched": sum(a["injury_available"] for a in audit),
               "injury_not_yet_submitted": sum(bool(a["injury_not_yet_submitted"]) for a in audit),
               "preview_decisions": dict(Counter(a["preview"]["decision"] for a in audit)),
               "team_note_decisions": dict(Counter(a["team_notes"]["decision"] for a in audit)),
               "narrative_inputs_approved": 0,
               "scope": config.get("scope", "November 1-10 in each configured season; early-season feasibility sample, not a train/test benchmark"),
               "injury_selection": "Verified internal timestamp <= cutoff; bounded URL probes, not guaranteed latest snapshot",
               "config": config, "config_path": args.config, "windows": []}
    for window in sample_windows(config):
        subset = [r for r in inputs if r["season"] == window["season"] and window["start"] <= r["game_date_et"] <= window["end"]]
        summary["windows"].append(dict(window, games=len(subset),
                                      injury_reports_matched=sum(r["injury_report"] is not None for r in subset)))
    write_json(processed_dir / "coverage.json", summary)
    with (processed_dir / "game_audit.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game_id", "date_et", "away", "home", "cutoff_utc", "injury_report_utc",
                         "injury_not_yet_submitted", "preview_decision", "team_notes_decision"])
        for record, row in zip(inputs, audit):
            writer.writerow([record["game_id"], record["game_date_et"], record["away_abbr"], record["home_abbr"],
                             record["cutoff_utc"], (record["injury_report"] or {}).get("report_time_utc"),
                             row["injury_not_yet_submitted"], row["preview"]["decision"], row["team_notes"]["decision"]])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
