"""Collect every recoverable pregame team game-notes document, 2012-13 through 2025-26.

Sources, each kept with its provenance tier:
  * archive_proven: Internet Archive captures (rolling `nba.com/gamenotes/{team}.pdf`,
    including query-string variants, and Cleveland's dated PDFs) whose replay time is
    before the game's cutoff (tipoff minus 60 minutes);
  * site_timestamped: Cleveland's dated PDFs fetched from the team CDN (Last-Modified
    before the cutoff) and Knicks/Blazers HTML game-notes posts (published before the
    cutoff), per the 2026-11-01 decision to accept publisher timestamps.
Every document must also pass header checks (this game's date and both teams; no
contradicting record or game number) and the postgame screen (no result of this game).

Stages: index (archive CDX listings), match (candidates per team-game), fetch (archive
replays, one at a time with backoff; CDN files), html (Knicks/Blazers posts), review.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import html
import http.client
import io
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request

from collect_pilot import Cache, ROOT, iso, stamp, write_json, write_jsonl
from postgame_screen import postgame_flags

CONFIG_PATH = ROOT / "config/notes_v2.json"
DATA = ROOT / "data/processed/notes_v2"
REPLAY = re.compile(r"https://web\.archive\.org/web/(\d{14})id_/(.+)")
DATED_NAME = re.compile(r"/(\d{6})-([a-z]{2,4})([a-z]{3})-notes\.pdf$")


def read_rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def utc(timestamp14):
    return datetime.strptime(timestamp14, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def load_history(config):
    games, teams = {}, set(config["slugs"].values())
    for path in config["history_paths"]:
        if not (ROOT / path).exists():
            print(f"Missing history {path}; matching without it", flush=True)
            continue
        for game in read_rows(ROOT / path):
            if {game["home_abbr"], game["away_abbr"]} <= teams:  # drops All-Star and exhibition teams
                games[game["game_id"]] = game
    return sorted(games.values(), key=lambda g: (g["scheduled_start_utc"], g["game_id"]))


def records_before(history):
    """Each team's (wins, losses) before every game, from completed results only."""
    record, before = Counter(), {}
    for g in history:
        for side in ("home", "away"):
            key = (g["season"], g[side + "_abbr"])
            before[g["game_id"], side] = (record[key, "w"], record[key, "l"])
        home_won = g["home_score"] > g["away_score"]
        record[(g["season"], g["home_abbr"]), "w" if home_won else "l"] += 1
        record[(g["season"], g["away_abbr"]), "l" if home_won else "w"] += 1
    return before


def cutoff_of(game, config):
    return stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"])


# ---------------------------------------------------------------- index and match

def patient_get(cache, url, kind, suffix, config, attempts=5):
    """The archive's index server rate-limits with 503s; wait and retry instead of failing."""
    for attempt in range(attempts):
        body, meta = cache.get(url, kind, suffix)
        if body is not None or meta.get("status") in (400, 403, 404, 410):
            return body, meta
        print(f"  {meta.get('status')} from {url[:90]}; retrying", flush=True)
        time.sleep(config["archive_backoff_seconds"] * (attempt + 1))
    raise RuntimeError(f"Archive request kept failing: {meta}")


def fetch_cdx(cache, config):
    """Archive listings: saved snapshots (with checksum) plus any live CDX queries."""
    rows = []
    for snapshot in config.get("cdx_snapshots", []):
        path = ROOT / snapshot["path"]
        parsed = json.loads(path.read_text())
        source = {"path": snapshot["path"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        rows += [dict(zip(parsed[0], r), cdx_source=source) for r in parsed[1:]]
    for query in config["cdx_queries"]:
        count_url = re.sub(r"&(output|fl|filter)=[^&]*", "", query) + "&showNumPages=true"
        pages, meta = patient_get(cache, count_url, "archive_cdx", ".txt", config)
        for page in range(int(pages.decode().strip() or 1)):
            body, meta = patient_get(cache, f"{query}&page={page}", "archive_cdx", ".json", config)
            parsed = json.loads(body) if body.strip() else []
            rows += [dict(zip(parsed[0], r), cdx_source=meta["raw_path"]) for r in parsed[1:]]
    return rows


def slug_team(original, when, config):
    match = re.search(r"/gamenotes/+([a-z0-9]+)\.pdf", original.lower())
    if not match:
        return None
    slug = match.group(1)
    if slug == "hornets":
        rule = config["hornets_slug"]
        if when < stamp(rule["NOP_until"] + "T00:00:00Z"):
            return "NOP"
        return "CHA" if when >= stamp(rule["CHA_from"] + "T00:00:00Z") else None
    return config["slugs"].get(slug)


def rolling_versions(captures, config):
    """First successful capture of each distinct file per team; placeholders shared across teams dropped."""
    first, teams_by_digest = {}, defaultdict(set)
    for c in sorted(captures, key=lambda c: c["timestamp"]):
        if c["statuscode"] != "200":
            continue
        when = utc(c["timestamp"])
        team = slug_team(c["original"], when, config)
        if not team:
            continue
        teams_by_digest[c["digest"]].add(team)
        first.setdefault((team, c["digest"]), dict(c, team=team, when=when))
    return [v for (team, digest), v in first.items() if len(teams_by_digest[digest]) == 1]


def match_rolling(versions, history, config):
    """Assign each team game the latest new version first captured after the previous game and before its cutoff."""
    by_team, games_by_team = defaultdict(list), defaultdict(list)
    for v in versions:
        by_team[v["team"]].append(v)
    for g in history:
        for side in ("home", "away"):
            games_by_team[g[side + "_abbr"]].append(g)
    candidates = []
    for team, games in games_by_team.items():
        previous = None
        for g in games:
            cutoff = cutoff_of(g, config)
            low = cutoff - timedelta(hours=config["max_window_hours"])
            if previous:
                low = max(low, stamp(previous["scheduled_start_utc"]) + timedelta(hours=config["previous_game_buffer_hours"]))
            inside = [v for v in by_team.get(team, []) if low < v["when"] <= cutoff]
            if inside:
                v = max(inside, key=lambda v: v["when"])
                candidates.append({"game_id": g["game_id"], "team": team, "kind": "rolling_archive", "tier": "archive_proven",
                                   "url": v["original"], "capture_timestamp": v["timestamp"], "digest": v["digest"],
                                   "replay_url": f"https://web.archive.org/web/{v['timestamp']}id_/{v['original']}",
                                   "cdx_source": v["cdx_source"], "cutoff_utc": iso(cutoff), "alternates": len(inside) - 1})
            previous = g
    return candidates


def dated_game(original, history_index):
    match = DATED_NAME.search(original.lower())
    if not match:
        return None
    day = datetime.strptime(match.group(1), "%y%m%d").date().isoformat()
    away, home = match.group(2).upper(), match.group(3).upper()
    return history_index.get((day, away, home))


def match_dated(captures, history, config):
    index = {(g["game_date_et"], g["away_abbr"], g["home_abbr"]): g for g in history}
    found = {}
    for c in captures:
        game = dated_game(c["original"], index)
        if not game or c["statuscode"] != "200":
            continue
        cutoff = cutoff_of(game, config)
        if utc(c["timestamp"]) <= cutoff:
            key = game["game_id"]
            if key not in found or c["timestamp"] > found[key]["capture_timestamp"]:
                found[key] = {"game_id": key, "team": config["dated_team"], "kind": "dated_archive", "tier": "archive_proven",
                              "url": c["original"], "capture_timestamp": c["timestamp"], "digest": c["digest"],
                              "replay_url": f"https://web.archive.org/web/{c['timestamp']}id_/{c['original']}",
                              "cdx_source": c["cdx_source"], "cutoff_utc": iso(cutoff), "alternates": 0}
    live = []
    for g in history:
        if config["dated_team"] in (g["home_abbr"], g["away_abbr"]) and g["season"] >= config["dated_first_season"]:
            day = datetime.fromisoformat(g["game_date_et"])
            url = config["dated_url"].format(day=day, pair=(g["away_abbr"] + g["home_abbr"]).lower())
            live.append({"game_id": g["game_id"], "team": config["dated_team"], "kind": "dated_live",
                         "tier": "site_timestamped", "url": url, "cutoff_utc": iso(cutoff_of(g, config))})
    return list(found.values()), live


# ---------------------------------------------------------------- fetch

def replay_path(url):
    return ROOT / "data/raw/archive_versions" / hashlib.sha256(url.encode()).hexdigest()[:24]


def fetch_replay(url, config):
    """Download one archive replay (cached by URL, same layout as the original collector)."""
    base = replay_path(url)
    meta_path, body_path = base.with_suffix(".meta.json"), base.with_suffix(".pdf")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("status") == 200 or meta.get("status") in (403, 404, 410):
            return meta, False
    meta = {"url": url, "retrieved_at_utc": iso(datetime.now(timezone.utc))}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": config["user_agent"]})
        with urllib.request.urlopen(request, timeout=40) as response:
            body = response.read()
            meta.update(status=response.status, final_url=response.url,
                        memento_datetime=response.headers.get("Memento-Datetime"),
                        content_type=response.headers.get("Content-Type"),
                        sha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
                        raw_path=str(body_path.relative_to(ROOT)))
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes(body)
    except urllib.error.HTTPError as exc:
        meta.update(status=exc.code, error=str(exc))
    except (OSError, TimeoutError, http.client.HTTPException) as exc:  # includes truncated bodies
        meta.update(status="network_error", error=str(exc))
    write_json(meta_path, meta)
    return meta, True


def fetch_all(candidates, config):
    cache = Cache(delay=0.3)
    for c in candidates:
        if c["kind"] == "dated_live":
            cache.get(c["url"], "team_notes", ".pdf")
    # Newest games first: recent seasons are the scarcest and matter most for evaluation.
    pending = sorted((c for c in candidates if c["kind"] != "dated_live"), key=lambda c: c["cutoff_utc"], reverse=True)
    failures, done = 0, 0
    for i, c in enumerate(pending, 1):
        for attempt in range(3):
            meta, networked = fetch_replay(c["replay_url"], config)
            if meta.get("status") not in ("network_error", 429, 500, 502, 503, 504):
                break
            print(f"  {meta.get('status')} on {c['replay_url']}; backing off", flush=True)
            time.sleep(config["archive_backoff_seconds"] * (attempt + 1))
        failures = failures + 1 if meta.get("status") not in (200, 403, 404, 410) else 0
        if failures >= config["max_consecutive_failures"]:
            print(f"Stopping after {failures} consecutive failures; rerun to resume", flush=True)
            return False
        if networked:
            done += 1
            time.sleep(config["archive_gap_seconds"])
        if i % 50 == 0:
            print(f"Archive replays {i}/{len(pending)} ({done} downloaded this run)", flush=True)
    return True


# ---------------------------------------------------------------- html posts

def strip_html(markup):
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", markup or "")
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6])>", "\n", text)
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


def blocks_text(blocks):
    """Article-owned content blocks only (paragraphs, headings, tables), in order."""
    parts = []
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else None
        parts.append(text if text else strip_html(block.get("html", "") if isinstance(block, dict) else str(block)))
    return "\n".join(p.strip() for p in parts if p and p.strip())


def fetch_html_posts(config):
    cache, posts = Cache(delay=0.5), []
    for team, source in config["html_sources"].items():
        for page in range(1, config["html_max_pages"] + 1):
            body, meta = cache.get(source["url"].format(page=page), "html_notes_api", ".json")
            if body is None:
                break
            data = json.loads(body)
            items = data.get("items") or data.get("results") or []
            if not items:
                break
            for item in items:
                if source["title_pattern"].lower() in (item.get("title") or "").lower():
                    posts.append({"team": team, "title": item.get("title"), "permalink": item.get("permalink"),
                                  "published_utc": item.get("date"), "modified_utc": item.get("modified"),
                                  "text": blocks_text(item.get("contentExpanded") or []), "source": meta})
    return posts


MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def post_game(post, history, config):
    """Map a post to its game: by the title's date (Knicks titles carry the real date; slugs do not),
    else the publishing team's next game within 48 hours whose opponent the title names (Blazers)."""
    team = post["team"]
    match = re.search(r"([A-Z][a-z]+)\.? (\d{1,2}),? (\d{4})", post["title"] or "")
    month = MONTHS.get(match.group(1).lower()[:3]) if match else None
    if month:
        day = f"{int(match.group(3)):04}-{month:02}-{int(match.group(2)):02}"
        found = [g for g in history if g["game_date_et"] == day and team in (g["home_abbr"], g["away_abbr"])]
        return found[0] if len(found) == 1 else None
    published, title = stamp(post["published_utc"]), (post["title"] or "").lower()
    for g in history:
        if team not in (g["home_abbr"], g["away_abbr"]) or not published < stamp(g["scheduled_start_utc"]) <= published + timedelta(hours=48):
            continue
        opponent = g["away_team"] if g["home_abbr"] == team else g["home_team"]
        return g if opponent.split()[-1].lower() in title else None
    return None


# ---------------------------------------------------------------- review

def normalized(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = text.replace("l.a.", "los angeles").replace("la clippers", "los angeles clippers").replace("la lakers", "los angeles lakers")
    return re.sub(r"\s+", " ", text)


def team_names(game, side, config):
    names = {normalized(game[side + "_team"])}
    names.update(config["team_aliases"].get(game[side + "_abbr"], []))
    nickname = game[side + "_team"].split()[-1].lower()
    names.add("trail blazers" if nickname == "blazers" else nickname)
    return names


SCHEDULE_ROWS = (re.compile(r"^\s*(?:\d{1,2}\s+)?(?:0?[1-9]|1[0-2])/\s*(?:0?[1-9]|[12]\d|3[01])\s+(?:@|[A-Za-z])"),
                 re.compile(r"^\s*(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\.?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                            r"\w*\.?\s+\d{1,2}\s+(?:@|[A-Za-z])", re.I))


def without_schedule(text):
    """Drop season-schedule rows (date then opponent), which list this game's date in any version of a packet."""
    return "\n".join(line for line in text.splitlines() if not any(p.match(line) for p in SCHEDULE_ROWS))


def header_checks(front, game, publisher, before, config):
    """This game's date and both teams in the opening pages (schedule rows removed); a printed
    record or game number must not contradict history."""
    compact = normalized(without_schedule(front)[:6000])
    reasons, checks = [], {}
    if len(front.split()) < 100:
        reasons.append("insufficient_extractable_text")
    day = datetime.fromisoformat(game["game_date_et"])
    month = day.strftime("%B").lower()
    patterns = [rf"\b0?{day.month}/0?{day.day}(?:/(?:{day.year}|{day.year % 100:02}))?\b",
                rf"\b{month}\s+0?{day.day}(?:st|nd|rd|th)?\b", rf"\b{month[:3]}\.?\s+0?{day.day}(?:st|nd|rd|th)?\b"]
    checks["header_date"] = any(re.search(p, compact) for p in patterns)
    if not checks["header_date"]:
        reasons.append("header_date_missing")
    # Only the masthead: later pages cite the same month/day in other years ("this day in history").
    years = re.findall(rf"\b(?:{month}|{month[:3]}\.?)\s+0?{day.day}(?:st|nd|rd|th)?\s*,?\s+(20\d{{2}})\b", compact[:800])
    if years and any(int(y) != day.year for y in years):
        reasons.append("header_year_mismatch")
    for side in ("home", "away"):
        names = team_names(game, side, config)
        if not any(re.search(rf"\b{re.escape(n)}\b", compact) for n in names):
            reasons.append(side + "_team_missing")
        for name in names:
            record = re.search(re.escape(name) + r"\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)", compact)
            if record:
                actual, expected = tuple(map(int, record.groups())), before[game["game_id"], side]
                checks[side + "_record"] = {"actual": actual, "expected": expected}
                if actual != expected:
                    reasons.append(side + "_record_contradicts_history")
                break
    # Some mastheads print no date (Lakers); both pre-game records matching history pin the game just as well,
    # since every game changes a record and the previous game's packet carries the previous records.
    if "header_date_missing" in reasons and all(
            checks.get(side + "_record", {}).get("actual") == before[game["game_id"], side] for side in ("home", "away")):
        reasons.remove("header_date_missing")
        checks["header_date_from_records"] = True
    number = re.search(r"(?:regular\s+)?season\s+game\s*#?\s*(\d+)", compact)
    if number:
        side = "home" if publisher == game["home_abbr"] else "away"
        expected = sum(before[game["game_id"], side]) + 1
        checks["game_number"] = {"actual": int(number[1]), "expected": expected}
        if int(number[1]) != expected:
            reasons.append("game_number_contradicts_history")
    return reasons, checks


def pdf_pages(raw_path):
    from pypdf import PdfReader
    body = (ROOT / raw_path).read_bytes()
    body = gzip.decompress(body) if body.startswith(b"\x1f\x8b") else body
    if not body.startswith(b"%PDF"):
        raise ValueError("not a PDF")
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages]


def extract(job):
    raw_path, text_path = job
    out = ROOT / text_path
    if out.exists():
        return text_path, None
    try:
        pages = pdf_pages(raw_path)
    except Exception as exc:  # unreadable or truncated captures are rejected, not fatal
        return text_path, f"{type(exc).__name__}: {exc}"[:200]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"\n=== PDF PAGE {i} ===\n{p}" for i, p in enumerate(pages, 1)))
    return text_path, None


def provenance(candidate, config):
    """Raw path plus the evidence that the bytes existed before the cutoff, or None."""
    cutoff = stamp(candidate["cutoff_utc"])
    if candidate["kind"] == "dated_live":
        meta_path = ROOT / "data/raw/team_notes" / (hashlib.sha256(candidate["url"].encode()).hexdigest()[:24] + ".meta.json")
        if not meta_path.exists():
            return None, "not_fetched"
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != 200 or not meta.get("last_modified"):
            return None, f"cdn_status_{meta.get('status')}"
        if parsedate_to_datetime(meta["last_modified"]) > cutoff:
            return None, "cdn_modified_after_cutoff"
        return {"raw_path": meta["raw_path"], "sha256": meta["sha256"], "retrieved_at_utc": meta["retrieved_at_utc"],
                "last_modified": meta["last_modified"], "url": candidate["url"]}, None
    meta_path = replay_path(candidate["replay_url"]).with_suffix(".meta.json")
    if not meta_path.exists():
        return None, "not_fetched"
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != 200:
        return None, f"replay_status_{meta.get('status')}"
    match = REPLAY.fullmatch(meta.get("final_url", "").replace("http://", "https://", 1))
    try:
        memento = parsedate_to_datetime(meta["memento_datetime"])
    except (TypeError, KeyError, ValueError):
        return None, "replay_without_memento_time"
    if not match or utc(match[1]) != memento or memento > cutoff:
        return None, "replay_not_before_cutoff"
    return {"raw_path": meta["raw_path"], "sha256": meta["sha256"], "memento_datetime": meta["memento_datetime"],
            "replay_url": meta["final_url"], "url": candidate["url"]}, None


def review(candidates, posts, history, config):
    games = {g["game_id"]: g for g in history}
    before = records_before(history)
    docs, jobs = [], []
    for c in candidates:
        source, problem = provenance(c, config)
        doc = dict(c, source=source, reasons=[problem] if problem else [])
        if source:
            doc["text_path"] = f"data/processed/notes_v2/texts/{c['game_id'].split(':')[-1]}-{c['team']}-{source['sha256'][:12]}.txt"
            jobs.append((source["raw_path"], doc["text_path"]))
        docs.append(doc)
    with ProcessPoolExecutor(max_workers=8) as pool:
        errors = dict(pool.map(extract, jobs, chunksize=4))
    for doc in docs:
        if doc["reasons"]:
            continue
        if errors.get(doc["text_path"]):
            doc["reasons"] = ["pdf_unreadable"]
            doc["error"] = errors[doc["text_path"]]
            continue
        text = (ROOT / doc["text_path"]).read_text()
        game = games[doc["game_id"]]
        front = text.split("=== PDF PAGE 3 ===")[0]
        reasons, checks = header_checks(front, game, doc["team"], before, config)
        flags = postgame_flags(text, dict(game, home_record_before=before[game["game_id"], "home"],
                                          away_record_before=before[game["game_id"], "away"]))
        doc.update(reasons=reasons + (["postgame_content"] if flags else []), checks=checks, postgame_flags=flags,
                   pages=text.count("=== PDF PAGE "), words=len(text.split()))
    for post in posts:
        game = post_game(post, history, config)
        doc = {"team": post["team"], "kind": "html_post", "tier": "site_timestamped", "url": post["permalink"],
               "title": post["title"], "published_utc": post["published_utc"], "modified_utc": post["modified_utc"],
               "source": {"api_raw_path": post["source"]["raw_path"], "api_sha256": post["source"]["sha256"]}}
        if not game:
            docs.append(dict(doc, game_id=None, reasons=["no_unique_game_for_title_date"]))
            continue
        cutoff = cutoff_of(game, config)
        doc.update(game_id=game["game_id"], cutoff_utc=iso(cutoff),
                   text_path=f"data/processed/notes_v2/texts/{game['game_id'].split(':')[-1]}-{post['team']}-html.txt")
        (ROOT / doc["text_path"]).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / doc["text_path"]).write_text(post["text"])
        reasons = [] if stamp(post["published_utc"]) <= cutoff else ["published_after_cutoff"]
        if len(post["text"].split()) < 100:
            reasons.append("insufficient_text")
        flags = postgame_flags(post["text"], dict(game, home_record_before=before[game["game_id"], "home"],
                                                  away_record_before=before[game["game_id"], "away"]))
        doc.update(reasons=reasons + (["postgame_content"] if flags else []), postgame_flags=flags,
                   modified_after_cutoff=stamp(post["modified_utc"]) > cutoff, words=len(post["text"].split()))
        docs.append(doc)
    return docs


def choose(docs):
    """One document per team-game: archive-proven first, then the latest capture, then the longest text."""
    approved = {}
    rank = lambda d: (d["tier"] == "archive_proven", d.get("capture_timestamp") or d.get("published_utc") or "", d.get("words", 0))
    for d in docs:
        key = (d["game_id"], d["team"])
        if not d["reasons"] and (key not in approved or rank(d) > rank(approved[key])):
            approved[key] = d
    return sorted(approved.values(), key=lambda d: (d["game_id"], d["team"]))


def summarize(docs, approved, history):
    games = {g["game_id"]: g for g in history}
    season = lambda d: games[d["game_id"]]["season"]
    by_season = defaultdict(Counter)
    for d in approved:
        by_season[season(d)]["team_games"] += 1
        by_season[season(d)][d["tier"]] += 1
    unique = defaultdict(set)
    for d in approved:
        unique[season(d)].add(d["game_id"])
    summary = {"documents_considered": len(docs), "approved_team_games": len(approved),
               "approved_games": len({d["game_id"] for d in approved}),
               "by_tier": dict(Counter(d["tier"] for d in approved)), "by_kind": dict(Counter(d["kind"] for d in approved)),
               "by_publisher": dict(Counter(d["team"] for d in approved).most_common()),
               "by_season": {str(s): dict(by_season[s], games=len(unique[s])) for s in sorted(by_season)},
               "rejections": dict(Counter(r.split(":")[0] for d in docs for r in d["reasons"]).most_common())}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("index", "fetch", "review", "all"), default="all")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    history = load_history(config)
    DATA.mkdir(parents=True, exist_ok=True)
    if args.stage in ("index", "all"):
        captures = fetch_cdx(Cache(delay=3), config)
        rolling = match_rolling(rolling_versions(captures, config), history, config)
        dated, live = match_dated(captures, history, config)
        candidates = rolling + dated + live
        write_jsonl(DATA / "candidates.jsonl", candidates)
        print(f"{len(captures)} archive captures; {len(rolling)} rolling, {len(dated)} dated archive and "
              f"{len(live)} dated CDN candidates", flush=True)
    candidates = read_rows(DATA / "candidates.jsonl")
    if args.stage in ("fetch", "all"):
        if not fetch_all(candidates, config):
            return
        write_jsonl(DATA / "html_posts.jsonl", fetch_html_posts(config))
    if args.stage in ("review", "all"):
        posts = read_rows(DATA / "html_posts.jsonl") if (DATA / "html_posts.jsonl").exists() else []
        docs = review(candidates, posts, history, config)
        approved = choose(docs)
        write_jsonl(DATA / "documents.jsonl", docs)
        write_jsonl(DATA / "approved.jsonl", approved)
        summary = summarize(docs, approved, history)
        write_json(DATA / "review_summary.json", summary)
        print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
