"""Bounded source expansion in a new namespace; prior collectors stay frozen.

Discovery is not approval. Old verified PDFs are rechecked with both plain and
layout extraction, retaining all failed reasons. Archived HTML is body-only and
is kept separate until its explicit source-type adapter is approved.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import urllib.error
import urllib.request

from audit_game_notes import archive_time_verified, decode_archive_body
from audit_narrative_sources import parse_article, article_checks
from bam_data import ROOT, canonical, digest, read_rows, stamp, write_json, write_rows
from review_game_notes import content_checks

EXECUTION_SCRIPT_BYTES = Path(__file__).read_bytes()
EXECUTION_SCRIPT_SHA256 = digest(EXECUTION_SCRIPT_BYTES)


def cutoff(game):
    return (stamp(game["scheduled_start_utc"]) - timedelta(minutes=60)).isoformat().replace("+00:00", "Z")


def metadata(game, team):
    return {**{k: game[k] for k in ("game_id", "game_date_et", "home_abbr", "away_abbr")}, "cutoff_utc": cutoff(game), "team": team}


def official_record_history(history, policy, root=ROOT):
    """Exclude verified Cup finals ONLY for official record/header validation.

    The frozen booster's feature history stays untouched. Actual Cup finals
    still belong to the schedule used to bound capture timing/rest windows.
    """
    by_id = {game["game_id"]: game for game in history}
    excluded = set()
    for rule in policy["excluded_only_for_official_record_and_header_validation"]:
        game = by_id.get(rule["game_id"])
        if game is None:
            raise ValueError("Cup final exclusion is absent from verified numerical history")
        if any(game[k] != rule[k] for k in ("game_date_et", "season", "home_abbr", "away_abbr", "schedule_source")):
            raise ValueError("Cup final exclusion metadata differs from verified history")
        path = Path(root) / game["schedule_source"]
        if digest(path.read_bytes()) != rule["scoreboard_sha256"]:
            raise ValueError("Cup final scoreboard provenance changed")
        matches = [event for event in json.loads(path.read_text())["events"] if "espn:" + str(event.get("id")) == rule["game_id"]]
        if len(matches) != 1:
            raise ValueError("Cup final event is ambiguous")
        competition = matches[0]["competitions"][0]
        if competition.get("type") != {"id": "39", "abbreviation": "CC"} or not any("championship" in n.get("headline", "").lower() for n in competition.get("notes", [])):
            raise ValueError("Only explicitly identified Cup championships may be excluded")
        excluded.add(rule["game_id"])
    return [game for game in history if game["game_id"] not in excluded], excluded


class BoundedCache:
    def __init__(self, config, offline=False):
        self.config, self.offline = config, offline
        self.directory = ROOT / "data/raw" / config["namespace"]
        self.requests = 0
        self.last_archive = self.last_publisher = 0.0
        self.archive_throttled = False

    def get(self, url, kind):
        host = urlsplit(url).hostname
        if host not in {"web.archive.org", "www.nba.com", "nba.com", "cdn.nba.com"}:
            raise ValueError("Unexpected collection host")
        key = digest(url.encode())
        index = self.directory / "requests" / (key + ".json")
        if index.exists():
            prior = json.loads(index.read_text())
            if prior.get("status") == 200 and prior.get("cache_usable") is True:
                body = (ROOT / prior["raw_path"]).read_bytes()
                if digest(body) != prior["sha256"]:
                    raise ValueError("Expansion cache changed")
                return body, prior
            retrieved = prior.get("retrieved_at_utc")
            if prior.get("status") in (400, 403, 404, 410):
                return None, {**prior, "status": "cached_http_failure_requires_explicit_retry", "prior_http_status": prior["status"]}
            if retrieved and prior.get("status") in (429, 503):
                if datetime.now(timezone.utc) - stamp(retrieved) < timedelta(minutes=5):
                    return None, {**prior, "status": "deferred_five_minute_retry", "prior_http_status": prior["status"]}
        if self.offline or self.requests >= self.config["max_network_requests"]:
            return None, {"url": url, "status": "offline_or_request_budget"}
        archive = host == "web.archive.org"
        if archive and self.archive_throttled:
            return None, {"url": url, "status": "deferred_archive_throttling"}
        gap = self.config["archive_gap_seconds" if archive else "publisher_gap_seconds"]
        previous = self.last_archive if archive else self.last_publisher
        wait = gap - (time.monotonic() - previous)
        clock_path = self.directory / ("archive_clock.json" if archive else "publisher_clock.json")
        if clock_path.exists():
            finished = stamp(json.loads(clock_path.read_text())["last_finished_utc"])
            wait = max(wait, gap - (datetime.now(timezone.utc) - finished).total_seconds())
        if wait > 0:
            time.sleep(wait)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        source = {"url": url, "retrieved_at_utc": now, "source_kind": kind}
        self.requests += 1
        body = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Stanford-CS230-source-audit/1.0)", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                received = response.read()
                source.update(status=response.status, final_url=response.url,
                              content_type=response.headers.get("Content-Type"), content_encoding=response.headers.get("Content-Encoding"),
                              memento_datetime=response.headers.get("Memento-Datetime"), last_modified=response.headers.get("Last-Modified"),
                              sha256=digest(received), bytes=len(received))
                raw = self.directory / kind / (source["sha256"] + ".bin")
                raw.parent.mkdir(parents=True, exist_ok=True)
                if raw.exists() and raw.read_bytes() != received:
                    raise ValueError("Content-addressed cache collision")
                if not raw.exists():
                    raw.write_bytes(received)
                source["raw_path"] = str(raw.relative_to(ROOT))
                decoded = decode_archive_body(received)
                usable = (decoded.lstrip().startswith(b"[") if kind == "indexes" else
                          decoded.lstrip().startswith(b"{") if kind == "publisher_api" else
                          decoded.startswith(b"%PDF-") if kind == "replays" else b"__NEXT_DATA__" in decoded)
                source["cache_usable"] = usable
                body = received if usable else None
                if not usable:
                    source["error"] = "HTTP success with invalid expected payload; retained but retryable"
                    if archive and any(marker in decoded[:4000].lower() for marker in (b"temporarily offline", b"access denied", b"too many requests")):
                        self.archive_throttled = True
        except urllib.error.HTTPError as error:
            source.update(status=error.code, error=str(error))
            if error.code in (429, 503):
                self.archive_throttled = archive
        except (OSError, TimeoutError) as error:
            source.update(status="network_error", error=str(error))
        if archive:
            self.last_archive = time.monotonic()
        else:
            self.last_publisher = time.monotonic()
        write_json(clock_path, {"last_finished_utc": datetime.now(timezone.utc).isoformat()})
        attempt = self.directory / "attempts" / (digest(canonical(source)) + ".json")
        if not attempt.exists():
            write_json(attempt, source)
        write_json(index, source)
        print(json.dumps({"request": self.requests, "url": url, "status": source["status"]}), flush=True)
        return body, source


def assess_pdf(body, source, game, team, url, history, review):
    from pypdf import PdfReader
    if digest(body) != source["sha256"] or not archive_time_verified(source, url, cutoff(game)):
        raise ValueError("PDF lacks exact pre-cutoff replay/time/hash proof")
    decoded = decode_archive_body(body)
    if not decoded.startswith(b"%PDF-"):
        raise ValueError("Archived body is not a PDF")
    reader = PdfReader(io.BytesIO(decoded))
    if not reader.pages:
        raise ValueError("Empty PDF")
    alternatives = []
    for method in ("plain", "layout"):
        try:
            text = reader.pages[0].extract_text(**({"extraction_mode": "layout", "layout_mode_strip_rotated": False} if method == "layout" else {})) or ""
            reasons, checks, cleaned = content_checks(text, game, team, history)
            if team == "GSW":
                from bam_pdf_notes import warriors_header_checks
                extra_reasons, extra_checks = warriors_header_checks(text, game, history)
                extra_checks["implementation_sha256"] = digest((ROOT / "scripts/bam_pdf_notes.py").read_bytes())
                checks["publisher_header_adapter"] = {"checks": extra_checks, "reasons": extra_reasons}
                if not extra_reasons:
                    allowed = {"home_team_missing_from_header", "away_team_missing_from_header", "missing_independent_prior_record_or_game_number_check"}
                    reasons = [reason for reason in reasons if reason not in allowed]
                    checks.update(extra_checks)
            if team not in review["reviewed_publisher_templates"]:
                reasons.append("publisher_template_not_reviewed")
            if source["sha256"] in review.get("excluded_source_hashes", {}):
                reasons.append(review["excluded_source_hashes"][source["sha256"]])
            alternatives.append({"method": method, "reasons": reasons, "checks": checks, "text": cleaned})
        except Exception as error:
            alternatives.append({"method": method, "reasons": ["extraction_failed:" + type(error).__name__], "checks": {}, "text": ""})
    accepted = next((a for a in alternatives if not a["reasons"]), None)
    audit = {**metadata(game, team), "publisher_url": url, "source": source, "source_kind": "pdf",
             "pdf_pages": len(reader.pages), "decision": "approved" if accepted else "pending_content_review",
             "extractions": [{k: v for k, v in a.items() if k != "text"} for a in alternatives]}
    note = None
    if accepted:
        note = {**metadata(game, team), "source_kind": "pdf", "publisher_url": url, "source": source,
                "text": accepted["text"], "review": {"decision": "approved_template_checked_excerpt", "approved_pdf_pages": [1],
                "method": "Reviewed template plus strict replay/time/date/season/prior-record/game-number checks; original plain/layout text, with literal Warriors nickname-header validation when applicable; no invented transcription",
                "review_config_sha256": digest(canonical(review)), "checks": accepted["checks"], "extraction_method": accepted["method"]}}
    return audit, note


def parse_cdx(body):
    rows = json.loads(decode_archive_body(body))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, list) and len(row) == len(rows[0]) for row in rows):
        raise ValueError("Invalid/blocked CDX response")
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def index_candidates(records, history, slugs, config):
    """Schedule windows are candidate filters, never document approval."""
    slug_team = {slug.lower(): team for team, slug in slugs.items()}
    slug_team.update({"76ers": "PHI", "trailblazers": "POR"})
    captures, digest_teams = [], defaultdict(set)
    for record in records:
        original = urlsplit(record.get("original", ""))
        if original.scheme not in {"http", "https"} or original.hostname not in {"nba.com", "www.nba.com"}:
            continue
        match = re.search(r"/gamenotes/+([^/?]+)\.pdf", record.get("original", ""), re.I)
        if not match or match[1].lower() not in slug_team:
            continue
        if not (record.get("statuscode") == "200" and record.get("mimetype") in ("application/pdf", "application/octet-stream") or record.get("mimetype") == "warc/revisit"):
            continue
        if not re.fullmatch(r"\d{14}", record.get("timestamp", "")) or not record.get("digest"):
            continue
        team = slug_team[match[1].lower()]
        captures.append((team, record))
        digest_teams[record["digest"]].add(team)
    first = {}
    for team, record in captures:
        key = team, record["digest"]
        first[key] = min(first.get(key, record["timestamp"]), record["timestamp"])
    by_team = defaultdict(list)
    for game in sorted(history, key=lambda g: (g["scheduled_start_utc"], g["game_id"])):
        for team in (game["home_abbr"], game["away_abbr"]):
            by_team[team].append(game)
    result = []
    for team, record in captures:
        if len(digest_teams[record["digest"]]) > 1:
            continue
        when = datetime.strptime(record["timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        first_when = datetime.strptime(first[team, record["digest"]], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        games = by_team[team]
        for i, game in enumerate(games):
            if game["game_id"] in config.get("excluded_target_game_ids", []):
                continue
            if not config["minimum_game_date"] <= game["game_date_et"] <= config["maximum_game_date"]:
                continue
            end = stamp(cutoff(game))
            start = end - timedelta(hours=config["rolling_max_age_hours"])
            if i:
                start = max(start, stamp(games[i - 1]["scheduled_start_utc"]) + timedelta(hours=config["previous_tip_buffer_hours"]))
            if start <= first_when <= when <= end:
                result.append({**metadata(game, team), "publisher_url": record["original"], "archive_timestamp": record["timestamp"],
                               "archive_digest_sha1_base32": record["digest"], "source_kind": "pdf", "lead": "master_cdx_strict_window_requires_content_review"})
    return result


def knicks_leads(report, history):
    by_key = {(g["game_date_et"], g["away_abbr"] + "@" + g["home_abbr"]): g for g in history}
    result = []
    for line in report.splitlines():
        columns = [part.strip() for part in line.split("|")]
        if len(columns) < 11 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", columns[1]) or columns[3] != "RS":
            continue
        game = by_key.get((columns[1], columns[2]))
        link = re.search(r"\]\((https://www\.nba\.com/knicks/news/[^)]+)\)", columns[4])
        if game is None or link is None or "NYK" not in (game["home_abbr"], game["away_abbr"]):
            continue
        capture = columns[9] if re.fullmatch(r"\d{14}", columns[9]) else None
        result.append({**metadata(game, "NYK"), "publisher_url": link[1], "source_kind": "html",
                       "archive_timestamp": capture, "reported_published": columns[5], "reported_modified": columns[6],
                       "reported_modified_vs_cutoff": columns[8], "flags": columns[10],
                       "lead": "durable_Claude_exact_Knicks_list_requires_independent_source_verification"})
    return result


def assess_html(body, source, game, url, team="NYK", history=None):
    if digest(body) != source["sha256"] or not archive_time_verified(source, url, cutoff(game)):
        raise ValueError("HTML lacks actual pre-cutoff replay proof")
    from bam_html_notes import parse_html_note
    provenance = {**source, "publisher_url": url, "publisher": team}
    safe_game = {**game, "cutoff_utc": cutoff(game)}
    if team == "POR":
        from bam_html_por import parse_por_html_note
        from review_game_notes import prior_record
        if history is None:
            raise ValueError("Portland requires independent official prior records")
        records = {abbr: list(prior_record(game, abbr, history)) for abbr in (game["home_abbr"], game["away_abbr"])}
        parsed = parse_por_html_note(body, provenance, safe_game, records)
    else:
        parsed = parse_html_note(body, provenance, safe_game)
    reasons = parsed["eligibility"]["reasons"]
    return {"parsed_article": parsed, "reasons": reasons,
            "decision": "verified_archive_html_pending_content_review" if not reasons else "quarantined_html_metadata_or_content",
            "source_kind": "html", "publication_evidence": "exact independent pre-cutoff archived bytes"}


def deduplicate_approved(notes):
    # Equal bytes at different replay times are provenance aliases, not expendable
    # duplicate records: A -> B -> A chronology needs the final A capture.
    unique = {(note["game_id"], note["source"]["sha256"], note["source"]["final_url"]): note for note in notes}
    return sorted(unique.values(), key=lambda n: (n["game_date_et"], n["game_id"], n["team"], n["source"]["sha256"], n["source"]["final_url"]))


def replay_urls(sources):
    """Track requested and actual replay aliases, including rejected redirects."""
    return {source[key] for source in sources for key in ("url", "final_url")
            if isinstance(source.get(key), str)}


def prepare(config, offline=False, stage="all"):
    if config["cutoff_minutes"] != 60:
        raise ValueError("Only tipoff minus 60 minutes is allowed")
    namespace = config["namespace"]
    if not re.fullmatch(r"bam_notes_expansion/[a-zA-Z0-9_-]+", namespace):
        raise ValueError("New expansion namespace required")
    directory = ROOT / "data/processed" / namespace
    report_directory = ROOT / "reports" / namespace
    directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    code_snapshot = ROOT / "data/raw" / namespace / "code_versions" / (EXECUTION_SCRIPT_SHA256 + ".py")
    code_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not code_snapshot.exists():
        code_snapshot.write_bytes(EXECUTION_SCRIPT_BYTES)
    header_adapter = ROOT / "scripts/bam_pdf_notes.py"
    if header_adapter.exists():
        adapter_bytes = header_adapter.read_bytes()
        adapter_version = code_snapshot.parent / (digest(adapter_bytes) + "_bam_pdf_notes.py")
        if adapter_version.exists() and adapter_version.read_bytes() != adapter_bytes:
            raise ValueError("Immutable PDF header-adapter code changed")
        if not adapter_version.exists():
            adapter_version.write_bytes(adapter_bytes)
    history = read_rows(ROOT / config["history_path"])
    policy_path = config.get("official_record_policy_path")
    record_history, excluded_targets = official_record_history(history, json.loads((ROOT / policy_path).read_text())) if policy_path else (history, set())
    config = {**config, "excluded_target_game_ids": sorted(excluded_targets)}
    games = {g["game_id"]: g for g in history}
    review = json.loads((ROOT / config["template_review_path"]).read_text())
    for extra in config.get("additional_template_reviews", []):
        if digest((ROOT / extra["image_path"]).read_bytes()) != extra["image_sha256"]:
            raise ValueError("New template visual-review evidence changed")
        if extra["team"] not in review["reviewed_publisher_templates"]:
            review["reviewed_publisher_templates"].append(extra["team"])
    review["additional_template_reviews"] = config.get("additional_template_reviews", [])
    review_bytes = canonical(review)
    review_version = ROOT / "data/raw" / namespace / "review_config_versions" / (digest(review_bytes) + ".json")
    if review_version.exists() and review_version.read_bytes() != review_bytes:
        raise ValueError("Immutable template-review configuration changed")
    if not review_version.exists():
        write_json(review_version, review)
    existing = read_rows(ROOT / config["old_approved_path"])
    old_hashes = {n["source"]["sha256"] for n in existing}
    old_approved_replays = {n["source"]["final_url"] for n in existing}
    old_games = {n["game_id"] for n in existing}
    audit_rows = read_rows(ROOT / config["old_audit_path"])
    cache = BoundedCache(config, offline)
    approved, audits, candidates, html_audits = [], [], [], []
    if (directory / "approved_excerpts.jsonl").exists():
        approved = read_rows(directory / "approved_excerpts.jsonl")
    if (directory / "pdf_candidate_queue.jsonl").exists():
        candidates = read_rows(directory / "pdf_candidate_queue.jsonl")
    for name in ("audit_progress.jsonl", "content_audit.jsonl"):
        if (directory / name).exists():
            audits.extend(read_rows(directory / name))
    if (directory / "html_audit.jsonl").exists():
        html_audits = read_rows(directory / "html_audit.jsonl")
    if stage in ("all", "recheck") and policy_path:
        policy = json.loads((ROOT / policy_path).read_text())
        finals = policy["excluded_only_for_official_record_and_header_validation"]
        def affected(game):
            return any(game["season"] == final["season"] and game["game_date_et"] > final["game_date_et"]
                       and {game["home_abbr"], game["away_abbr"]} & {final["home_abbr"], final["away_abbr"]} for final in finals)
        targets = {}
        for old in audit_rows:
            if (old.get("text_path") and old["game_id"] not in excluded_targets
                    and (stage == "recheck" or affected(games[old["game_id"]]))):
                source = old["archive_version"]["source"]
                targets[old["game_id"], old["team"], source["final_url"]] = (old["game_id"], old["team"], old["url"], source)
        for row in audits:
            source = row.get("source") or {}
            if (source.get("raw_path") and row.get("game_id") in games and row["game_id"] not in excluded_targets
                    and (stage == "recheck" or affected(games[row["game_id"]]))):
                targets[row["game_id"], row["team"], source.get("final_url")] = (row["game_id"], row["team"], row["publisher_url"], source)
        for gid, team, url, source in targets.values():
            try:
                audit, note = assess_pdf((ROOT / source["raw_path"]).read_bytes(), source, games[gid], team, url, record_history, review)
                audits.append({**audit, "origin": "cached_strict_content_reassessment" if stage == "recheck" else "official_regular_season_record_reassessment"})
                approved = [n for n in approved if not (n["game_id"] == gid and n["source"]["final_url"] == source["final_url"])]
                if note and source["final_url"] not in old_approved_replays:
                    approved.append(note)
            except Exception as error:
                audits.append({"game_id": gid, "team": team, "publisher_url": url, "source": source,
                               "decision": "quarantined_reassessment", "reason": str(error)})
        print(json.dumps({"stage": "official_record_reassessment", "sources": len(targets), "approved_records": len(approved)}), flush=True)
    if stage in ("all", "offline"):
        seen = set()
        for old in audit_rows:
            source = (old.get("archive_version") or {}).get("source") or {}
            sha = source.get("sha256")
            if not old.get("text_path") or source.get("final_url") in old_approved_replays or (old["game_id"], source.get("final_url")) in seen:
                continue
            seen.add((old["game_id"], source.get("final_url")))
            try:
                if old["game_id"] in excluded_targets:
                    continue
                audit, note = assess_pdf((ROOT / source["raw_path"]).read_bytes(), source, games[old["game_id"]], old["team"], old["url"], record_history, review)
                audits.append({**audit, "origin": "existing_verified_content_recheck"})
                if note:
                    approved.append(note)
            except Exception as error:
                audits.append({"game_id": old["game_id"], "team": old["team"], "decision": "quarantined_existing_source", "reason": str(error)})
        print(json.dumps({"stage": "offline_recheck", "sources": len(seen), "approved": len(approved)}), flush=True)
        write_rows(directory / "approved_excerpts.jsonl", approved)
        write_rows(directory / "content_audit.jsonl", audits)
    for old in audit_rows:
        if old["game_id"] in excluded_targets:
            continue
        if old.get("text_path"):
            continue
        snapshot = (old.get("archive_check") or {}).get("snapshot") or {}
        timestamp = snapshot.get("timestamp")
        if re.fullmatch(r"\d{14}", timestamp or "") and timestamp <= stamp(old["cutoff_utc"]).strftime("%Y%m%d%H%M%S"):
            candidates.append({**metadata(games[old["game_id"]], old["team"]), "publisher_url": old["url"], "archive_timestamp": timestamp,
                               "source_kind": "pdf", "lead": "existing_failed_replay_retry"})
    if stage in ("all", "index"):
        leads_paths = config.get("index_lead_paths", [])
        if leads_paths:
            provenance = json.loads((ROOT / config["research_directory"] / "indices/manifest.json").read_text())
            pins = {f["path"]: f["sha256"] for f in provenance["files"]}
            records = []
            for path in leads_paths:
                payload = (ROOT / path).read_bytes()
                if digest(payload) != pins[path]:
                    raise ValueError("Durable discovery index checksum changed")
                records.extend(parse_cdx(payload))
            source = {"source_kind": "durable_discovery_index_lead_only", "paths": leads_paths,
                      "sha256": {path: pins[path] for path in leads_paths}}
            body = True
        else:
            body, source = cache.get(config["cdx_url"], "indexes")
            records = parse_cdx(body) if body else []
        if body:
            try:
                slugs = json.loads((ROOT / "config/game_notes_collection.json").read_text())["rolling_teams"]
                found = index_candidates(records, history, slugs, config)
                candidates.extend(found)
                write_json(report_directory / "master_cdx.json", {"source": source, "candidate_packets": len(found), "note": "Index matches are unapproved leads"})
            except Exception as error:
                write_json(report_directory / "master_cdx.json", {"source": source, "decision": "quarantined_invalid_index", "reason": str(error)})
        else:
            write_json(report_directory / "master_cdx.json", {"source": source, "decision": "index_unavailable_retryable"})
    research = ROOT / config["research_directory"]
    leads = knicks_leads((research / "team_sites.md").read_text(), history)
    write_rows(directory / "knicks_exact_leads.jsonl", leads)
    all_candidates = {(c["game_id"], c["team"], c["publisher_url"], c["archive_timestamp"]): c for c in candidates if c["game_id"] not in excluded_targets}
    candidates = sorted(all_candidates.values(), key=lambda c: (c["game_id"] in old_games, c["team"] not in review["reviewed_publisher_templates"], -int(c["game_date_et"][:4]), c["game_date_et"], c["game_id"], c["team"], -int(c["archive_timestamp"]), c["publisher_url"]))
    write_rows(directory / "pdf_candidate_queue.jsonl", candidates)
    if stage in ("all", "fetch"):
        seen_game_publisher = set()
        approved_replays = replay_urls(n["source"] for n in existing + approved)
        # The offline pass already assessed every source-verified old replay.
        # Its original raw bytes remain readable: do not redownload a rejected
        # header or an already approved packet just because the namespace is new.
        verified_old_replays = replay_urls((row.get("archive_version") or {}).get("source", {})
                                          for row in audit_rows if row.get("text_path"))
        finished_replays = replay_urls(row.get("source", {}) for row in audits
                                      if row.get("decision") in ("approved", "pending_content_review", "quarantined_replay"))
        attempted = 0
        for candidate in candidates:
            key = candidate["game_id"], candidate["team"]
            url = "https://web.archive.org/web/" + candidate["archive_timestamp"] + "id_/" + candidate["publisher_url"]
            if url in approved_replays or url in verified_old_replays or url in finished_replays:
                continue
            if key in seen_game_publisher or attempted >= config["max_pdf_replays"]:
                continue
            before = cache.requests
            body, source = cache.get(url, "replays")
            attempted += cache.requests - before
            if body is not None or cache.requests != before:
                seen_game_publisher.add(key)
            if body is None:
                audits.append({**candidate, "source": source, "decision": "replay_unverified"})
                write_rows(directory / "audit_progress.jsonl", audits)
                if cache.archive_throttled:
                    break  # Keep remaining queue; retry a later bounded pass.
                continue
            try:
                cdx_digest = candidate.get("archive_digest_sha1_base32")
                if cdx_digest and base64.b32encode(hashlib.sha1(decode_archive_body(body)).digest()).decode() != cdx_digest:
                    raise ValueError("Replay bytes disagree with CDX SHA-1 digest")
                audit, note = assess_pdf(body, source, games[candidate["game_id"]], candidate["team"], candidate["publisher_url"], record_history, review)
                audits.append({**audit, "origin": candidate["lead"]})
                if note:
                    approved.append(note)
            except Exception as error:
                audits.append({**candidate, "source": source, "decision": "quarantined_replay", "reason": str(error)})
            write_rows(directory / "audit_progress.jsonl", audits)
    if stage in ("all", "html"):
        approved_html = read_rows(directory / "approved_html.jsonl") if (directory / "approved_html.jsonl").exists() else []
        html_reviews_path = config.get("html_content_review_path")
        html_reviews = {r["source_sha256"]: r for r in read_rows(ROOT / html_reviews_path)} if html_reviews_path and str(html_reviews_path).endswith(".jsonl") else {
            r["source_sha256"]: r for r in json.loads((ROOT / html_reviews_path).read_text())} if html_reviews_path else {}
        archived = [lead for lead in leads if lead["archive_timestamp"]]
        archived.sort(key=lambda c: (c["game_id"] in old_games, c["game_date_et"], c["game_id"]))
        html_requests = 0
        for lead in archived:
            url = "https://web.archive.org/web/" + lead["archive_timestamp"] + "id_/" + lead["publisher_url"]
            if html_requests >= config["max_html_replays"]:
                break
            before = cache.requests
            body, source = cache.get(url, "html")
            html_requests += cache.requests - before
            if body is None:
                html_audits.append({**lead, "source": source, "decision": "html_replay_unverified"})
                if cache.archive_throttled:
                    break
                continue
            try:
                audit = {**lead, "source": source, **assess_html(body, source, games[lead["game_id"]], lead["publisher_url"])}
                html_audits.append(audit)
                parsed = audit["parsed_article"]
                content_review = html_reviews.get(source["sha256"])
                if audit["reasons"]:
                    approved_html = [note for note in approved_html if note["source"]["sha256"] != source["sha256"]]
                if content_review and content_review["decision"] != "approved_source_for_pregame_research":
                    approved_html = [note for note in approved_html if note["source"]["sha256"] != source["sha256"]]
                    audit["decision"] = "quarantined_manual_content_review"
                    audit["manual_content_review"] = content_review
                if not audit["reasons"] and content_review and content_review["decision"] == "approved_source_for_pregame_research":
                    if content_review["game_id"] != lead["game_id"] or content_review["sections_sha256"] != digest(canonical(parsed["sections"])):
                        raise ValueError("HTML content review does not match exact source sections/game")
                    review_bytes = canonical(content_review)
                    review_leaf = report_directory / "html_reviews" / (source["sha256"] + "_" + digest(review_bytes)[:12] + ".json")
                    if review_leaf.exists() and review_leaf.read_bytes() != review_bytes:
                        raise ValueError("Immutable per-source HTML content review changed")
                    if not review_leaf.exists():
                        write_json(review_leaf, content_review)
                    approved_html.append({**metadata(games[lead["game_id"]], "NYK"), "source_kind": "html", "publisher_url": lead["publisher_url"],
                                          "source": source, "text": "\n\n".join(section["text"] for section in parsed["sections"]),
                                          "review": {"decision": "approved_pregame_html_article", "method": "Exact pre-cutoff archived article plus strict parser and source-only full-article content review",
                                                     "sections_sha256": content_review["sections_sha256"], "approved_sections": content_review["reviewed_sections"],
                                                     "content_review_path": str(review_leaf.relative_to(ROOT)), "content_review_sha256": digest(review_bytes),
                                                     "parser_sha256": digest((ROOT / "scripts/bam_html_notes.py").read_bytes())}})
                    audit["decision"] = "approved_archived_html_article"
            except Exception as error:
                html_audits.append({**lead, "source": source, "decision": "quarantined_html", "reason": str(error)})
        write_rows(directory / "approved_html.jsonl", deduplicate_approved(approved_html))
    if stage == "por":
        por_rows = []
        approved_por = []
        review_path = config.get("por_html_content_review_path")
        por_reviews = {r["source_sha256"]: r for r in json.loads((ROOT / review_path).read_text())} if review_path else {}
        html_requests = 0
        for lead in read_rows(directory / "por_candidate_leads.jsonl"):
            if not lead["archive_leads"]:
                continue
            game = games[lead["game_id"]]
            if (lead["team"] != "POR" or "POR" not in {game["home_abbr"], game["away_abbr"]}
                    or any(lead[k] != game[k] for k in ("game_date_et", "home_abbr", "away_abbr"))
                    or lead["cutoff_utc"] != cutoff(game)
                    or not lead["publisher_url"].startswith("https://www.nba.com/blazers/news/game-notes-")):
                raise ValueError("Portland discovery lead differs from verified schedule/publisher")
            capture = max(lead["archive_leads"], key=lambda row: row["timestamp"])
            timestamp = capture["timestamp"]
            if not re.fullmatch(r"\d{14}", timestamp) or timestamp > stamp(cutoff(game)).strftime("%Y%m%d%H%M%S"):
                raise ValueError("Portland discovery capture exceeds prediction cutoff")
            if html_requests >= config["max_html_replays"]:
                break
            url = "https://web.archive.org/web/" + timestamp + "id_/" + lead["publisher_url"]
            before = cache.requests
            body, source = cache.get(url, "html")
            html_requests += cache.requests - before
            row = {**lead, "source": source, "included_in_model_inputs": False}
            if body is None:
                por_rows.append({**row, "decision": "html_replay_unverified"})
                if cache.archive_throttled:
                    break
                continue
            try:
                if base64.b32encode(hashlib.sha1(decode_archive_body(body)).digest()).decode() != capture["digest"]:
                    raise ValueError("Portland replay bytes disagree with CDX SHA-1")
                assessment = assess_html(body, source, game, lead["publisher_url"], team="POR", history=record_history)
                audit = {**row, **assessment, "decision": "pending_publisher_adapter_and_full_article_content_review"}
                parsed = assessment["parsed_article"]
                content_review = por_reviews.get(source["sha256"])
                if content_review and content_review["decision"] != "approved_source_for_pregame_research":
                    audit["decision"] = "quarantined_manual_content_review"
                if not assessment["reasons"] and content_review and content_review["decision"] == "approved_source_for_pregame_research":
                    if content_review["game_id"] != lead["game_id"] or content_review["sections_sha256"] != digest(canonical(parsed["sections"])):
                        raise ValueError("Portland content review differs from exact source sections/game")
                    review_bytes = canonical(content_review)
                    leaf = report_directory / "por_html_reviews" / (source["sha256"] + "_" + digest(review_bytes)[:12] + ".json")
                    if leaf.exists() and leaf.read_bytes() != review_bytes:
                        raise ValueError("Immutable Portland content review changed")
                    if not leaf.exists():
                        write_json(leaf, content_review)
                    from review_game_notes import prior_record
                    records = {abbr: list(prior_record(game, abbr, record_history)) for abbr in (game["home_abbr"], game["away_abbr"])}
                    approved_por.append({**metadata(game, "POR"), "source_kind": "html", "publisher_url": lead["publisher_url"], "source": source,
                                         "text": "\n\n".join(s["text"] for s in parsed["sections"]),
                                         "review": {"decision": "approved_pregame_html_article", "method": "Strict archived Portland opening plus independent official prior records, verified schedule and full source-only content review",
                                                    "sections_sha256": content_review["sections_sha256"], "approved_sections": content_review["reviewed_sections"],
                                                    "content_review_path": str(leaf.relative_to(ROOT)), "content_review_sha256": digest(review_bytes),
                                                    "parser_sha256": digest((ROOT / "scripts/bam_html_notes.py").read_bytes()),
                                                    "publisher_parser_sha256": digest((ROOT / "scripts/bam_html_por.py").read_bytes()),
                                                    "header_prior_records": records, "source_warnings": content_review["source_warnings"]}})
                    audit["decision"] = "approved_archived_html_article"
                por_rows.append(audit)
            except Exception as error:
                por_rows.append({**row, "decision": "quarantined_html", "reason": str(error)})
        write_rows(directory / "por_html_audit.jsonl", por_rows)
        write_rows(directory / "approved_por_html.jsonl", deduplicate_approved(approved_por))
    if stage in ("all", "publisher"):
        lookup = {lead["publisher_url"].rstrip("/"): lead for lead in leads}
        publisher_rows = []
        for page in range(1, config.get("max_knicks_api_pages", 22) + 1):
            url = f"https://www.nba.com/knicks/api/content/category/game-notes?page={page}"
            body, source = cache.get(url, "publisher_api")
            if body is None:
                publisher_rows.append({"page": page, "source": source, "decision": "publisher_api_unavailable"})
                continue
            try:
                data = json.loads(decode_archive_body(body))
                if not isinstance(data.get("items"), list):
                    raise ValueError("Publisher API has no item list")
                for item in data["items"]:
                    lead = lookup.get(str(item.get("permalink", "")).rstrip("/"))
                    if lead is None:
                        continue
                    from collect_pilot import TextOnly
                    blocks = []
                    for block in item.get("contentStructured", []):
                        if block.get("type") in ("paragraph", "heading", "list"):
                            parser = TextOnly()
                            parser.feed(block.get("html") or "")
                            text = " ".join(" ".join(parser.parts).split())
                            if text:
                                blocks.append(text)
                    reasons = []
                    try:
                        published, modified = stamp(item["date"]), stamp(item["modified"])
                        if published > stamp(lead["cutoff_utc"]) or modified > stamp(lead["cutoff_utc"]):
                            reasons.append("publisher_current_revision_after_cutoff")
                        if modified < published:
                            reasons.append("modified_before_published")
                    except (KeyError, TypeError, ValueError):
                        reasons.append("missing_or_invalid_publisher_times")
                    if not blocks:
                        reasons.append("empty_article_body")
                    publisher_rows.append({**lead, "source": source, "title": item.get("title"),
                                           "published": item.get("date"), "modified": item.get("modified"),
                                           "article_body": "\n\n".join(blocks), "reasons": reasons,
                                           "decision": "quarantined_current_revision" if reasons else "cms_clean_weaker_evidence_pending_archive_and_content_review",
                                           "included_in_model_inputs": False})
            except Exception as error:
                publisher_rows.append({"page": page, "source": source, "decision": "publisher_api_parse_error", "reason": str(error)})
            write_rows(directory / "knicks_publisher_candidates.jsonl", publisher_rows)
        write_rows(directory / "knicks_publisher_candidates.jsonl", publisher_rows)
    approved = deduplicate_approved(approved)
    dedup_audits = {}
    for audit in audits:
        source = audit.get("source") or {}
        key = (audit.get("game_id"), audit.get("team"), source.get("final_url") or source.get("url") or audit.get("publisher_url"), source.get("sha256"))
        dedup_audits[key] = audit
    audits = list(dedup_audits.values())
    if stage in ("all", "offline", "fetch", "recheck"):
        write_rows(directory / "approved_excerpts.jsonl", approved)
        write_rows(directory / "content_audit.jsonl", audits)
    if stage in ("all", "html"):
        html_audits = list({(row["game_id"], row.get("source", {}).get("url")): row for row in html_audits}.values())
        write_rows(directory / "html_audit.jsonl", html_audits)
    summary = {"namespace": namespace, "network_requests": cache.requests, "approved_new_packet_versions": len(approved),
               "approved_unique_raw_hashes": len({note["source"]["sha256"] for note in approved}),
               "new_unique_raw_hashes": len({note["source"]["sha256"] for note in approved} - old_hashes),
               "new_unique_games": len({n["game_id"] for n in approved} - old_games), "by_publisher": dict(Counter(n["team"] for n in approved)),
               "pdf_candidates": len(candidates), "knicks_exact_schedule_matched_regular_season_leads": len(leads),
               "knicks_archive_leads": sum(bool(lead["archive_timestamp"]) for lead in leads),
               "audit_decisions": dict(Counter(a["decision"] for a in audits)), "html_decisions": dict(Counter(a["decision"] for a in html_audits)),
               "archive_throttled": cache.archive_throttled,
               "official_record_excluded_cup_final_ids": sorted(excluded_targets),
               "frozen_booster_feature_history": "unchanged; includes these real prior Cup finals and therefore is not strictly regular-season-only",
               "original_collectors_settings_and_256_snapshot": "unchanged",
               "html_model_inputs_approved": len(read_rows(directory / "approved_html.jsonl")) if (directory / "approved_html.jsonl").exists() else 0,
               "script_sha256": EXECUTION_SCRIPT_SHA256, "script_snapshot_path": str(code_snapshot.relative_to(ROOT)),
               "script_hash_semantics": "captured at module execution start", "config": config,
               "provenance_sha256": {name: digest((research / name).read_bytes()) for name in ("web_archives.md", "team_sites.md", "manifest.json")}}
    write_json(report_directory / ("summary_" + stage + ".json"), summary)
    immutable_summary = ROOT / "data/raw" / namespace / "pass_summaries" / (digest(canonical(summary)) + ".json")
    if immutable_summary.exists() and immutable_summary.read_bytes() != canonical(summary):
        raise ValueError("Immutable collection-pass summary changed")
    if not immutable_summary.exists():
        write_json(immutable_summary, summary)
    if stage in ("all", "offline", "fetch", "index", "recheck"):
        write_json(report_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/bam_notes_expansion.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--stage", choices=("all", "offline", "index", "fetch", "html", "por", "publisher", "recheck"), default="all")
    parser.add_argument("--passes", type=int, default=1, help="Sequential bounded replay passes; stop on throttling or no new HTTP requests")
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("passes must be positive")
    for number in range(1, args.passes + 1):
        print(json.dumps({"pass": number, "stage": args.stage}), flush=True)
        summary = prepare(json.loads(args.config.read_text()), args.offline, args.stage)
        if summary["archive_throttled"] or not summary["network_requests"]:
            break


if __name__ == "__main__":
    main()
