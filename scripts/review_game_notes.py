"""Approve only archived front pages passing reviewed template and schedule checks."""
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
import re
import unicodedata

from pypdf import PdfReader
from audit_game_notes import archive_time_verified, decode_archive_body
from collect_pilot import ROOT, write_json, write_jsonl
from run_gradient_boosting import read_rows, sha256

DATA = ROOT / "data/processed/game_notes_collection"
REPORT = ROOT / "reports/game_notes_collection"
REVIEW = ROOT / "config/game_notes_content_review.json"


def clean_front_page(text):
    # Schedule columns are not useful narrative. Drop only standalone rows with
    # month/day followed by an opponent name or @; player GP/GS rows survive.
    numeric_row = r"^\s*(?:\d{1,2}\s+)?(?:0?[1-9]|1[0-2])/\s*(?:0?[1-9]|[12]\d|3[01])\s+(?:@|[A-Za-z])"
    named_row = r"^\s*(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\.?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2}\s+(?:@|[A-Za-z])"
    lines = [line for line in text.splitlines() if not re.match(numeric_row, line) and not re.match(named_row, line, re.I)]
    text = "\n".join(lines)
    text = re.split(r"CAVALIERS PRONUNCIATIONS", text, maxsplit=1)[0]
    return text.strip()


def normalized(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = text.replace("l.a.", "los angeles").replace("la clippers", "los angeles clippers").replace("la lakers", "los angeles lakers")
    return re.sub(r"\s+", " ", text)


def prior_record(game, team, history):
    earlier = [g for g in history if g["season"] == game["season"] and g["game_date_et"] < game["game_date_et"] and team in (g["home_abbr"], g["away_abbr"])]
    wins = sum((g["home_score"] > g["away_score"]) == (g["home_abbr"] == team) for g in earlier)
    return wins, len(earlier) - wins


def content_checks(text, game, publisher, history):
    clean = clean_front_page(text)
    header = re.split(r"LAST GAMES? STARTERS|PROBABLE STARTERS|PREVIOUS STARTING LINEUP|PLAYER /|Forward #", clean, maxsplit=1, flags=re.I)[0]
    compact = normalized(header)
    reasons, checks = [], {}
    if len(clean.split()) < 150:
        reasons.append("insufficient_extractable_text")
    day = datetime.fromisoformat(game["game_date_et"])
    month = day.strftime("%B").lower()
    date_patterns = [rf"\b{day.month:02}/{day.day:02}/(?:{day.year}|{day.year % 100:02})\b",
                     rf"\b{day.month}/{day.day}/(?:{day.year}|{day.year % 100:02})\b",
                     rf"\b{month}\s+0?{day.day}(?:st|nd|rd|th)?\b",
                     rf"\b{month[:3]}\.?\s+0?{day.day}(?:st|nd|rd|th)?\b"]
    checks["header_date"] = any(re.search(p, compact) for p in date_patterns)
    if not checks["header_date"]:
        reasons.append("header_date_mismatch_or_missing")
    # A month/day match cannot rescue a header with the wrong explicit year.
    explicit_years = re.findall(rf"\b(?:{month}|{month[:3]}\.?)\s+0?{day.day}(?:st|nd|rd|th)?\s*,?\s+(20\d{{2}})\b", compact)
    if explicit_years and any(int(y) != day.year for y in explicit_years):
        reasons.append("header_year_mismatch")
    season = f"{game['season']}-{(game['season'] + 1) % 100:02}"
    if season not in text:
        reasons.append("season_mismatch_or_missing")
    records_found = 0
    for side in ("home", "away"):
        name = normalized(game[side + "_team"])
        if name not in compact:
            reasons.append(side + "_team_missing_from_header")
        record = re.search(re.escape(name) + r"\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)", compact)
        if record:
            records_found += 1
            expected = prior_record(game, game[side + "_abbr"], history)
            actual = tuple(map(int, record.groups()))
            checks[side + "_record"] = {"actual": actual, "expected": expected}
            if actual != expected:
                reasons.append(side + "_record_disagrees_with_prior_history")
    number = re.search(r"(?:regular\s+)?season\s+game\s*#?\s*(\d+)", compact)
    if number:
        expected = sum(prior_record(game, publisher, history)) + 1
        checks["publisher_game_number"] = {"actual": int(number[1]), "expected": expected}
        if int(number[1]) != expected:
            reasons.append("publisher_game_number_disagrees_with_history")
    elif records_found != 2:
        reasons.append("missing_independent_prior_record_or_game_number_check")
    # These reviewed packet templates label the publisher's home/road game.
    home_header = bool(re.search(r"\bhome\s+game\s*#?\s*\d+", compact))
    away_header = bool(re.search(r"\b(?:away|road)\s+game\s*#?\s*\d+", compact))
    if home_header and not away_header and publisher != game["home_abbr"]:
        reasons.append("header_home_away_mismatch")
    if away_header and not home_header and publisher != game["away_abbr"]:
        reasons.append("header_home_away_mismatch")
    return reasons, checks, clean


def main():
    review = json.loads(REVIEW.read_text())
    config = json.loads((ROOT / "config/game_notes_collection.json").read_text())
    history = read_rows(ROOT / config["history_path"])
    games = {g["game_id"]: g for g in history}
    rows = read_rows(DATA / "audit.jsonl")
    candidates, decisions = [], []
    for row in rows:
        result = {k: row[k] for k in ("game_id", "game_date_et", "team", "url")}
        if not row.get("text_path"):
            result["reasons"] = [row["decision"]]
            decisions.append(result)
            continue
        source = row["archive_version"]["source"]
        if not archive_time_verified(source, row["url"], row["cutoff_utc"]) or sha256(ROOT / source["raw_path"]) != source["sha256"]:
            raise ValueError("Archived publication proof/checksum changed")
        # Replay extraction directly from verified archived bytes, not a mutable
        # candidate text file, and verify the prior extraction too.
        body = decode_archive_body((ROOT / source["raw_path"]).read_bytes())
        text = PdfReader(io.BytesIO(body)).pages[0].extract_text() or ""
        if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("Front-page extraction changed")
        reasons, checks, clean = content_checks(text, games[row["game_id"]], row["team"], history)
        if row["team"] not in review["reviewed_publisher_templates"]:
            reasons.append("publisher_template_not_reviewed")
        if source["sha256"] in review.get("excluded_source_hashes", {}):
            reasons.append(review["excluded_source_hashes"][source["sha256"]])
        result.update(reasons=reasons, checks=checks, archive_sha256=source["sha256"])
        decisions.append(result)
        if not reasons:
            candidates.append({**{k: row[k] for k in ("game_id", "game_date_et", "home_abbr", "away_abbr", "cutoff_utc", "team")},
                               "text": clean, "publisher_url": row["url"], "source": source,
                               "review": {"decision": "approved_template_checked_excerpt", "approved_pdf_pages": [1],
                                          "method": "reviewed publisher template plus per-document archived-time/header/date/record checks; not individual manual review of every sentence",
                                          "review_config_sha256": sha256(REVIEW), "checks": checks}})
    # Exactly one packet per game; fixed order independent of labels/performance.
    # Prefer a longer verified front page, then alphabetical publisher and hash.
    chosen = {}
    for row in sorted(candidates, key=lambda r: (r["game_id"], -len(r["text"].split()), r["team"], r["source"]["sha256"])):
        chosen.setdefault(row["game_id"], row)
    approved = sorted(chosen.values(), key=lambda r: (r["game_date_et"], r["game_id"]))
    write_jsonl(DATA / "content_audit.jsonl", decisions)
    write_jsonl(DATA / "approved_excerpts.jsonl", approved)
    summary = {"attempted_packet_versions": len(rows), "verified_archived_versions": sum(bool(r.get("text_path")) for r in rows),
               "approved_packet_versions": len(candidates), "approved_unique_games": len(approved),
               "by_publisher": dict(Counter(r["team"] for r in approved)),
               "by_season": dict(Counter(str(games[r["game_id"]]["season"]) for r in approved)),
               "rejection_reasons": dict(Counter(reason for r in decisions for reason in r["reasons"])),
               "review_method": "template review plus per-document mechanical checks, not every sentence manually reviewed",
               "scope": "Schedule-defined discovery; archive/template availability creates selective coverage, not leaguewide representativeness"}
    write_json(REPORT / "content_review.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
