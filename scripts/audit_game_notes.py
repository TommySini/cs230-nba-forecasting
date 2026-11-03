"""Discover rich team game notes, preserving provenance and quarantine."""
import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

from pypdf import PdfReader

from audit_narrative_sources import article_checks, audit_article, parse_article
from collect_pilot import Cache, ROOT, iso, stamp, write_json, write_jsonl

CONFIG = ROOT / "config/game_notes_sources.json"
DATA = ROOT / "data/processed/game_notes_sources"
REPORT = ROOT / "reports/game_notes_sources"


def schedule_match(spec, history):
    found = [g for g in history if all(g[k] == spec[k] for k in ("game_date_et", "away_abbr", "home_abbr"))]
    if len(found) != 1:
        raise ValueError(f"Expected one historical schedule match: {spec['url']}")
    return found[0]


def archive_lookup(cache, url, cutoff):
    query = "https://archive.org/wayback/available?" + urlencode({"url": url, "timestamp": stamp(cutoff).strftime("%Y%m%d%H%M%S")})
    body, source = cache.get(query, "archive_checks", ".json")
    result = {"source": source, "decision": "lookup_failed", "approved": False}
    if body is not None:
        try:
            snapshot = json.loads(body).get("archived_snapshots", {}).get("closest")
            result.update(snapshot=snapshot, decision="no_snapshot_returned")
            if snapshot and snapshot.get("available") is True and snapshot.get("status") == "200":
                before = snapshot.get("timestamp", "9") <= stamp(cutoff).strftime("%Y%m%d%H%M%S")
                result["decision"] = "pre_cutoff_snapshot_requires_bytes_review" if before else "closest_snapshot_after_cutoff"
        except (ValueError, TypeError, AttributeError):
            result["decision"] = "unparseable_archive_response"
    return result


def archive_time_verified(source, original_url, cutoff):
    """Check the actual replay, including redirects, rather than the request alone."""
    match = re.fullmatch(r"https://web\.archive\.org/web/(\d{14})(?:id_)?/(.+)", source.get("final_url", ""))
    if not match or match[2] != original_url or source.get("status") != 200:
        return False
    try:
        captured = datetime.strptime(match[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        memento = parsedate_to_datetime(source["memento_datetime"])
    except (ValueError, TypeError, KeyError):
        return False
    return memento == captured and captured <= stamp(cutoff)


def decode_archive_body(body):
    # Preserve downloaded bytes/checksum; decode only for extraction. Wayback
    # can replay gzip content even without an explicit Accept-Encoding request.
    return gzip.decompress(body) if body.startswith(b"\x1f\x8b") else body


def fetch_archive_version(cache, row, suffix, game):
    check = row["archive_check"]
    if check["decision"] != "pre_cutoff_snapshot_requires_bytes_review":
        return {"decision": "no_pre_cutoff_replay_selected"}
    snapshot = check["snapshot"]
    url = snapshot["url"].replace("http://web.archive.org/", "https://web.archive.org/", 1)
    url = re.sub(r"/web/(\d{14})/", r"/web/\1id_/", url, count=1)
    base = ROOT / "data/raw/archive_versions" / hashlib.sha256(url.encode()).hexdigest()[:24]
    meta_path, body_path = base.with_suffix(".meta.json"), base.with_suffix(suffix)
    body = None
    if meta_path.exists():
        source = json.loads(meta_path.read_text())
        if source["status"] == 200:
            body = body_path.read_bytes()
            if hashlib.sha256(body).hexdigest() != source["sha256"]:
                raise ValueError("Archived version checksum mismatch")
        elif not cache.offline and source["status"] in ("network_error", 429, 500, 502, 503, 504):
            # Preserve failed attempts, but do not turn transient failures into
            # permanent evidence that a historical version is unavailable.
            attempt = meta_path.parent / "failed_attempts" / (meta_path.stem + "-" + hashlib.sha256(meta_path.read_bytes()).hexdigest()[:12] + ".json")
            write_json(attempt, source)
            meta_path.unlink()
    if not meta_path.exists() and cache.offline:
        source = {"url": url, "status": "not_cached"}
    elif not meta_path.exists():
        source = {"url": url, "retrieved_at_utc": iso(datetime.now(timezone.utc))}
        attempts = []
        for trial in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "CS230-academic-data-pilot/0.1"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    body = response.read()
                    source.update(status=response.status, final_url=response.url,
                                  memento_datetime=response.headers.get("Memento-Datetime"),
                                  content_type=response.headers.get("Content-Type"),
                                  content_encoding=response.headers.get("Content-Encoding"),
                                  sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), raw_path=str(body_path.relative_to(ROOT)))
                    body_path.parent.mkdir(parents=True, exist_ok=True)
                    body_path.write_bytes(body)
                    break
            except urllib.error.HTTPError as exc:
                source.update(status=exc.code, error=str(exc))
            except (OSError, TimeoutError) as exc:
                source.update(status="network_error", error=str(exc))
            attempts.append({"status": source["status"], "error": source.get("error"), "attempt": trial + 1})
            if source["status"] not in ("network_error", 429, 500, 502, 503, 504):
                break
            time.sleep(0.5 * (trial + 1))
        if attempts:
            source["failed_attempts"] = attempts
        write_json(meta_path, source)
    result = {"source": source, "decision": "replay_unverified", "publication_time_verified": False}
    if body is None or not archive_time_verified(source, row["url"], row["cutoff_utc"]):
        return result
    result["publication_time_verified"] = True
    result["decision"] = "pre_cutoff_replay_content_requires_review"
    if suffix == ".pdf":
        result["exact_bytes_match_current"] = source["sha256"] == row["source"]["sha256"]
        result["pages"] = len(PdfReader(io.BytesIO(decode_archive_body(body))).pages)
    else:
        try:
            decoded = decode_archive_body(body)
            archived = parse_article(decoded, article_types=("NewsArticle", "Article"))
            current_text = (ROOT / row["text_path"]).read_text().strip()
            result.update(article_text_matches_current=archived["text"].strip() == current_text,
                          decoded_sha256=hashlib.sha256(decoded).hexdigest(),
                          archived_title=archived["title"], archived_published=archived["published"], archived_modified=archived["modified"])
            result["rejection_reasons"] = article_checks(archived, game, stamp(row["cutoff_utc"]), row["url"])
            text_path = DATA / (row["game_date_et"] + "-" + row["team"] + "-archived.txt")
            text_path.write_text(archived["text"] + "\n")
            result["text_path"] = str(text_path.relative_to(ROOT))
        except (ValueError, KeyError, TypeError) as exc:
            result.update(decision="archived_article_parse_failed", error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    history = [json.loads(line) for line in (ROOT / "data/processed/seasonal/game_history.jsonl").read_text().splitlines()]
    # All historical joins must succeed before writing outputs.
    games = {s["url"]: schedule_match(s, history) for s in config["articles"] + config["pdfs"] if s.get("game_date_et")}
    DATA.mkdir(parents=True, exist_ok=True)
    cache = Cache(offline=args.offline)
    articles = []
    for spec in config["articles"]:
        row = audit_article(cache, spec, games[spec["url"]], config["cutoff_minutes"], DATA)
        row["archive_check"] = archive_lookup(cache, spec["url"], row["cutoff_utc"])
        row["archive_version"] = fetch_archive_version(cache, row, ".html", games[spec["url"]])
        articles.append(row)
    pdfs = []
    for spec in config["pdfs"]:
        body, source = cache.get(spec["url"], "game_notes", ".pdf")
        row = {**spec, "source": source, "included_in_inputs": False, "decision": "fetch_failed"}
        if body is not None:
            reader = PdfReader(io.BytesIO(body))
            text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            path = DATA / f"{spec['team']}-{spec.get('game_date_et', 'rolling')}.txt"
            path.write_text(text)
            row.update(pages=len(reader.pages), words=len(text.split()),
                       pdf_creation_date=(reader.metadata or {}).get("/CreationDate"),
                       pdf_modification_date=(reader.metadata or {}).get("/ModDate"),
                       text_path=str(path.relative_to(ROOT)),
                       decision="rolling_current_version_not_historical_evidence" if spec["kind"] == "rolling" else "candidate_requires_content_and_publication_review")
        if spec.get("game_date_et"):
            game = games[spec["url"]]
            cutoff = iso(stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"]))
            row.update(game_id=game["game_id"], cutoff_utc=cutoff, archive_check=archive_lookup(cache, spec["url"], cutoff))
            row["archive_version"] = fetch_archive_version(cache, row, ".pdf", game)
        pdfs.append(row)
    approved = []
    reviews = {r["url"]: r for r in json.loads((ROOT / "config/game_notes_reviews.json").read_text())}
    for row in articles + pdfs:
        review = reviews.get(row["url"])
        if review is None:
            continue
        archived = row.get("archive_version", {})
        source = archived.get("source", {})
        if source.get("sha256") != review["archive_sha256"] or row.get("game_id") != review["game_id"]:
            raise ValueError("Reviewed source/version changed; new review required")
        row["manual_review"] = review
        row["decision"] = review["decision"]
        if review["decision"] == "approved_reviewed_pregame_excerpt":
            if not archived.get("publication_time_verified") or not archive_time_verified(source, row["url"], row["cutoff_utc"]):
                raise ValueError("Approved excerpt lacks verified pre-cutoff capture")
            body = (ROOT / source["raw_path"]).read_bytes()
            if hashlib.sha256(body).hexdigest() != review["archive_sha256"]:
                raise ValueError("Reviewed archive bytes changed")
            reader = PdfReader(io.BytesIO(body))
            text = "\n\n".join(reader.pages[p - 1].extract_text() or "" for p in review["approved_pdf_pages"])
            if not text.strip():
                raise ValueError("Empty approved excerpt")
            approved.append({"game_id": row["game_id"], "game_date_et": row["game_date_et"],
                             "home_abbr": row["home_abbr"], "away_abbr": row["away_abbr"],
                             "cutoff_utc": row["cutoff_utc"], "text": text,
                             "publisher_url": row["url"], "source": source, "review": review})
    write_jsonl(DATA / "approved_excerpts.jsonl", approved)
    result = {"scope": "Purposive discovery only; not a random coverage estimate", "approved_excerpt_count": len(approved),
              "existing_model_inputs_changed": False,
              "articles": articles, "pdfs": pdfs,
              "publication_rule": "Dated filenames, PDF creation metadata and retrospective headers alone do not establish the exact bytes available before cutoff. Archive lookup failure does not establish absence."}
    write_json(REPORT / "coverage.json", result)
    lines = ["# Rich game-note source discovery", "", "User clarified the target comparison: original boosting versus boosting probability plus game notes. The earlier neural pilot used injury excerpts only.", "",
             f"These are purposively selected source examples. **{len(approved)} reviewed pregame excerpt** is now available for constructing the future hybrid cohort. Existing model inputs are unchanged; this is not a leaguewide coverage estimate.", "",
             "| Source | Format | Pages | Decision |", "| --- | --- | ---: | --- |"]
    for row in articles + pdfs:
        lines.append(f"| [{row['team']} {row.get('game_date_et', 'rolling packet')}]({row['url']}) | {'PDF' if 'kind' in row else 'Article'} | {row.get('pages', '-')} | {row['decision']} |")
    lines += ["", "## Historical availability checks", ""]
    for row in articles + pdfs:
        if "archive_check" in row:
            lines.append(f"- {row['team']} {row['game_date_et']}: {row['archive_check']['decision']}; replay: {row['archive_version']['decision']}. Publisher modification: {row.get('modified', row['source'].get('last_modified', 'unknown'))}.")
        if row.get("manual_review"):
            lines.append(f"  Review: {row['manual_review']['review_note']}")
    lines += ["", "The live Knicks PDF is a reusable URL; it cannot stand in for historical game packets. The dated Cavaliers example provides a promising archive format. Existing 12 Cavaliers candidates and previously rejected Knicks article revisions remain separate. The 140 injury-covered games are not automatically a 140-game game-note dataset.", "",
              "The first-page Cavaliers excerpt is pinned to exact archived bytes in `config/game_notes_reviews.json`; the other 43 pages remain outside the approved excerpt. The January 28 Knicks archive is genuinely pregame, but a manual review found a wrong venue and a stale heading, so it remains quarantined. No broad claim about Knicks coverage follows from these examples.", "",
              "Next: expand pre-cutoff versions and review relevant sections, then count eligible train/dev/test games. The hybrid must be trained using out-of-sample boosting probabilities and evaluated against the unchanged booster on identical held-out games.", "",
              "Reproduce: `.venv/bin/python scripts/audit_game_notes.py --offline`. Raw downloads, provenance, and quarantined extracted text are git-ignored. No model training is performed by this audit.", ""]
    (REPORT / "findings.md").write_text("\n".join(lines))
    print(json.dumps({"pdfs": [{k: r.get(k) for k in ("team", "pages", "words", "decision")} for r in pdfs],
                      "articles": [{k: r.get(k) for k in ("game_date_et", "decision", "published", "modified", "rejection_reasons")} for r in articles],
                      "archives": [r["archive_check"]["decision"] for r in articles + pdfs if "archive_check" in r],
                      "approved_excerpts": len(approved)}, indent=2))


if __name__ == "__main__":
    main()
