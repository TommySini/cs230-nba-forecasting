"""Immutable, outcome-separated BAM inputs built from the game-notes v2 dataset.

Each prepare invocation freezes the approved notes_v2 manifest. A later
collector update requires a new snapshot. Raw byte hashes identify sources;
page/line citations are stable within the snapshot, not inferred by a model.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import io
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "bam-inputs-v3"
METADATA = ("game_id", "game_date_et", "cutoff_utc", "home_abbr", "away_abbr")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Naive timestamps are forbidden")
    return result.astimezone(timezone.utc)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def write_rows(path, rows):
    path.write_bytes(b"".join(canonical(row) for row in rows))


def availability_time(doc):
    """When these exact bytes were demonstrably public: archive capture, CDN Last-Modified, or last article edit."""
    source = doc["source"]
    if doc["kind"] == "rolling_archive":
        return parsedate_to_datetime(source["memento_datetime"]).astimezone(timezone.utc)
    if doc["kind"] == "dated_live":
        return parsedate_to_datetime(source["last_modified"]).astimezone(timezone.utc)
    if doc["kind"] == "html_post":
        return max(stamp(doc["published_utc"]), stamp(doc.get("modified_utc") or doc["published_utc"]))
    raise ValueError("Unknown notes_v2 document kind: " + doc["kind"])


def article_item(doc, root=ROOT):
    """The single article object for an HTML post inside its saved API listing page."""
    body = (Path(root) / doc["source"]["api_raw_path"]).read_bytes()
    if digest(body) != doc["source"]["api_sha256"]:
        raise ValueError("API response checksum mismatch")
    data = json.loads(body)
    found = [item for item in data.get("items") or data.get("results") or [] if item.get("permalink") == doc["url"]]
    if len(found) != 1:
        raise ValueError("Article missing or repeated in its saved API page: " + doc["url"])
    return found[0]


def source_bytes(doc, root=ROOT):
    """Exact source bytes: the raw PDF, or the canonical article object of an HTML post."""
    if doc["kind"] == "html_post":
        return canonical(article_item(doc, root))
    source = doc["source"]
    path = (Path(root) / source["raw_path"]).resolve()
    if not path.is_relative_to((Path(root) / "data/raw").resolve()):
        raise ValueError("Raw source path escapes data/raw")
    body = path.read_bytes()
    if digest(body) != source["sha256"]:
        raise ValueError("Raw source checksum mismatch")
    return body


def notes_v2_games(docs, schedule, root=ROOT):
    """One row per game holding every eligible team document, plus source bytes and exclusion counts.

    HTML posts revised after the cutoff stay out: the saved revision is not shown to
    have existed at prediction time. Games outside the numeric history are counted, not kept.
    """
    games, bodies, excluded = {}, {}, Counter()
    for doc in sorted(docs, key=lambda d: (d["game_id"], d["team"])):
        game = schedule.get(doc["game_id"])
        if game is None:
            excluded["outside_numeric_history"] += 1
            continue
        if doc.get("modified_after_cutoff"):
            excluded["html_modified_after_cutoff"] += 1
            continue
        cutoff = stamp(game["scheduled_start_utc"]) - timedelta(minutes=60)
        if stamp(doc["cutoff_utc"]) != cutoff:
            raise ValueError("Note cutoff differs from scheduled tipoff minus 60 minutes")
        available = availability_time(doc)
        if available > cutoff:
            raise ValueError("Source not shown public before cutoff: " + doc["text_path"])
        body = source_bytes(doc, root)
        sha = digest(body)
        bodies[sha] = body
        html = doc["kind"] == "html_post"
        row = games.setdefault(doc["game_id"], {**{k: game[k] for k in METADATA if k != "cutoff_utc"},
                                                "cutoff_utc": doc["cutoff_utc"], "sources": []})
        row["sources"].append({
            "source_id": "sha256:" + sha, "sha256": sha, "publisher": doc["team"], "url": doc["url"],
            "archive_time": available.isoformat().replace("+00:00", "Z"),
            "archive_url": doc.get("replay_url", doc["url"]), "evidence_tier": doc["tier"],
            "notes_v2_kind": doc["kind"], "source_kind": "html" if html else "pdf",
            "pagination": "article_sections" if html else "pdf_pages",
            "original_raw_path": doc["source"]["api_raw_path" if html else "raw_path"],
            "active_for_research": True})
    return sorted(games.values(), key=lambda r: (r["game_date_et"], r["game_id"])), bodies, excluded


PRODUCED_BY = {"train": "prior_seasons", "dev": "train_fit", "test": "train_fit"}


def attach_boosting(rows, history_by_id, splits, predictions, features, excluded):
    """Season split, schedule fields and the boosting forecast of every game, from the boosting outputs.

    A game is forecast-eligible iff the boosting pipeline wrote a prediction for it:
    fitted on earlier TRAIN seasons for TRAIN, the TRAIN-fitted model for DEV and TEST.
    """
    for row in rows:
        game = history_by_id[row["game_id"]]
        if game["season"] not in splits:
            raise ValueError(f"Season {game['season']} has no configured split")
        split = splits[game["season"]]
        prediction = predictions.get(row["game_id"])
        if prediction and (prediction["split"] != split or prediction["game_date_et"] != row["game_date_et"]
                           or prediction["produced_by"] != PRODUCED_BY[split]):
            raise ValueError("Boosting prediction disagrees with the configured split or schedule: " + row["game_id"])
        reason = None if prediction else (excluded[row["game_id"]]["reason"] if row["game_id"] in excluded
                                          else "no_out_of_sample_boosting_prediction")
        row.update(scheduled_start_utc=game["scheduled_start_utc"], season=game["season"], split=split,
                   p_boost=prediction["p_boost"] if prediction else None, forecast_eligible=prediction is not None,
                   boosting_produced_by=prediction["produced_by"] if prediction else None,
                   forecast_ineligibility_reason=reason, features=features.get(row["game_id"]))
    return rows


def pages_with_lines(pages):
    result = []
    for number, page in enumerate(pages, 1):
        if page["page"] != number or digest(page["text"].encode()) != page["text_sha256"]:
            raise ValueError("Page order or text checksum changed")
        # keepends guarantees joining line text reconstructs every source byte.
        lines = page["text"].splitlines(keepends=True) or [""]
        result.append({**page, "lines": [{"line": i, "text": text} for i, text in enumerate(lines, 1)]})
    return result


def load_snapshot_game(directory, game_id, active_only=True):
    """Materialize the research input interface without loading any outcome file.

    Call validate_snapshot once before a training run. Per-game loading checks
    input and source-page hashes against that frozen manifest again.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    body = (directory / "inputs.jsonl").read_bytes()
    if digest(body) != manifest["artifact_sha256"]["inputs.jsonl"]:
        raise ValueError("Snapshot input checksum changed")
    found = [json.loads(line) for line in body.splitlines() if json.loads(line)["game_id"] == game_id]
    if len(found) != 1:
        raise KeyError(game_id)
    game = found[0]
    sources = []
    for source in game["sources"]:
        if active_only and not source["active_for_research"]:
            continue
        content = (directory / source["pages_path"]).read_bytes()
        if digest(content) != source["pages_sha256"]:
            raise ValueError("Snapshot page checksum changed")
        packet = json.loads(content)
        if packet["source_sha256"] != source["sha256"]:
            raise ValueError("Source page hash association changed")
        sources.append({**source, "pages": packet["pages"]})
    return {**game, "sources": sources, "snapshot_id": manifest["snapshot_id"]}


def select_page_text(plain, layout):
    """Prefer layout text unless it drops characters that plain extraction recovers."""
    flags = []
    if len("".join(layout.split())) < .98 * len("".join(plain.split())):
        selected, method = plain, "pypdf_plain_layout_omitted_text"
        flags.append("layout_omitted_text_recovered_from_plain")
    elif layout.strip():
        selected, method = layout, "pypdf_layout"
    else:
        selected, method = plain, "pypdf_plain_fallback"
    if len("".join(selected.split())) < 40:
        flags.append("sparse_or_empty_text_needs_visual_review")
    if "�" in selected or re.search(r"[\x00-\x08\x0b\x0e-\x1f]", selected):
        flags.append("suspicious_glyphs_need_visual_review")
    return selected, method, flags


def article_pages(item):
    """Title plus one section per article-owned block; each section keeps the title and last heading as context."""
    from collect_notes_v2 import blocks_text
    blocks = [("title", item.get("title") or "")]
    blocks += [(block.get("type") or "block", blocks_text([block])) for block in item.get("contentExpanded") or []]
    pages = []
    for label, text in blocks:
        if text.strip():
            number = len(pages) + 1
            pages.append({"page": number, "section": number, "label": label, "page_kind": "article_section",
                          "text": text, "text_sha256": digest(text.encode()), "method": "notes_v2_article_blocks", "flags": []})
    pages = pages_with_lines(pages)
    title = pages[0] if pages and pages[0]["label"] == "title" else None
    heading = None
    for page in pages:
        context = [p for p in (title, heading) if p and p["page"] != page["page"]]
        page["context_sections"] = [{k: p[k] for k in ("section", "label", "lines")} for p in context]
        if page["label"] == "heading":
            heading = page
    return pages


def extract_pages(task):
    """Citable pages of one source: PDF pages, or article sections for an HTML post."""
    kind, raw_path, raw_hash = task
    body = Path(raw_path).read_bytes()
    if digest(body) != raw_hash:
        raise ValueError("Source changed during extraction")
    if kind == "html":
        return raw_hash, article_pages(json.loads(body))
    from pypdf import PdfReader
    pdf = gzip.decompress(body) if body.startswith(b"\x1f\x8b") else body
    pages = []
    for number, page in enumerate(PdfReader(io.BytesIO(pdf)).pages, 1):
        flags = []
        try:
            plain = page.extract_text() or ""
        except Exception as error:
            plain = ""
            flags.append("plain_extraction_failed:" + type(error).__name__)
        try:
            layout = page.extract_text(extraction_mode="layout", layout_mode_strip_rotated=False) or ""
        except Exception as error:
            layout = ""
            flags.append("layout_extraction_failed:" + type(error).__name__)
        text, method, selected_flags = select_page_text(plain, layout)
        pages.append({"page": number, "page_kind": "physical_pdf_page", "text": text, "text_sha256": digest(text.encode()),
                      "plain_text": plain, "layout_text": layout, "method": method, "flags": flags + selected_flags})
    return raw_hash, pages_with_lines(pages)


def validate_snapshot(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, expected in manifest["artifact_sha256"].items():
        if digest((directory / path).read_bytes()) != expected:
            raise ValueError("Snapshot artifact checksum changed: " + path)
    rows = read_rows(directory / "inputs.jsonl")
    labels = read_rows(directory / "labels.jsonl")
    if [r["game_id"] for r in rows] != [r["game_id"] for r in labels] or len({r["game_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate or unaligned games")
    date_splits = defaultdict(set)
    fit_ids = set(manifest["boosting_fit_game_ids"])
    for row in rows:
        date_splits[row["game_date_et"]].add(row["split"])
        if any(k in row for k in ("home_win", "home_score", "away_score", "recap", "postmortem")):
            raise ValueError("Outcome content in predictor input")
        expected = stamp(row["scheduled_start_utc"]) - timedelta(minutes=60)
        if stamp(row["cutoff_utc"]) != expected:
            raise ValueError("Prediction cutoff differs from tipoff minus 60 minutes")
        if row["split"] != manifest["season_splits"].get(str(row["season"])):
            raise ValueError("Row split differs from its season's configured split")
        if row["split"] != "train" and (row["game_date_et"] <= manifest["prediction_model_cutoff_date"] or row["game_id"] in fit_ids):
            raise ValueError("Training-only game in evaluation")
        p = row["p_boost"]
        if row["forecast_eligible"] != (p is not None) or p is not None and (not math.isfinite(p) or not 0 < p < 1):
            raise ValueError("Invalid forecast eligibility/probability")
        if row["boosting_produced_by"] != (PRODUCED_BY[row["split"]] if p is not None else None):
            raise ValueError("Boosting forecast does not come from the model its split requires")
        for source in row["sources"]:
            if stamp(source["archive_time"]) > stamp(row["cutoff_utc"]):
                raise ValueError("Source archive is later than cutoff")
            if digest((directory / source["raw_path"]).read_bytes()) != source["sha256"]:
                raise ValueError("Snapshot raw source changed")
            packet = json.loads((directory / source["pages_path"]).read_text())
            if digest((directory / source["pages_path"]).read_bytes()) != source["pages_sha256"]:
                raise ValueError("Source pages reference checksum changed")
            pages = pages_with_lines(packet["pages"])
            if pages != packet["pages"] or packet["source_sha256"] != source["sha256"]:
                raise ValueError("Page/line citations changed")
    if any(len(splits) != 1 for splits in date_splits.values()):
        raise ValueError("An ET date crosses split boundaries")
    split_order = {"train": 0, "dev": 1, "test": 2}
    if [split_order[r["split"]] for r in rows] != sorted(split_order[r["split"]] for r in rows):
        raise ValueError("Splits are not chronological")
    return manifest["summary"]


def prepare(config, snapshot_id=None, workers=4, root=ROOT):
    import joblib
    from run_gradient_boosting import load_verified_history, season_splits
    root = Path(root)
    if root.resolve() != ROOT.resolve():
        raise ValueError("Schedule verifier uses this repository root; alternate roots are unsupported")
    base_path = root / config.get("baseline_config", "config/gradient_boosting.json")
    baseline = json.loads(base_path.read_text())
    if baseline["cutoff_minutes"] != 60:
        raise ValueError("BAM requires scheduled tipoff minus 60 minutes")
    splits = season_splits(baseline)
    history_by_id = {r["game_id"]: r for r in load_verified_history(baseline)}
    boosting = root / "data/processed/gradient_boosting"
    # Only boosting outputs that match their validated metrics (validate_gradient_boosting.py).
    for name, expected in json.loads((root / "reports/gradient_boosting/metrics.json").read_text())["artifact_sha256"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Boosting artifact differs from reports/gradient_boosting/metrics.json: " + name)
    predictions = {r["game_id"]: r for r in read_rows(boosting / "predictions.jsonl")}
    features = {r["game_id"]: r["features"] for r in read_rows(boosting / "inputs.jsonl")}
    boosting_excluded = {r["game_id"]: r for r in read_rows(boosting / "excluded.jsonl")}
    fit_ids = set(joblib.load(boosting / "model.joblib")["final"]["fit_game_ids"])
    provenance = {}
    def freeze(path):
        path = Path(path)
        provenance[str(path.relative_to(root))] = digest(path.read_bytes())
    for path in (base_path, *(root / s[k] for s in baseline["histories"] for k in ("path", "config")), Path(__file__),
                 *(boosting / name for name in ("predictions.jsonl", "inputs.jsonl", "excluded.jsonl", "model.joblib")),
                 root / "scripts/run_gradient_boosting.py", root / "scripts/box_score_features.py",
                 root / "scripts/collect_pilot.py", root / "scripts/collect_notes_v2.py", root / config["notes_path"]):
        freeze(path)
    docs = read_rows(root / config["notes_path"])
    rows, bodies, excluded = notes_v2_games(docs, history_by_id, root)
    kept = {r["game_id"] for r in rows}
    for doc in docs:
        if doc["game_id"] in kept and not doc.get("modified_after_cutoff"):
            freeze(root / doc["source"]["api_raw_path" if doc["kind"] == "html_post" else "raw_path"])
    rows = attach_boosting(rows, history_by_id, splits, predictions, features, boosting_excluded)
    # Dev/test games of the earlier text pilots, whose labels and scores were already inspected.
    inspected = set(config.get("previously_inspected_game_ids", []))
    identity = {"schema_version": SCHEMA_VERSION, "config": config, "provenance_sha256": provenance,
                "runtime_versions": {name: version(name) for name in ("pypdf", "numpy", "scikit-learn", "joblib")}}
    snapshot_id = snapshot_id or digest(canonical(identity))[:20]
    if not snapshot_id or Path(snapshot_id).name != snapshot_id or snapshot_id in (".", ".."):
        raise ValueError("Snapshot ID must be one directory name")
    parent = root / "data/processed/bam_pipeline"
    final = parent / snapshot_id
    if final.exists():
        existing = json.loads((final / "manifest.json").read_text())
        if existing["identity"] != identity:
            raise FileExistsError("Snapshot ID already exists with different inputs/config")
        validate_snapshot(final)
        return final
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preparing-" + snapshot_id + "-", dir=parent))
    try:
        tasks, kinds = {}, {}
        for row in rows:
            for source in row["sources"]:
                sha = source["sha256"]
                source.update(raw_path="raw/" + sha + ".bin", pages_path="sources/" + sha + ".json")
                raw = staging / source["raw_path"]
                if not raw.exists():
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_bytes(bodies[sha])
                tasks[sha] = (source["source_kind"], str(raw), sha)
                kinds[sha] = source
        print(json.dumps({"snapshot_id": snapshot_id, "games": len(rows), "unique_sources": len(tasks),
                          "excluded_documents": dict(excluded), "stage": "extracting"}), flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for sha, pages in executor.map(extract_pages, [tasks[k] for k in sorted(tasks)]):
                write_json(staging / "sources" / (sha + ".json"), {
                    "source_sha256": sha, "source_kind": kinds[sha]["source_kind"], "pagination": kinds[sha]["pagination"],
                    "pages": pages, "visual_completeness": "Every PDF page or article section retained; figures and extraction flags require separate review."})
                print(json.dumps({"source_sha256": sha, "pages": len(pages)}), flush=True)
        for row in rows:
            for source in row["sources"]:
                source["pages_sha256"] = digest((staging / source["pages_path"]).read_bytes())
        separate_labels = [{"game_id": r["game_id"], "home_win": int(history_by_id[r["game_id"]]["home_score"] > history_by_id[r["game_id"]]["away_score"])} for r in rows]
        write_rows(staging / "inputs.jsonl", rows)
        write_rows(staging / "labels.jsonl", separate_labels)
        for split in ("train", "dev", "test"):
            membership = {r["game_id"] for r in rows if r["split"] == split}
            write_rows(staging / ("labels_" + split + ".jsonl"), [r for r in separate_labels if r["game_id"] in membership])
        reused = sorted(r["game_id"] for r in rows if r["split"] == "test" and r["game_id"] in inspected)
        summary = {"games": len(rows), "sources": len(tasks), "excluded_documents": dict(excluded),
                   "evidence_tiers": dict(Counter(s["evidence_tier"] for r in rows for s in r["sources"])),
                   "splits": dict(Counter(r["split"] for r in rows)),
                   "forecast_eligible": dict(Counter(r["split"] for r in rows if r["forecast_eligible"])),
                   "boosting_produced_by": dict(Counter(r["boosting_produced_by"] for r in rows if r["forecast_eligible"])),
                   "research_only": dict(Counter(r["forecast_ineligibility_reason"] for r in rows if not r["forecast_eligible"])),
                   "previously_inspected_test_ids": reused,
                   "test_status": "exploratory_previously_inspected" if reused else "candidate_uninspected_requires_model_knowledge_audit",
                   "full_packet_quality": "All pages preserved; notes_v2 approval checks headers and results, not full-packet semantics."}
        artifacts = {str(p.relative_to(staging)): digest(p.read_bytes()) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id, "identity": identity,
                    "season_splits": {str(season): split for season, split in sorted(splits.items())},
                    "prediction_model_cutoff_date": config["prediction_model_cutoff_date"],
                    "boosting_fit_game_ids": sorted(fit_ids), "artifact_sha256": artifacts, "summary": summary}
        write_json(staging / "manifest.json", manifest)
        validate_snapshot(staging)
        # Check input files again before publishing an atomically complete snapshot.
        for name, expected in provenance.items():
            if digest((root / name).read_bytes()) != expected:
                raise ValueError("Input changed during preparation; retry after collector publishes: " + name)
        staging.rename(final)
        report = root / "reports/bam_pipeline" / snapshot_id
        report.mkdir(parents=True, exist_ok=False)
        write_json(report / "data_summary.json", summary)
        write_json(report / "snapshot_manifest.json", manifest)
        print(json.dumps(summary, indent=2), flush=True)
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--snapshot-id")
    prepare_parser.add_argument("--workers", type=int, default=4)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.workers < 1:
            parser.error("workers must be positive")
        print(prepare(json.loads(args.config.read_text()), args.snapshot_id, args.workers))
    else:
        print(json.dumps(validate_snapshot(args.directory), indent=2))


if __name__ == "__main__":
    main()
