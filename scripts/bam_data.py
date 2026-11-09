"""Outcome-separated BAM views over shared games, features, forecasts and notes.

The shared tables preserve the admitted corpus and exact page/line citations.
Legacy execution bundles remain readable for validation during migration.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

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
    from nba_dataset import resolve_source_path
    body = resolve_source_path(root, doc["source"]["api_raw_path"]).read_bytes()
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
    from nba_dataset import resolve_source_path
    path = resolve_source_path(root, source["raw_path"]).resolve()
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


def load_snapshot_game(directory, game_id, active_only=True, root=ROOT):
    """Materialize the research input interface without loading any outcome file.

    Call validate_snapshot once before a training run. Per-game loading checks
    input and source-page hashes against that frozen manifest again.
    """
    directory = Path(directory)
    if (directory / "bam_manifest.json").exists():
        manifest = _verify_shared_files(directory, root=root)
        found = [row for row in read_snapshot_inputs(directory, root=root) if row["game_id"] == game_id]
        if len(found) != 1:
            raise KeyError(game_id)
        game = found[0]
        sources = []
        for source in game["sources"]:
            if active_only and not source["active_for_research"]:
                continue
            packet = read_note_packet(directory, source["sha256"])
            if digest(canonical(packet)) != source["pages_sha256"]:
                raise ValueError("Source page checksum changed")
            sources.append({**source, "pages": packet["pages"]})
        return {**game, "sources": sources, "snapshot_id": manifest["snapshot_id"]}
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


def validate_snapshot(directory, root=ROOT, *, label_splits=("train", "dev", "test")):
    directory = Path(directory)
    if (directory / "bam_manifest.json").exists():
        return validate_shared_notes(directory, root=root, label_splits=label_splits)
    requested = set(label_splits)
    if not requested <= {"train", "dev", "test"}:
        raise ValueError("Unknown BAM label split")
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, expected in manifest["artifact_sha256"].items():
        if path.startswith("labels") and requested != {"train", "dev", "test"}:
            if path not in {f"labels_{split}.jsonl" for split in requested}:
                continue
        if digest((directory / path).read_bytes()) != expected:
            raise ValueError("Snapshot artifact checksum changed: " + path)
    rows = read_rows(directory / "inputs.jsonl")
    label_rows = [row for row in rows if row["split"] in requested]
    if requested == {"train", "dev", "test"}:
        labels = read_rows(directory / "labels.jsonl")
    else:
        labels = []
        for split in ("train", "dev", "test"):
            if split not in requested:
                continue
            path = directory / f"labels_{split}.jsonl"
            if not path.exists():
                raise ValueError("Scoped legacy validation requires separate outcome partitions")
            labels.extend(read_rows(path))
    if [r["game_id"] for r in label_rows] != [r["game_id"] for r in labels] or len({r["game_id"] for r in rows}) != len(rows):
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


def snapshot_manifest_path(directory):
    directory = Path(directory)
    return directory / ("bam_manifest.json" if (directory / "bam_manifest.json").exists() else "manifest.json")


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_shared_files(directory, root=ROOT):
    """Verify the outcome-blind view's dependencies; never open a label store."""
    directory = Path(directory)
    manifest = json.loads((directory / "bam_manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        if Path(name).name != name or "label" in name:
            raise ValueError("Invalid outcome-blind BAM dependency")
        if _file_hash(directory / name) != expected:
            raise ValueError("Shared BAM artifact checksum changed: " + name)
    # Other models append to the shared prediction table. The main dataset manifest
    # pins its current bytes; BAM pins only its original boosting cohort and values.
    from nba_dataset import Dataset
    predictions = Dataset(root=root, processed_dir=directory).predictions("gradient_boosting", stage="baseline")
    by_id = {}
    for row in predictions:
        if row["game_id"] in by_id:
            raise ValueError("Duplicate gradient forecast in shared table")
        by_id[row["game_id"]] = row
    relevant = [by_id[row["game_id"]] for row in manifest["cohort"] if row["game_id"] in by_id]
    if digest(canonical(relevant)) != manifest["gradient_forecasts_sha256"]:
        raise ValueError("BAM gradient forecast cohort or values changed")
    return manifest


def _page_schema():
    import pyarrow as pa
    line = pa.struct([("line", pa.int32()), ("text", pa.string())])
    context = pa.struct([("section", pa.int32()), ("label", pa.string()), ("lines", pa.list_(line))])
    return pa.schema([
        ("source_sha256", pa.string()), ("_fields", pa.list_(pa.string())),
        ("page", pa.int32()), ("page_kind", pa.string()), ("text", pa.large_string()),
        ("text_sha256", pa.string()), ("method", pa.string()), ("flags", pa.list_(pa.string())),
        ("plain_text", pa.large_string()), ("layout_text", pa.large_string()),
        ("lines", pa.list_(line)), ("section", pa.int32()), ("label", pa.string()),
        ("context_sections", pa.list_(context)),
    ])


def read_note_packet(directory, source_sha256):
    import pyarrow.parquet as pq
    directory = Path(directory)
    table = pq.read_table(directory / "note_pages.parquet", filters=[("source_sha256", "=", source_sha256)])
    pages = [{key: row[key] for key in row["_fields"]} for row in table.to_pylist()]
    if not pages:
        raise KeyError(source_sha256)
    pages.sort(key=lambda page: page["page"])
    notes = pq.read_table(directory / "notes.parquet", filters=[("sha256", "=", source_sha256)],
                          columns=["packet_metadata"]).to_pylist()
    metadata = json.loads(notes[0]["packet_metadata"])
    return {**metadata, "pages": pages}


def read_snapshot_inputs(directory, root=ROOT, *, legacy_paths=False, splits=None):
    """Reconstruct admitted pregame rows by ID without opening outcomes."""
    from nba_dataset import Dataset, read_rows as table_rows
    directory = Path(directory)
    allowed = {"train", "dev", "test"} if splits is None else set(splits)
    if not allowed <= {"train", "dev", "test"}:
        raise ValueError("Unknown BAM split")
    if not (directory / "bam_manifest.json").exists():
        return [row for row in read_rows(directory / "inputs.jsonl") if row["split"] in allowed]
    manifest = json.loads((directory / "bam_manifest.json").read_text())
    dataset = Dataset(root=root, processed_dir=directory)
    features = {row["game_id"]: row["features"] for row in dataset.inputs(splits=allowed)}
    predictions = {row["game_id"]: row for row in dataset.predictions("gradient_boosting", stage="baseline")}
    sources = defaultdict(list)
    for note in table_rows(directory / "notes.parquet"):
        source = {key: note[key] for key in note["source_fields"]}
        source["archive_time"] = stamp(note["archive_time"]).isoformat().replace("+00:00", "Z")
        if legacy_paths:
            source["raw_path"] = note["legacy_raw_path"]
            source["pages_path"] = note["legacy_pages_path"]
        sources[note["game_id"]].append((note["source_order"], source))
    rows = []
    for cohort in manifest["cohort"]:
        if cohort["split"] not in allowed:
            continue
        game_id = cohort["game_id"]
        prediction = predictions.get(game_id)
        row = {**cohort, "features": features.get(game_id),
               "p_boost": prediction["p_boost"] if prediction else None,
               "sources": [source for _, source in sorted(sources[game_id])]}
        if row["forecast_eligible"] != (prediction is not None):
            raise ValueError("Shared forecast membership differs from admitted BAM cohort")
        rows.append(row)
    return rows


def _admitted_source_bytes(source, root):
    from nba_dataset import resolve_source_path
    body = resolve_source_path(root, source["original_raw_path"]).read_bytes()
    if source["source_kind"] == "html":
        listing = json.loads(body)
        items = [item for item in listing.get("items") or listing.get("results") or []
                 if item.get("permalink") == source["url"]]
        if len(items) != 1:
            raise ValueError("Admitted article missing or repeated in original listing")
        return canonical(items[0])
    return body


def validate_shared_notes(directory, root=ROOT, *, label_splits=("train", "dev", "test")):
    """Check migrated corpus, exact citations and source admission without new extraction."""
    from nba_dataset import Dataset
    directory = Path(directory)
    manifest = _verify_shared_files(directory, root=root)
    rows = read_snapshot_inputs(directory, root=root, legacy_paths=True)
    if digest(canonical(rows)) != manifest["logical_inputs_sha256"]:
        raise ValueError("Shared BAM inputs differ from the frozen corpus")
    if not set(label_splits) <= {"train", "dev", "test"}:
        raise ValueError("Unknown BAM label split")
    if label_splits:
        dataset = Dataset(root=root, processed_dir=directory)
        labels = {row["game_id"]: row for row in dataset.labels(label_splits, eligible_only=False)}
        for split in label_splits:
            ordered_labels = [{"game_id": row["game_id"], "home_win": labels[row["game_id"]]["home_win"]}
                              for row in rows if row["split"] == split]
            if digest(canonical(ordered_labels)) != manifest["logical_labels_sha256_by_split"][split]:
                raise ValueError("Shared BAM labels differ from the frozen corpus")
    seen = set()
    for row in rows:
        if any(key in row for key in ("home_win", "home_score", "away_score", "recap", "postmortem")):
            raise ValueError("Outcome content in predictor input")
        if stamp(row["scheduled_start_utc"]) - stamp(row["cutoff_utc"]) != timedelta(minutes=60):
            raise ValueError("Prediction cutoff differs from tipoff minus 60 minutes")
        if manifest["season_splits"].get(str(row["season"])) != row["split"]:
            raise ValueError("BAM season split changed")
        for source in row["sources"]:
            if stamp(source["archive_time"]) > stamp(row["cutoff_utc"]):
                raise ValueError("Source archive is later than cutoff")
            if source["sha256"] in seen:
                continue
            seen.add(source["sha256"])
            if digest(_admitted_source_bytes(source, root)) != source["sha256"]:
                raise ValueError("Original note source checksum changed")
            packet = read_note_packet(directory, source["sha256"])
            if digest(canonical(packet)) != source["pages_sha256"]:
                raise ValueError("Migrated page packet checksum changed")
            if pages_with_lines(packet["pages"]) != packet["pages"]:
                raise ValueError("Page/line citations changed")
    return manifest["summary"]


def migrate_notes(root, staging):
    """Losslessly stage existing note records/pages; never extract PDFs or write raw sources."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nba_dataset import write_table, resolve_source_path
    root, staging = Path(root), Path(staging)
    old = root / "data/processed/bam_pipeline/bam_notes_v2"
    old_notes = root / "data/processed/notes_v2"
    manifest = json.loads((old / "manifest.json").read_text())
    rows, labels = read_rows(old / "inputs.jsonl"), read_rows(old / "labels.jsonl")
    known_bam = {"manifest.json", "inputs.jsonl", "labels.jsonl", "labels_train.jsonl", "labels_dev.jsonl", "labels_test.jsonl"}
    known_bam.update(source[key] for row in rows for source in row["sources"] for key in ("raw_path", "pages_path"))
    known_notes = {"candidates.jsonl", "documents.jsonl", "approved.jsonl", "html_posts.jsonl", "review_summary.json", "fetch.log"}
    known_notes.update(str(path.relative_to(old_notes)) for path in (old_notes / "texts").glob("*.txt"))
    for directory, known in ((old, known_bam), (old_notes, known_notes)):
        unknown = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()} - known
        if unknown:
            raise ValueError("Unmapped notes cleanup files: " + str(sorted(unknown)))
    for name in ("inputs.jsonl", "labels.jsonl"):
        if _file_hash(old / name) != manifest["artifact_sha256"][name]:
            raise ValueError("Frozen BAM migration input changed: " + name)
    notes, packet_hashes, seen, input_hashes, coverage = [], {}, set(), {}, {}
    def cover(path, replacement, expected=None):
        relative = str(path.relative_to(root))
        actual = _file_hash(path) if expected is None else expected
        input_hashes[relative] = actual
        coverage[relative] = {"sha256": actual, "replacement": replacement}
    cover(old / "manifest.json", "bam_manifest.json:legacy_manifest")
    cover(old / "inputs.jsonl", "bam_manifest.json:cohort + shared features/predictions/notes; logical_inputs_sha256")
    cover(old / "labels.jsonl", "labels_train/dev/test.parquet; logical_labels_sha256")
    for split in ("train", "dev", "test"):
        path = old / f"labels_{split}.jsonl"
        if path.exists():
            expected = [label for row, label in zip(rows, labels) if row["split"] == split]
            if read_rows(path) != expected or _file_hash(path) != manifest["artifact_sha256"][path.name]:
                raise ValueError("Frozen BAM label partition changed: " + split)
            cover(path, f"labels_{split}.parquet; logical_labels_sha256_by_split")
    source_keys = set()
    page_schema = _page_schema()
    with pq.ParquetWriter(staging / "note_pages.parquet", page_schema, compression="zstd") as writer:
        for row in rows:
            for order, source in enumerate(row["sources"]):
                source_keys.update(source)
                packet_path = old / source["pages_path"]
                if _file_hash(packet_path) != source["pages_sha256"]:
                    raise ValueError("Frozen note page checksum changed")
                packet = json.loads(packet_path.read_text())
                if digest(canonical(packet)) != source["pages_sha256"]:
                    raise ValueError("Page packet is not in the expected canonical encoding")
                if digest(_admitted_source_bytes(source, root)) != source["sha256"]:
                    raise ValueError("Original note bytes differ from admitted snapshot")
                if _file_hash(old / source["raw_path"]) != source["sha256"]:
                    raise ValueError("Frozen snapshot raw copy changed")
                cover(packet_path, "note_pages.parquet + notes.parquet:packet_metadata; sha256=" + source["sha256"],
                      expected=source["pages_sha256"])
                cover(old / source["raw_path"], "original source " + source["original_raw_path"] +
                      ("; canonical article selected by URL " + source["url"] if source["source_kind"] == "html" else ""),
                      expected=source["sha256"])
                if source["sha256"] not in seen:
                    seen.add(source["sha256"])
                    if pages_with_lines(packet["pages"]) != packet["pages"]:
                        raise ValueError("Frozen citation line structure changed")
                    pages = [{"source_sha256": source["sha256"], "_fields": list(page), **page}
                             for page in packet["pages"]]
                    unknown = set().union(*(set(page) for page in pages)) - set(page_schema.names)
                    if unknown:
                        raise ValueError("Unmapped page fields: " + str(unknown))
                    writer.write_table(pa.Table.from_pylist(pages, schema=page_schema))
                    packet_hashes[source["sha256"]] = source["pages_sha256"]
                notes.append({**source, "game_id": row["game_id"], "source_order": order,
                              "source_fields": list(source), "archive_time": stamp(source["archive_time"]),
                              "legacy_raw_path": source["raw_path"], "legacy_pages_path": source["pages_path"],
                              "raw_path": source["original_raw_path"],
                              "pages_path": "note_pages.parquet#" + source["sha256"],
                              "packet_metadata": json.dumps({k: v for k, v in packet.items() if k != "pages"}, sort_keys=True)})
    fields = [(key, pa.bool_() if key == "active_for_research" else
               pa.timestamp("us", tz="UTC") if key == "archive_time" else pa.string()) for key in sorted(source_keys)]
    schema = pa.schema(fields + [("game_id", pa.string()), ("source_order", pa.int32()),
                                 ("source_fields", pa.list_(pa.string())), ("legacy_raw_path", pa.string()),
                                 ("legacy_pages_path", pa.string()), ("packet_metadata", pa.string())])
    write_table(staging / "notes.parquet", notes, schema=schema)
    # Heterogeneous rejection/header checks remain inspectable metadata; lengthy text is stored once.
    audit = {name: read_rows(old_notes / (name + ".jsonl"))
             for name in ("candidates", "documents", "approved", "html_posts")}
    text_rows = []
    for path in sorted((old_notes / "texts").glob("*.txt")):
        raw = path.read_bytes()
        text_rows.append({"text_path": str(path.relative_to(root)), "text": raw.decode("utf-8"), "sha256": digest(raw)})
        cover(path, "note_texts.parquet:text_path=" + str(path.relative_to(root)), expected=digest(raw))
    for number, post in enumerate(audit["html_posts"]):
        text = post.pop("text")
        text_id = "html-post:" + str(number)
        text_rows.append({"text_path": text_id, "text": text, "sha256": digest(text.encode())})
        post["text_id"] = text_id
    write_table(staging / "note_texts.parquet", text_rows,
                schema=pa.schema([("text_path", pa.string()), ("text", pa.large_string()), ("sha256", pa.string())]))
    audit["review_summary"] = json.loads((old_notes / "review_summary.json").read_text())
    if (old_notes / "fetch.log").exists():
        audit["collector_logs"] = {"fetch.log": (old_notes / "fetch.log").read_bytes().decode("utf-8")}
        cover(old_notes / "fetch.log", "note_audit.json:collector_logs.fetch.log")
    write_json(staging / "note_audit.json", audit)
    for path in sorted(old_notes.glob("*.json*")):
        cover(path, "note_audit.json:" + path.stem + (" + note_texts.parquet" if path.stem == "html_posts" else ""))
    dependent = ("notes.parquet", "note_pages.parquet", "note_texts.parquet", "note_audit.json",
                 "features.parquet", "games.parquet", "feature_lineage.parquet")
    compact = {key: manifest[key] for key in ("snapshot_id", "season_splits", "prediction_model_cutoff_date",
                                             "boosting_fit_game_ids", "summary", "identity")}
    compact.update(schema_version="bam-shared-v1",
                   cohort=[{key: value for key, value in row.items() if key not in ("features", "sources", "p_boost")} for row in rows],
                   logical_inputs_sha256=digest(canonical(rows)), logical_labels_sha256=digest(canonical(labels)),
                   gradient_forecasts_sha256=digest(canonical([
                       {**{key: row[key] for key in ("game_id", "season", "game_date_et", "split", "p_boost")},
                        "produced_by": row["boosting_produced_by"]} for row in rows if row["p_boost"] is not None])),
                   logical_labels_sha256_by_split={
                       split: digest(canonical([label for row, label in zip(rows, labels) if row["split"] == split]))
                       for split in ("train", "dev", "test")},
                   legacy_manifest=manifest, source_sha256=input_hashes, cleanup_coverage=coverage,
                   artifact_sha256={name: _file_hash(staging / name) for name in dependent if (staging / name).exists()})
    write_json(staging / "bam_manifest.json", compact)
    # Round-trip every source packet, including optional page fields and exact line endings.
    for source_hash, expected in packet_hashes.items():
        if digest(canonical(read_note_packet(staging, source_hash))) != expected:
            raise ValueError("Parquet changed a note packet: " + source_hash)
    return {"games": len(rows), "documents": len(notes), "sources": len(seen), "reviewed_texts": len(text_rows),
            "source_sha256": input_hashes, "cleanup_coverage": coverage,
            "remove_candidates": [str(old.relative_to(root)), str(old_notes.relative_to(root))], "install_files": {}}


def prepare(config, snapshot_id=None, workers=4, root=ROOT):
    """Return the validated shared view; preparation no longer copies a dataset."""
    directory = Path(root) / config.get("dataset_path", "data/processed")
    if not (directory / "bam_manifest.json").exists():
        raise FileNotFoundError("Migrate the shared dataset with nba_dataset.py before preparing BAM")
    manifest = _verify_shared_files(directory, root=root)
    if snapshot_id is not None and snapshot_id != manifest["snapshot_id"]:
        raise ValueError("BAM view identity is fixed by the migrated dataset")
    if config.get("prediction_model_cutoff_date") != manifest["prediction_model_cutoff_date"]:
        raise ValueError("Prediction model cutoff differs from the admitted corpus")
    validate_shared_notes(directory, root=root, label_splits=())
    return directory


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
