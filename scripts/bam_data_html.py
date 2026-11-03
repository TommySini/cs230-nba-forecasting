"""Expanded immutable BAM snapshots with explicit PDF pages / HTML sections.

This independent v2 builder leaves the active v1 builder and its snapshot frozen.
It reuses pure feature/split/source helpers, never rewrites their global paths.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import bam_data as base
from bam_html_notes import parse_html_note

ROOT = base.ROOT
SCHEMA_VERSION = "bam-inputs-v2.1-html-sections"


def parse_article_source(raw, provenance, game, prior_records=None):
    if provenance.get("publisher") == "POR":
        from bam_html_por import parse_por_html_note
        if prior_records is None:
            raise ValueError("Portland needs validated official prior records")
        return parse_por_html_note(raw, provenance, game, prior_records)
    return parse_html_note(raw, provenance, game)


def verify_content_review(reviewed, game_id, sha, sections):
    if (reviewed.get("decision") != "approved_source_for_pregame_research"
            or reviewed.get("game_id") != game_id or reviewed.get("source_sha256") != sha
            or reviewed.get("sections_sha256") != base.digest(base.canonical(sections))
            or reviewed.get("reviewed_sections") != [section["section"] for section in sections]):
        raise ValueError("HTML needs exact full-article approval, not a quarantined or partial review")


def verify_html_note(note, game, prior_records=None):
    if note.get("source_kind") != "html" or note.get("review", {}).get("decision") != "approved_pregame_html_article":
        raise ValueError("HTML requires explicit source kind and content approval")
    source = note["source"]
    path = (ROOT / source["raw_path"]).resolve()
    if not path.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("HTML path escapes raw source directory")
    body = path.read_bytes()
    if note["team"] == "POR":
        review = note["review"]
        if base.digest((ROOT / "scripts/bam_html_por.py").read_bytes()) != review["publisher_parser_sha256"]:
            raise ValueError("Portland adapter changed after source content approval")
        if prior_records is not None and prior_records != review["header_prior_records"]:
            raise ValueError("Portland reviewed prior records differ from official history")
        prior_records = prior_records if prior_records is not None else review["header_prior_records"]
    parsed = parse_article_source(body, {**source, "publisher_url": note["publisher_url"], "publisher": note["team"]}, game, prior_records)
    if parsed["eligibility"]["status"] != "verified_pregame":
        raise ValueError("HTML source no longer passes strict pregame validation")
    review = note["review"]
    if base.digest(base.canonical(parsed["sections"])) != review["sections_sha256"]:
        raise ValueError("HTML section extraction differs from reviewed content")
    if base.digest((ROOT / review["content_review_path"]).read_bytes()) != review["content_review_sha256"]:
        raise ValueError("HTML content-review evidence changed")
    reviewed = json.loads((ROOT / review["content_review_path"]).read_text())
    verify_content_review(reviewed, note["game_id"], source["sha256"], parsed["sections"])
    if base.digest((ROOT / "scripts/bam_html_notes.py").read_bytes()) != review["parser_sha256"]:
        raise ValueError("HTML parser changed after source content approval")
    return body, parsed


def sections_as_pages(sections, with_context=True, context_map=None):
    """The citation index calls these pages; metadata explicitly says sections."""
    pages = []
    title = next((section for section in sections if section["label"] == "title"), None)
    heading = None
    for number, section in enumerate(sections, 1):
        if section["section"] != number or base.digest(section["text"].encode()) != section["text_sha256"]:
            raise ValueError("HTML section numbering/text checksum changed")
        if "".join(line["text"] for line in section["lines"]) != section["text"]:
            raise ValueError("HTML source lines do not reconstruct the section")
        pages.append({"page": number, "section": number, "label": section["label"], "page_kind": "article_section",
                      "text": section["text"], "text_sha256": section["text_sha256"], "lines": section["lines"],
                      "method": "strict_archived_article_owned_blocks", "flags": []})
        if with_context:
            context = [item for item in (title, heading) if item and item["section"] != number]
            if context_map is not None:
                context = [sections[index - 1] for index in context_map[str(number)]]
            pages[-1]["context_sections"] = [{k: item[k] for k in ("section", "label", "lines")} for item in context]
        if section["label"] == "heading":
            heading = section
    return pages


def extract_source(task):
    source, note, game, cache, overrides = task
    sha = source["sha256"]
    kind = note.get("source_kind", "pdf")
    if kind == "html":
        _, parsed = verify_html_note(note, game)
        pages = sections_as_pages(parsed["sections"], context_map=parsed.get("publisher_adapter", {}).get("context_sections"))
        audit = {k: parsed[k] for k in ("title", "article_url", "article_date", "article_modified", "archive_time", "excluded_blocks", "text_sha256", "parser_version")}
        if "publisher_adapter" in parsed:
            audit.update(publisher_adapter=parsed["publisher_adapter"], header_prior_records=note["review"]["header_prior_records"])
        return sha, {"source_sha256": sha, "source_kind": "html", "pagination": "article_sections", "pages": pages,
                     "article_audit": audit}
    if kind != "pdf":
        raise ValueError("Unsupported source kind")
    if cache:
        path, expected = cache
        body = Path(path).read_bytes()
        if base.digest(body) != expected:
            raise ValueError("Frozen PDF extraction cache changed")
        packet = json.loads(body)
        if packet["source_sha256"] != sha:
            raise ValueError("Frozen extraction cache belongs to another raw source")
        pages = base.pages_with_lines(packet["pages"])
    else:
        _, pages = base.extract_pages((str(ROOT / source["raw_path"]), sha, None, overrides))
    return sha, {"source_sha256": sha, "source_kind": "pdf", "pagination": "pdf_pages", "pages": pages,
                 "visual_completeness": "Every PDF page retained; image/table extraction limitations remain visible in flags."}


def validate_snapshot(directory):
    result = base.validate_snapshot(directory)
    directory = Path(directory)
    schema = json.loads((directory / "manifest.json").read_text())["schema_version"]
    for row in base.read_rows(directory / "inputs.jsonl"):
        for source in row["sources"]:
            kind = source.get("source_kind")
            packet = json.loads((directory / source["pages_path"]).read_text())
            if packet["source_kind"] != kind:
                raise ValueError("Source kind/extraction mismatch")
            if kind == "html":
                if source["pagination"] != "article_sections" or any(p.get("page_kind") != "article_section" for p in packet["pages"]):
                    raise ValueError("HTML must use explicit article-section metadata")
                raw = (directory / source["raw_path"]).read_bytes()
                parsed = parse_article_source(raw, {**source["archive_provenance"], "publisher_url": source["url"], "publisher": source["publisher"]}, row,
                                              packet.get("article_audit", {}).get("header_prior_records"))
                if parsed["eligibility"]["status"] != "verified_pregame" or sections_as_pages(parsed["sections"], with_context=schema != "bam-inputs-v2-html-sections", context_map=parsed.get("publisher_adapter", {}).get("context_sections")) != packet["pages"]:
                    raise ValueError("Frozen HTML differs from strict article-only parsing")
                if packet.get("article_audit", {}).get("publisher_adapter") != parsed.get("publisher_adapter"):
                    raise ValueError("Publisher article header/context audit changed")
                if "article_audit" in packet and packet["article_audit"]["excluded_blocks"] != parsed["excluded_blocks"]:
                    raise ValueError("HTML nontext-block exclusion audit changed")
            elif kind != "pdf" or source["pagination"] != "pdf_pages":
                raise ValueError("Unknown or mislabeled source pagination")
    return result


def prepare(config, notes_paths, snapshot_id=None, workers=8, cache_snapshot=None):
    import joblib
    from run_gradient_boosting import (FEATURE_NAMES, arrays, load_verified_history, new_model, prepare_examples)
    config = {**config, "notes_paths": list(notes_paths)}
    baseline_path = ROOT / config.get("baseline_config", "config/gradient_boosting.json")
    baseline = json.loads(baseline_path.read_text())
    if baseline["cutoff_minutes"] != 60:
        raise ValueError("Prediction cutoff must be tipoff minus 60 minutes")
    history = load_verified_history(baseline)
    numeric, labels, _ = prepare_examples(history, baseline)
    games = {g["game_id"]: g for g in history}
    by_id, y_by_id = {r["game_id"]: r for r in numeric}, {r["game_id"]: r for r in labels}
    model_path = ROOT / "data/processed/gradient_boosting/model.joblib"
    original = joblib.load(model_path)
    fit_ids = set(original["fit_game_ids"]) | set(original.get("selection_game_ids", []))
    provenance = {}
    def freeze(path):
        path = Path(path)
        provenance[str(path.relative_to(ROOT))] = base.digest(path.read_bytes())
    for path in (baseline_path, model_path, ROOT / baseline["history_path"], ROOT / baseline["history_config"],
                 Path(__file__), ROOT / "scripts/bam_data.py", ROOT / "scripts/bam_html_notes.py",
                 ROOT / "scripts/run_gradient_boosting.py", ROOT / "scripts/collect_pilot.py", ROOT / "scripts/prepare_full_game_notes.py"):
        freeze(path)
    for name in ("scripts/audit_game_notes.py", "scripts/review_game_notes.py", "scripts/bam_collect_notes.py",
                 "scripts/bam_pdf_notes.py", "scripts/bam_html_por.py",
                 "config/game_notes_content_review.json", "config/bam_notes_expansion.json"):
        if (ROOT / name).exists():
            freeze(ROOT / name)
    expansion_config_path = ROOT / "config/bam_notes_expansion.json"
    expansion_config = {}
    if expansion_config_path.exists():
        expansion_config = json.loads(expansion_config_path.read_text())
        if expansion_config.get("official_record_policy_path"):
            freeze(ROOT / expansion_config["official_record_policy_path"])
        for review in expansion_config.get("additional_template_reviews", []):
            freeze(ROOT / review["image_path"])
            if base.digest((ROOT / review["image_path"]).read_bytes()) != review["image_sha256"]:
                raise ValueError("Expanded publisher-template visual proof changed")
    notes, bodies, kinds, games_for_notes = [], {}, {}, {}
    official_history = None
    for path in notes_paths:
        freeze(ROOT / path)
        notes.extend(base.read_rows(ROOT / path))
    for note in notes:
        game = games.get(note["game_id"])
        if game is None or any(note[k] != game[k] for k in ("game_date_et", "home_abbr", "away_abbr")):
            raise ValueError("Source note differs from verified schedule")
        if base.stamp(note["cutoff_utc"]) != base.stamp(game["scheduled_start_utc"]) - timedelta(minutes=60):
            raise ValueError("Source cutoff differs from tipoff minus 60 minutes")
        safe_game = {**{k: game[k] for k in ("game_id", "game_date_et", "scheduled_start_utc", "season", "home_abbr", "away_abbr")}, "cutoff_utc": note["cutoff_utc"]}
        kind = note.get("source_kind", "pdf")
        if kind == "html":
            records = None
            if note["team"] == "POR":
                from bam_collect_notes import official_record_history
                from review_game_notes import prior_record
                if official_history is None:
                    official_history, _ = official_record_history(history, json.loads((ROOT / expansion_config["official_record_policy_path"]).read_text()))
                records = {abbr: list(prior_record(game, abbr, official_history)) for abbr in (game["home_abbr"], game["away_abbr"])}
            body, _ = verify_html_note(note, safe_game, records)
            freeze(ROOT / note["review"]["content_review_path"])
        elif kind == "pdf":
            body = base.verified_bytes(note)
            review_sha = note.get("review", {}).get("review_config_sha256")
            if review_sha and note.get("source_kind") == "pdf":
                review_path = ROOT / "data/raw" / expansion_config.get("namespace", "bam_notes_expansion/v1") / "review_config_versions" / (review_sha + ".json")
                if base.digest(review_path.read_bytes()) != review_sha:
                    raise ValueError("Expanded PDF template-review configuration changed")
                freeze(review_path)
        else:
            raise ValueError("Unknown note source kind")
        sha = note["source"]["sha256"]
        if sha in kinds and kinds[sha] != kind:
            raise ValueError("Same raw bytes have conflicting source kinds")
        kinds[sha], bodies[sha], games_for_notes[note["game_id"]] = kind, body, safe_game
        freeze(ROOT / note["source"]["raw_path"])
    rows = base.merge_notes(notes)
    warnings_by_sha = {note["source"]["sha256"]: note.get("review", {}).get("source_warnings", []) for note in notes}
    plans, assignments = base.prior_probability_plan(rows, numeric, set(original["fit_game_ids"]), config.get("minimum_oof_fit_games", 200))
    parameters = config.get("boosting_parameters", {"n_estimators": 100, "max_depth": 1})
    expected = original["model"].get_params()
    if any(expected[k] != v for k, v in {**baseline["model_parameters"], **parameters, "random_state": baseline["seed"]}.items()):
        raise ValueError("OOF recipe differs from original frozen booster")
    for fold, detail in plans.items():
        prior = [by_id[gid] for gid in detail["fit_game_ids"]]
        x, y = arrays(prior, [y_by_id[r["game_id"]] for r in prior])
        model = original["model"] if fold == "original_frozen" else new_model(baseline, parameters).fit(x, y)
        for row in rows:
            if assignments[row["game_id"]]["fold"] == fold:
                row["p_boost"] = float(model.predict_proba([[by_id[row["game_id"]]["features"][name] for name in FEATURE_NAMES]])[0, 1])
    for row in rows:
        gid = row["game_id"]
        row.update(scheduled_start_utc=games[gid]["scheduled_start_utc"], season=games[gid]["season"], p_boost=row.get("p_boost"),
                   forecast_eligible=row.get("p_boost") is not None, boosting_fold=assignments[gid]["fold"],
                   forecast_ineligibility_reason=assignments[gid]["reason"])
        if gid in by_id:
            row.update(features=by_id[gid]["features"], prior_game_ids=by_id[gid]["prior_game_ids"])
        for source in row["sources"]:
            source.update(source_kind=kinds[source["sha256"]], pagination="article_sections" if kinds[source["sha256"]] == "html" else "pdf_pages")
            if warnings_by_sha[source["sha256"]]:
                source["source_warnings"] = warnings_by_sha[source["sha256"]]
    policy = config["split_policy"]
    rows = base.assign_splits(rows, fit_ids, policy)
    if config.get("minimum_eval_date") and any(row["split"] != "train" and row["game_date_et"] < config["minimum_eval_date"] for row in rows):
        raise ValueError("Heldout game predates prediction-time model eligibility; freeze a consistent split policy")
    overrides_path = ROOT / "config/full_game_notes_extraction_overrides.json"
    overrides = json.loads(overrides_path.read_text())["pages"] if overrides_path.exists() else {}
    if overrides_path.exists():
        freeze(overrides_path)
        for value in overrides.values():
            freeze(ROOT / value["image_path"])
            if base.digest((ROOT / value["image_path"]).read_bytes()) != value["image_sha256"]:
                raise ValueError("Reviewed visual transcription evidence changed")
    cache = {}
    if cache_snapshot:
        cache_snapshot = Path(cache_snapshot).resolve()
        base.validate_snapshot(cache_snapshot)
        cache_manifest = json.loads((cache_snapshot / "manifest.json").read_text())
        freeze(cache_snapshot / "manifest.json")
        same_overrides = cache_manifest["identity"]["provenance_sha256"].get("config/full_game_notes_extraction_overrides.json") == provenance.get("config/full_game_notes_extraction_overrides.json")
        if same_overrides:
            for row in base.read_rows(cache_snapshot / "inputs.jsonl"):
                for source in row["sources"]:
                    if source.get("source_kind", "pdf") == "pdf":
                        cache[source["sha256"]] = (str(cache_snapshot / source["pages_path"]), source["pages_sha256"])
    identity = {"schema_version": SCHEMA_VERSION, "config": config, "provenance_sha256": provenance,
                "runtime_versions": {name: version(name) for name in ("pypdf", "numpy", "scikit-learn", "joblib")}}
    snapshot_id = snapshot_id or base.digest(base.canonical(identity))[:20]
    if Path(snapshot_id).name != snapshot_id or snapshot_id in (".", ".."):
        raise ValueError("Snapshot ID must be one directory name")
    parent, report_parent = ROOT / "data/processed/bam_pipeline", ROOT / "reports/bam_pipeline"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / snapshot_id
    if final.exists():
        if json.loads((final / "manifest.json").read_text())["identity"] != identity:
            raise FileExistsError("Expanded snapshot ID has another identity")
        validate_snapshot(final)
        return final
    staging = Path(tempfile.mkdtemp(prefix=".preparing-" + snapshot_id + "-", dir=parent))
    try:
        note_by_hash = {n["source"]["sha256"]: n for n in notes}
        tasks = [(note["source"], note, games_for_notes[note["game_id"]], cache.get(sha), overrides) for sha, note in sorted(note_by_hash.items())]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for sha, packet in pool.map(extract_source, tasks):
                base.write_json(staging / "sources" / (sha + ".json"), packet)
                raw = staging / "raw" / (sha + ".bin")
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(bodies[sha])
                print(json.dumps({"source": sha, "kind": packet["source_kind"], "units": len(packet["pages"])}), flush=True)
        for row in rows:
            for source in row["sources"]:
                sha = source["sha256"]
                source.update(original_raw_path=source["raw_path"], raw_path="raw/" + sha + ".bin",
                              pages_path="sources/" + sha + ".json", pages_sha256=base.digest((staging / "sources" / (sha + ".json")).read_bytes()))
        labels = [{"game_id": row["game_id"], "home_win": int(games[row["game_id"]]["home_score"] > games[row["game_id"]]["away_score"])} for row in rows]
        base.write_rows(staging / "inputs.jsonl", rows)
        base.write_rows(staging / "labels.jsonl", labels)
        for split in ("train", "dev", "test"):
            membership = {r["game_id"] for r in rows if r["split"] == split}
            base.write_rows(staging / ("labels_" + split + ".jsonl"), [label for label in labels if label["game_id"] in membership])
        base.write_json(staging / "boosting_lineage.json", plans)
        source_counts = Counter(kinds.values())
        summary = {"games": len(rows), "sources": len(kinds), "source_kinds": dict(source_counts),
                   "splits": dict(Counter(r["split"] for r in rows)), "forecast_eligible": dict(Counter(r["split"] for r in rows if r["forecast_eligible"])),
                   "research_only": sum(not r["forecast_eligible"] for r in rows),
                   "test_status": "exploratory_historical_prior_pilot_stats_market_exposure",
                   "html_pagination": "Numbered article sections, not physical PDF pages; source_kind and pagination are explicit.",
                   "source_quality": "Timing/content identity approved; not full factual certification of every source assertion."}
        artifacts = {str(path.relative_to(staging)): base.digest(path.read_bytes()) for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id, "identity": identity, "policy": policy,
                    "original_fit_game_ids": sorted(fit_ids), "artifact_sha256": artifacts, "summary": summary}
        base.write_json(staging / "manifest.json", manifest)
        validate_snapshot(staging)
        for path, expected_sha in provenance.items():
            if base.digest((ROOT / path).read_bytes()) != expected_sha:
                raise ValueError("Input changed during expanded preparation; freeze manifests first: " + path)
        staging.rename(final)
        report = report_parent / snapshot_id
        report.mkdir(exist_ok=False)
        base.write_json(report / "data_summary.json", summary)
        base.write_json(report / "snapshot_manifest.json", manifest)
        print(json.dumps(summary, indent=2))
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--notes-path", action="append", required=True)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--cache-snapshot", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    print(prepare(json.loads(args.config.read_text()), args.notes_path, args.snapshot_id, args.workers, args.cache_snapshot))


if __name__ == "__main__":
    main()
