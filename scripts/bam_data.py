"""Immutable, outcome-separated BAM inputs. Does not modify prior experiments.

Each prepare invocation freezes the complete approved-note manifests. A later
collector update requires a new snapshot. Raw byte hashes identify sources;
page/line citations are stable within the snapshot, not inferred by a model.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "bam-inputs-v1"
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


def verified_bytes(note, root=ROOT):
    from audit_game_notes import archive_time_verified
    if note.get("review", {}).get("decision") != "approved_template_checked_excerpt":
        raise ValueError("Unapproved source packet")
    source = note["source"]
    if not archive_time_verified(source, note["publisher_url"], note["cutoff_utc"]):
        raise ValueError("Source lacks verified archived bytes before prediction cutoff")
    path = (Path(root) / source["raw_path"]).resolve()
    if not path.is_relative_to((Path(root) / "data/raw").resolve()):
        raise ValueError("Raw source path escapes data/raw")
    body = path.read_bytes()
    if digest(body) != source["sha256"]:
        raise ValueError("Raw source checksum mismatch")
    return body


def merge_notes(notes):
    """Merge packets by game and content, retaining aliases and version warnings.

    Distinct historical versions are retained for audit. Only the latest verified
    pre-cutoff version of each publisher URL is active for research, preventing
    duplicate documents from receiving more weight. Different URLs stay active.
    """
    games = {}
    for note in sorted(notes, key=lambda n: canonical(n)):
        meta = {key: note[key] for key in METADATA}
        game = games.setdefault(note["game_id"], {**meta, "sources": []})
        if any(game[key] != meta[key] for key in METADATA):
            raise ValueError("Conflicting game metadata across note packets")
        source = note["source"]
        archive_time = parsedate_to_datetime(source["memento_datetime"]).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        entry = {"source_id": "sha256:" + source["sha256"], "sha256": source["sha256"],
                 "publisher": note.get("team", note.get("publisher", "unknown")),
                 "url": note["publisher_url"], "raw_path": source["raw_path"],
                 "archive_time": archive_time, "archive_url": source["final_url"],
                 "archive_provenance": source, "approval_scope": note["review"], "aliases": []}
        existing = next((s for s in game["sources"] if s["sha256"] == entry["sha256"]), None)
        if existing:
            alias = {k: entry[k] for k in ("publisher", "url", "raw_path", "archive_time", "archive_url", "archive_provenance", "approval_scope")}
            if alias not in existing["aliases"]:
                existing["aliases"].append(alias)
        else:
            game["sources"].append(entry)
    for game in games.values():
        capture_keys = ("publisher", "url", "raw_path", "archive_time", "archive_url", "archive_provenance", "approval_scope")
        groups = defaultdict(list)
        for source in game["sources"]:
            # Identical bytes can recur after an intervening changed version.
            # Keep every capture, then choose the newest capture as the primary
            # provenance record. Hash dedup must not discard version chronology.
            captures = [{k: source[k] for k in capture_keys}, *source["aliases"]]
            unique = {canonical(capture): capture for capture in captures}
            captures = sorted(unique.values(), key=lambda c: (c["archive_time"], canonical(c)))
            source.update(captures[-1])
            source["aliases"] = sorted(captures[:-1], key=canonical)
            for capture in captures:
                groups[capture["url"]].append((source["sha256"], capture))
        group_details = {}
        for url, captures in groups.items():
            by_time = defaultdict(set)
            for sha, capture in captures:
                by_time[capture["archive_time"]].add(sha)
            if any(len(hashes) > 1 for hashes in by_time.values()):
                raise ValueError("Conflicting byte versions for the same archive capture")
            latest = max(by_time)
            group_details[url] = {"url": url, "source_version_count": len({sha for sha, _ in captures}),
                                  "selected_sha256": next(iter(by_time[latest])), "latest_archive_time": latest}
        for source in game["sources"]:
            urls = {source["url"], *(alias["url"] for alias in source["aliases"])}
            source["source_version_groups"] = [group_details[url] for url in sorted(urls)]
            source["active_for_research"] = any(group_details[url]["selected_sha256"] == source["sha256"] for url in urls)
            source["source_version_count"] = max(group_details[url]["source_version_count"] for url in urls)
        game["sources"].sort(key=lambda s: (s["publisher"], s["url"], s["archive_time"], s["sha256"]))
    return sorted(games.values(), key=lambda r: (r["game_date_et"], r["game_id"]))


def assign_splits(rows, fit_ids, policy):
    """Keep whole ET dates together; pre-backbone and booster-fit games train only."""
    rows = sorted(rows, key=lambda r: (r["game_date_et"], r["game_id"]))
    cutoff = policy["prediction_model_cutoff_date"]
    train_only_dates = {r["game_date_et"] for r in rows
                        if r["game_date_et"] <= cutoff or r["game_id"] in fit_ids or not r["forecast_eligible"]}
    last_train_only = max(train_only_dates, default="")
    minimum = policy.get("minimum_split_games", {"train": 30, "dev": 8, "test": 8})
    explicit = policy.get("dev_start_date"), policy.get("test_start_date")
    if any(explicit):
        a_date, b_date = explicit
        if not a_date or not b_date or a_date >= b_date or a_date <= max(last_train_only, cutoff):
            raise ValueError("Invalid explicit chronological split boundaries")
        result = [dict(r, split="train" if r["game_date_et"] < a_date else "dev" if r["game_date_et"] < b_date else "test") for r in rows]
    else:
        n = len(rows)
        target = policy.get("requested_fractions", [0.75, 0.125, 0.125])
        boundaries = [i for i in range(1, n) if rows[i - 1]["game_date_et"] != rows[i]["game_date_et"]]
        choices = []
        for a in boundaries:
            if rows[a]["game_date_et"] <= max(last_train_only, cutoff):
                continue
            for b in boundaries:
                counts = a, b - a, n - b
                if b <= a or any(c < minimum[k] for c, k in zip(counts, ("train", "dev", "test"))):
                    continue
                choices.append((sum((c / n - f) ** 2 for c, f in zip(counts, target)), a, b))
        if not choices:
            raise ValueError("Insufficient chronological unseen games for dev/test; do not silently reuse training games")
        _, a, b = min(choices)
        result = [dict(r, split="train" if i < a else "dev" if i < b else "test") for i, r in enumerate(rows)]
    if any(sum(r["split"] == split for r in result) < minimum[split] for split in minimum):
        raise ValueError("Too few games for requested split minima")
    return result


def prior_probability_plan(targets, numeric, fit_ids, minimum=200):
    """Pure earlier-only membership; no target outcome is consulted here."""
    by_id = {r["game_id"]: r for r in numeric}
    if not fit_ids or not fit_ids <= by_id.keys():
        raise ValueError("Original booster fitting membership is missing from numeric history")
    last_original = max(by_id[g]["game_date_et"] for g in fit_ids)
    plans, assignments = {}, {}
    for target in targets:
        gid = target["game_id"]
        if gid not in by_id:
            assignments[gid] = {"fold": None, "reason": "baseline_warmup"}
            continue
        month = target["game_date_et"][:7] + "-01"
        original = target["game_date_et"] > last_original
        fold = "original_frozen" if original else month
        prior = sorted(fit_ids) if original else sorted(r["game_id"] for r in numeric if r["game_date_et"] < month)
        if len(prior) < minimum:
            assignments[gid] = {"fold": None, "reason": "insufficient_earlier_boosting_history"}
            continue
        if gid in prior or max(by_id[g]["game_date_et"] for g in prior) >= target["game_date_et"]:
            raise ValueError("Target or future game entered booster fitting")
        plans[fold] = {"fit_game_ids": prior, "last_fit_date": max(by_id[g]["game_date_et"] for g in prior),
                       "fit_before_date": None if original else month, "policy": "frozen_original" if original else "earlier_calendar_months_only"}
        assignments[gid] = {"fold": fold, "reason": None}
    return plans, assignments


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


def extract_pages(task):
    """No mutation of the existing preparer's ROOT/OUTPUT/global paths."""
    raw_path, raw_hash, cached, overrides = task
    from audit_game_notes import decode_archive_body
    from prepare_full_game_notes import select_page_text
    from pypdf import PdfReader
    body = Path(raw_path).read_bytes()
    if digest(body) != raw_hash:
        raise ValueError("Source changed during extraction")
    if cached:
        cache_path, cache_sha = cached
        content = Path(cache_path).read_bytes()
        if digest(content) != cache_sha:
            raise ValueError("Extraction cache checksum changed")
        packet = json.loads(content)
        if packet["source"]["sha256"] != raw_hash:
            raise ValueError("Extraction cache belongs to another source")
        from prepare_full_game_notes import validate_page_coverage
        validate_page_coverage(packet["pages"], packet["chunks"], packet["pdf_pages"])
        return raw_hash, pages_with_lines(packet["pages"])
    pages = []
    for number, page in enumerate(PdfReader(io.BytesIO(decode_archive_body(body))).pages, 1):
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
        override = overrides.get(raw_hash + ":" + str(number))
        text, method, selected_flags = select_page_text(plain, layout, override)
        pages.append({"page": number, "text": text, "text_sha256": digest(text.encode()),
                      "plain_text": plain, "layout_text": layout, "method": method, "flags": flags + selected_flags})
        if override:
            pages[-1]["visual_transcription"] = {k: v for k, v in override.items() if k != "text"}
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
    fit_ids = set(manifest["original_fit_game_ids"])
    for row in rows:
        date_splits[row["game_date_et"]].add(row["split"])
        if any(k in row for k in ("home_win", "home_score", "away_score", "recap", "postmortem")):
            raise ValueError("Outcome content in predictor input")
        expected = stamp(row["scheduled_start_utc"]) - timedelta(minutes=60)
        if stamp(row["cutoff_utc"]) != expected:
            raise ValueError("Prediction cutoff differs from tipoff minus 60 minutes")
        if row["split"] != "train" and (row["game_date_et"] <= manifest["policy"]["prediction_model_cutoff_date"] or row["game_id"] in fit_ids):
            raise ValueError("Training-only game in evaluation")
        p = row["p_boost"]
        if row["forecast_eligible"] != (p is not None) or p is not None and (not math.isfinite(p) or not 0 < p < 1):
            raise ValueError("Invalid forecast eligibility/probability")
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
    from run_gradient_boosting import (FEATURE_NAMES, arrays, load_verified_history,
                                      new_model, prepare_examples)
    root = Path(root)
    if root.resolve() != ROOT.resolve():
        raise ValueError("Feature verifier uses this repository root; alternate roots are unsupported")
    base_path = root / config.get("baseline_config", "config/gradient_boosting.json")
    baseline = json.loads(base_path.read_text())
    if baseline["cutoff_minutes"] != 60:
        raise ValueError("BAM requires scheduled tipoff minus 60 minutes")
    history = load_verified_history(baseline)
    numeric, labels, _ = prepare_examples(history, baseline)
    by_id = {r["game_id"]: r for r in numeric}
    history_by_id = {r["game_id"]: r for r in history}
    labels_by_id = {r["game_id"]: r for r in labels}
    original_path = root / "data/processed/gradient_boosting/model.joblib"
    original = joblib.load(original_path)
    fit_ids = set(original["fit_game_ids"]) | set(original.get("selection_game_ids", []))
    provenance = {}
    def freeze(path):
        path = Path(path)
        provenance[str(path.relative_to(root))] = digest(path.read_bytes())
    for path in (base_path, original_path, root / baseline["history_path"], root / baseline["history_config"], Path(__file__),
                 root / "scripts/run_gradient_boosting.py", root / "scripts/collect_pilot.py",
                 root / "scripts/audit_game_notes.py", root / "scripts/prepare_full_game_notes.py"):
        freeze(path)
    notes = []
    for name in config.get("notes_paths", ["data/processed/game_notes_collection/approved_excerpts.jsonl"]):
        path = root / name
        content = path.read_bytes()
        provenance[name] = digest(content)
        notes.extend(json.loads(line) for line in content.splitlines() if line.strip())
    bodies = {}
    for note in notes:
        game = history_by_id.get(note["game_id"])
        if game is None:
            raise ValueError("Note game missing verified schedule; extend numeric history separately")
        for key in ("game_date_et", "home_abbr", "away_abbr"):
            if note[key] != game[key]:
                raise ValueError("Note differs from verified schedule: " + key)
        if stamp(note["cutoff_utc"]) != stamp(game["scheduled_start_utc"]) - timedelta(minutes=60):
            raise ValueError("Note cutoff differs from scheduled tipoff minus 60 minutes")
        bodies[note["source"]["sha256"]] = verified_bytes(note, root)
        freeze(root / note["source"]["raw_path"])
    rows = merge_notes(notes)
    minimum = config.get("minimum_oof_fit_games", 200)
    # fit_game_ids describes actual final fit; selection IDs are additionally
    # forbidden in hybrid dev/test but are not silently added to model fitting.
    plans, assignments = prior_probability_plan(rows, numeric, set(original["fit_game_ids"]), minimum)
    booster_parameters = config.get("boosting_parameters", {"n_estimators": 100, "max_depth": 1})
    expected_parameters = original["model"].get_params()
    model_parameters = {**baseline["model_parameters"], **booster_parameters, "random_state": baseline["seed"]}
    if any(expected_parameters[k] != v for k, v in model_parameters.items()):
        raise ValueError("OOF booster parameters differ from original frozen model")
    for fold, plan in plans.items():
        selected = [by_id[g] for g in plan["fit_game_ids"]]
        x, y = arrays(selected, [labels_by_id[r["game_id"]] for r in selected])
        model = original["model"] if fold == "original_frozen" else new_model(baseline, booster_parameters).fit(x, y)
        targets = [row for row in rows if assignments[row["game_id"]]["fold"] == fold]
        for row in targets:
            row["p_boost"] = float(model.predict_proba([[by_id[row["game_id"]]["features"][n] for n in FEATURE_NAMES]])[0, 1])
    for row in rows:
        game = history_by_id[row["game_id"]]
        row.update(scheduled_start_utc=game["scheduled_start_utc"], season=game["season"],
                   p_boost=row.get("p_boost"), forecast_eligible=row.get("p_boost") is not None,
                   boosting_fold=assignments[row["game_id"]]["fold"],
                   forecast_ineligibility_reason=assignments[row["game_id"]]["reason"])
        if row["game_id"] in by_id:
            row.update(features=by_id[row["game_id"]]["features"], prior_game_ids=by_id[row["game_id"]]["prior_game_ids"])
    policy = config.get("split_policy", config)
    rows = assign_splits(rows, fit_ids, policy)
    inspected = set()
    for name in config.get("prior_inspected_inputs", ["data/processed/boosting_notes/inputs.jsonl", "data/processed/text_ablation/inputs.jsonl"]):
        path = root / name
        if path.exists():
            freeze(path)
            inspected.update(r["game_id"] for r in read_rows(path) if r["split"] in ("dev", "test", "validation"))
    cache = {}
    manifest_path = root / "data/processed/full_game_notes/manifest.json"
    if manifest_path.exists():
        freeze(manifest_path)
        for item in json.loads(manifest_path.read_text())["packets"]:
            cache[item["source"]["sha256"]] = (str(root / item["directory"] / "packet.json"), item["packet_sha256"])
    overrides_path = root / "config/full_game_notes_extraction_overrides.json"
    overrides = {}
    if overrides_path.exists():
        freeze(overrides_path)
        overrides = json.loads(overrides_path.read_text())["pages"]
        for value in overrides.values():
            image_path = root / value["image_path"]
            freeze(image_path)
            if digest(image_path.read_bytes()) != value["image_sha256"]:
                raise ValueError("Visual transcription reference changed")
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("visual_override_sha256") != (digest(overrides_path.read_bytes()) if overrides_path.exists() else None):
        cache = {}  # Prior extraction used different reviewed transcriptions.
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
        tasks = {}
        for row in rows:
            for source in row["sources"]:
                sha = source["sha256"]
                original_raw_path = source["raw_path"]
                tasks[sha] = (str(root / original_raw_path), sha, cache.get(sha), overrides)
                source.update(original_raw_path=original_raw_path, raw_path="raw/" + sha + ".bin",
                              pages_path="sources/" + sha + ".json")
                raw = staging / source["raw_path"]
                if not raw.exists():
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_bytes(bodies[sha])
        print(json.dumps({"snapshot_id": snapshot_id, "games": len(rows), "unique_sources": len(tasks),
                          "cached_sources": sum(t[2] is not None for t in tasks.values()), "stage": "extracting"}), flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for sha, pages in executor.map(extract_pages, [tasks[k] for k in sorted(tasks)]):
                write_json(staging / "sources" / (sha + ".json"), {"source_sha256": sha, "pages": pages,
                           "visual_completeness": "Text extraction preserves all PDF pages; figures and extraction flags require separate review."})
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
        write_json(staging / "boosting_lineage.json", plans)
        reused = sorted(r["game_id"] for r in rows if r["split"] == "test" and r["game_id"] in inspected)
        summary = {"games": len(rows), "sources": len(tasks), "splits": dict(Counter(r["split"] for r in rows)),
                   "forecast_eligible": dict(Counter(r["split"] for r in rows if r["forecast_eligible"])),
                   "research_only": sum(not r["forecast_eligible"] for r in rows),
                   "previously_inspected_test_ids": reused,
                   "test_status": "exploratory_previously_inspected" if reused else "candidate_uninspected_requires_model_knowledge_audit",
                   "full_packet_quality": "All archived pages preserved; first-page approval is not a full-packet semantic review."}
        artifacts = {str(p.relative_to(staging)): digest(p.read_bytes()) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id, "identity": identity,
                    "policy": policy, "original_fit_game_ids": sorted(fit_ids), "artifact_sha256": artifacts, "summary": summary}
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
