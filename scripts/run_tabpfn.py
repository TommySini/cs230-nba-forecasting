"""Frozen-feature TabPFN baseline; TRAIN diagnostic first, explicit DEV, no TEST stage.

The default command prepares a 16–32-game TRAIN-only diagnostic. The user explicitly
authorized local compute for this additional baseline. No TEST stage exists.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

import numpy as np

from collect_pilot import ROOT, write_json, write_jsonl
from run_gradient_boosting import (FEATURE_NAMES, bootstrap_brier, new_model,
                                   score_predictions, season_splits, write_csv)

CONFIG_PATH = ROOT / "config/tabpfn.json"
BASELINE_REVISION = "768163157c971beefe5a4c8f5b47ccfefc00e85a"
ADAPTATION_CONFIG_PATH = ROOT / "config/tabpfn_adaptation.json"
STAGES = ("train-smoke", "dev")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return parsed


def repo_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a repository-relative path: {value}")
    return ROOT / path


def verify_hashes(expected, directory=None, *, source_revision=None):
    for name, expected_hash in expected.items():
        if directory is None:
            path = repo_path(name)
        else:
            if Path(name).name != name:
                raise ValueError("Bundle artifact names must be plain filenames")
            path = directory / name
        if sha256(path) != expected_hash:
            # Completed runs bind their original source, which remains in git.
            # This opt-in is only for inspection; model stages use current code.
            historical_source = (directory is None and source_revision == BASELINE_REVISION
                                 and name in {"scripts/run_tabpfn.py", "scripts/validate_tabpfn.py",
                                              "requirements-tabpfn.txt"})
            if historical_source:
                blob = subprocess.check_output(["git", "show", f"{source_revision}:{name}"], cwd=ROOT)
                if hashlib.sha256(blob).hexdigest() == expected_hash:
                    continue
            raise ValueError(f"Changed frozen artifact: {name}")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("compute_platform") != "local":
        raise ValueError("This TabPFN experiment is configured for authorized local compute")
    if not 16 <= config["smoke_games"] <= 32:
        raise ValueError("The TRAIN diagnostic requires 16–32 total games")
    if not 2 <= config["smoke_context_games"] < config["smoke_games"]:
        raise ValueError("TRAIN context must leave later diagnostic query games")
    if config["bootstrap_replicates"] < 1 or config["query_batch_size"] < 1 or config["max_runtime_seconds"] <= 0:
        raise ValueError("Bootstrap repeats, query batch size and runtime bound must be positive")
    output = repo_path(config["output_dir"]).resolve()
    allowed = (ROOT / "data/processed/tabpfn").resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("TabPFN outputs must stay inside data/processed/tabpfn")
    if repo_path(config["report_dir"]).resolve() != (ROOT / "reports/tabpfn").resolve():
        raise ValueError("TabPFN reports must stay in reports/tabpfn")
    model = config["model"]
    if model["version"] != "v2.5" or model["package_version"] != "9.2.0":
        raise ValueError("Use the qualified TabPFN package and explicit model version")
    if len(model["sha256"]) != 64 or len(model["revision"]) != 40:
        raise ValueError("Model checkpoint needs an immutable revision and SHA256")
    if model["device"] not in ("cpu", "mps", "cuda") or model["n_estimators"] < 1:
        raise ValueError("Expected explicit device and positive ensemble size")
    return config


def output_directory(config, stage):
    if stage not in STAGES:
        raise ValueError("Only TRAIN diagnostic and DEV stages exist; TEST is excluded")
    return repo_path(config["output_dir"]) / stage


def read_source_rows(input_path, label_path, allowed_splits):
    """Do not deserialize outcomes for excluded splits in the mixed source files.

    The baseline hashes bind both streams and their positional alignment. All
    selected pairs are still checked by ID; excluded label bytes are only skipped.
    """
    rows, labels = [], {}
    with input_path.open() as input_stream, label_path.open() as label_stream:
        for input_line, label_line in zip_longest(input_stream, label_stream):
            if input_line is None or label_line is None:
                raise ValueError("Source input and label row counts differ")
            row = json.loads(input_line)
            if row["split"] not in allowed_splits:
                continue
            label = json.loads(label_line)
            if row["game_id"] != label["game_id"]:
                raise ValueError("Source inputs and labels are not aligned")
            if row["game_id"] in labels:
                raise ValueError("Duplicate game ID")
            if set(label) != {"game_id", "home_win"} or type(label["home_win"]) is not int or label["home_win"] not in (0, 1):
                raise ValueError("Expected separate binary home-win labels")
            rows.append(row)
            labels[row["game_id"]] = label
    return rows, labels


def select_cohort(inputs, labels_by_id, stage, smoke_games=32, context_games=24):
    """Choose by dates and fixed limits only; never choose using outcomes."""
    if stage not in STAGES:
        raise ValueError("TEST is excluded")
    ordered = sorted(inputs, key=lambda r: (r["cutoff_utc"], r["game_id"]))
    if any(r["split"] not in ({"train"} if stage == "train-smoke" else {"train", "dev"}) for r in ordered):
        raise ValueError("Unexpected split in requested cohort")
    if stage == "train-smoke":
        if not 16 <= smoke_games <= 32 or not 2 <= context_games < smoke_games:
            raise ValueError("Invalid TRAIN diagnostic size")
        if len(ordered) < smoke_games:
            raise ValueError("Insufficient TRAIN games for the diagnostic")
        ordered = ordered[:smoke_games]
        # Keep entire ET days on one side. The limit is fixed before labels are read.
        query_day = ordered[context_games]["game_date_et"]
        fit_rows = [r for r in ordered if r["game_date_et"] < query_day]
        query_rows = [r for r in ordered if r["game_date_et"] >= query_day]
    else:
        fit_rows = [r for r in ordered if r["split"] == "train"]
        query_rows = [r for r in ordered if r["split"] == "dev"]
    if not fit_rows or not query_rows:
        raise ValueError("Both context and query partitions must be nonempty")
    if max(r["game_date_et"] for r in fit_rows) >= min(r["game_date_et"] for r in query_rows):
        raise ValueError("Context and query games must be strictly chronological by ET date")
    combined = fit_rows + query_rows
    ids = [r["game_id"] for r in combined]
    if len(ids) != len(set(ids)) or any(i not in labels_by_id for i in ids):
        raise ValueError("Duplicate games or missing labels")
    return combined, [labels_by_id[i] for i in ids], [r["game_id"] for r in fit_rows], [r["game_id"] for r in query_rows]


def validate_rows(inputs, labels, names, baseline_config):
    if names != list(FEATURE_NAMES) or len(set(names)) != len(names):
        raise ValueError("Feature names/order differ from the frozen statistical baseline")
    if [r["game_id"] for r in inputs] != [r["game_id"] for r in labels]:
        raise ValueError("Prepared input/label alignment changed")
    splits = season_splits(baseline_config)
    for row, label in zip(inputs, labels):
        if row["split"] not in ("train", "dev") or splits.get(row["season"]) != row["split"]:
            raise ValueError("TEST or an incorrect season assignment entered the bundle")
        if set(row["features"]) != set(names):
            raise ValueError("Unexpected feature schema")
        if any(k in row for k in ("home_win", "home_score", "away_score", "outcome")):
            raise ValueError("Outcome field in pregame inputs")
        if timestamp(row["scheduled_start_utc"]) - timestamp(row["cutoff_utc"]) != timedelta(minutes=baseline_config["cutoff_minutes"]):
            raise ValueError("Prediction cutoff differs from the frozen baseline")
        if any(row["game_id"] in row["prior_game_ids"][side] for side in ("home", "away")):
            raise ValueError("Current game appears in its own feature history")
        values = np.asarray([row["features"][name] for name in names], dtype=float)
        if np.isinf(values).any():
            raise ValueError("Infinite pregame feature")
        if set(label) != {"game_id", "home_win"} or type(label["home_win"]) is not int or label["home_win"] not in (0, 1):
            raise ValueError("Expected separate binary labels")


def require_smoke(config):
    directory = output_directory(config, "train-smoke")
    result = json.loads((directory / "run.json").read_text())
    if result.get("stage") != "train-smoke" or result.get("status") != "complete":
        raise ValueError("Finish the TRAIN diagnostic before preparing or running DEV")
    if result["manifest_sha256"] != sha256(directory / "manifest.json"):
        raise ValueError("TRAIN diagnostic manifest changed")
    verify_hashes(result["artifact_sha256"], directory)
    smoke_manifest = json.loads((directory / "manifest.json").read_text())
    if smoke_manifest["config"] != config:
        raise ValueError("Configuration changed since the TRAIN diagnostic")


def prepare_dataset(config, config_path, *, stage="train-smoke"):
    output = output_directory(config, stage)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared output already exists; never overwrite a frozen experiment")
    if stage == "dev":
        require_smoke(config)
    source_path = repo_path(config["source_manifest"])
    source = json.loads(source_path.read_text())
    baseline_path = repo_path(config["baseline_config"])
    baseline = json.loads(baseline_path.read_text())
    if source["config"] != baseline or source["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Frozen baseline manifest/config/features do not match this checkout")
    frozen = source["source_sha256"] | source["artifact_sha256"]
    for required in (config["baseline_config"], config["source_inputs"], config["source_labels"], config["source_predictions"]):
        if required not in frozen:
            raise ValueError(f"Source manifest does not pin {required}")
    verify_hashes(frozen)
    frozen[config["source_manifest"]] = sha256(source_path)
    for path in (Path(__file__).resolve(), ROOT / "scripts/validate_tabpfn.py", ROOT / "requirements-tabpfn.txt", Path(config_path).resolve()):
        frozen[str(path.relative_to(ROOT))] = sha256(path)
    rows, labels = read_source_rows(repo_path(config["source_inputs"]), repo_path(config["source_labels"]),
                                    {"train"} if stage == "train-smoke" else {"train", "dev"})
    inputs, outcomes, fit_ids, query_ids = select_cohort(rows, labels, stage, config["smoke_games"], config["smoke_context_games"])
    validate_rows(inputs, outcomes, source["feature_names"], baseline)
    if len(fit_ids) > config["model"]["max_context_rows"]:
        raise ValueError("All TRAIN rows exceed the pinned model limit; no silent subsampling is allowed")
    if set(labels[i]["home_win"] for i in fit_ids) != {0, 1}:
        raise ValueError("The fixed context needs both outcome classes; do not select using labels")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "inputs.jsonl", inputs)
    write_jsonl(output / "labels.jsonl", outcomes)
    artifacts = {name: sha256(output / name) for name in ("inputs.jsonl", "labels.jsonl")}
    if stage == "dev":
        predictions = {}
        with repo_path(config["source_predictions"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] != "dev":
                    continue
                if row["produced_by"] != "train_fit" or row["game_id"] in predictions:
                    raise ValueError("Expected unique TRAIN-fit frozen DEV predictions")
                predictions[row["game_id"]] = row
        if set(predictions) != set(query_ids):
            raise ValueError("Frozen boosting DEV cohort differs from TabPFN query cohort")
        write_csv(output / "baseline_predictions.csv", [{"game_id": i, "rich_gradient_boosting": predictions[i]["p_boost"]} for i in query_ids])
        artifacts["baseline_predictions.csv"] = sha256(output / "baseline_predictions.csv")
    manifest = {"schema_version": 1, "created_at_utc": utc_now(), "stage": stage, "config": config,
                "implementation_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(config_path), "feature_names": source["feature_names"],
                "baseline_config": baseline, "fit_game_ids": fit_ids, "query_game_ids": query_ids,
                "source_manifest_path": config["source_manifest"], "source_manifest_sha256": sha256(source_path),
                "source_sha256": frozen, "artifact_sha256": artifacts,
                "protocol": "Frozen pregame features; labels separate; context strictly before query ET dates; TEST excluded",
                "baseline_comparison": "same-context fixed first booster candidate" if stage == "train-smoke" else "preserved full-TRAIN DEV-selected booster forecasts",
                "diagnostic_limitation": "Small TRAIN-only check demonstrates functioning inference, not generalization or gradient learning."}
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {stage}: {len(fit_ids)} context, {len(query_ids)} query, {len(source['feature_names'])} features", flush=True)
    return manifest


def load_prepared(config, stage, *, source_revision=None):
    output = output_directory(config, stage)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["config"] != config or manifest["stage"] != stage:
        raise ValueError("Configuration/stage changed after preparation")
    if source_revision is None and manifest["implementation_sha256"] != sha256(Path(__file__)):
        raise ValueError("TabPFN implementation changed after preparation")
    if source_revision is not None:
        verify_hashes({"scripts/run_tabpfn.py": manifest["implementation_sha256"]}, source_revision=source_revision)
    verify_hashes(manifest["source_sha256"], source_revision=source_revision)
    verify_hashes(manifest["artifact_sha256"], output)
    inputs = [json.loads(line) for line in (output / "inputs.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    validate_rows(inputs, labels, manifest["feature_names"], manifest["baseline_config"])
    ids = [r["game_id"] for r in inputs]
    if ids != manifest["fit_game_ids"] + manifest["query_game_ids"] or len(ids) != len(set(ids)):
        raise ValueError("Prepared partitions/membership changed")
    fit_count = len(manifest["fit_game_ids"])
    if not 0 < fit_count < len(inputs) or max(r["game_date_et"] for r in inputs[:fit_count]) >= min(r["game_date_et"] for r in inputs[fit_count:]):
        raise ValueError("Prepared partitions are not strictly chronological")
    if any(r["split"] != "train" for r in inputs[:fit_count]):
        raise ValueError("Only TRAIN outcomes may enter model context")
    if any(r["split"] != ("train" if stage == "train-smoke" else "dev") for r in inputs[fit_count:]):
        raise ValueError("Unexpected query split")
    return manifest, inputs, labels


def positive_probabilities(model, x):
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Expected exactly binary outcome classes 0 and 1")
    matrix = np.asarray(model.predict_proba(x), dtype=float)
    if matrix.shape != (len(x), 2) or not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)) or not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("Invalid class probability matrix")
    return matrix[:, int(np.flatnonzero(classes == 1)[0])]


def model_for(config):
    settings = config["model"]
    if version("tabpfn") != settings["package_version"]:
        raise ValueError("Installed TabPFN version differs from the experiment pin")
    path = repo_path(settings["checkpoint_path"])
    if not path.is_file() or sha256(path) != settings["sha256"]:
        raise ValueError("Install the exact pinned checkpoint before inference; implicit downloads are disabled")
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    model_version = ModelVersion.V2 if settings["version"] == "v2" else ModelVersion.V2_5
    return TabPFNClassifier.create_default_for_version(
        model_version, model_path=path, n_estimators=settings["n_estimators"],
        device=settings["device"], random_state=config["seed"],
        fit_mode="fit_preprocessors", n_preprocessing_jobs=1, memory_saving_mode="auto",
        inference_config={"SUBSAMPLE_SAMPLES": None})


def download_model(config):
    """Fetch only the named immutable public checkpoint; verify before publishing."""
    settings = config["model"]
    path = repo_path(settings["checkpoint_path"])
    if path.exists():
        if sha256(path) != settings["sha256"]:
            raise ValueError("Existing checkpoint hash is wrong; inspect it before replacing it")
        print(f"Verified existing checkpoint: {path}", flush=True)
        return path
    url = f"https://huggingface.co/{settings['repository']}/resolve/{settings['revision']}/{settings['filename']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if sha256(temporary) != settings["sha256"]:
            raise ValueError("Downloaded checkpoint SHA256 differs from the immutable model pin")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified checkpoint: {path}", flush=True)
    return path


def load_progress(path, manifest_sha256, query_rows, versions):
    """Resume complete, atomic batches bound to the exact data/model/software."""
    if not path.exists():
        return {"manifest_sha256": manifest_sha256, "versions": versions,
                "predictions": [], "runtime_seconds": 0.0, "attempts": 0}
    progress = json.loads(path.read_text())
    if progress["manifest_sha256"] != manifest_sha256 or progress["versions"] != versions:
        raise ValueError("Cannot resume with changed data, model configuration or package versions")
    rows = progress["predictions"]
    if len(rows) > len(query_rows):
        raise ValueError("Resume progress contains too many query games")
    for row, expected in zip(rows, query_rows):
        if set(row) != {"game_id", "game_date_et", "tabpfn"} or any(row[key] != expected[key] for key in ("game_id", "game_date_et")):
            raise ValueError("Resume progress is not a prefix of the frozen query cohort")
        value = row["tabpfn"]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid probability in resume progress")
    expected_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if progress.get("predictions_sha256") != expected_hash:
        raise ValueError("Resume prediction checksum changed")
    return progress


def write_report(config, manifest, result):
    report = repo_path(config["report_dir"])
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "metrics.json", result)
    primary = result["primary_comparison"]
    low, high = primary["difference_95_interval"]
    stage = result["stage"]
    interpretation = ("This small TRAIN diagnostic checks that the pipeline works. It does not measure generalization."
                      if stage == "train-smoke" else
                      "The paired DEV interval favors TabPFN." if high < 0 else
                      "The paired DEV interval favors gradient boosting." if low > 0 else
                      "The paired DEV interval includes zero; this run does not establish an advantage.")
    lines = ["# TabPFN statistical baseline", "", f"Stage: **{stage}**. {interpretation}", "",
             f"Both models predict the same {len(manifest['query_game_ids']):,} query games using "
             f"{len(manifest['feature_names'])} frozen pregame features. TabPFN context: "
             f"{len(manifest['fit_game_ids']):,} earlier TRAIN games. TEST is excluded.", "",
             "| Model | Brier | Log loss | Accuracy |", "| --- | ---: | ---: | ---: |"]
    for name, metrics in result["metrics"].items():
        lines.append(f"| {name} | {metrics['brier']:.6f} | {metrics['log_loss']:.6f} | {metrics['accuracy']:.1%} |")
    lines += ["", f"Brier difference (TabPFN minus boosting): {primary['brier_difference']:+.6f}; "
              f"paired ET-date bootstrap 95% interval [{low:+.6f}, {high:+.6f}]. Negative favors TabPFN.", "",
              "TabPFN uses the pinned synthetic-only v2.5 default-2 checkpoint with fixed pretrained weights, "
              f"{config['model']['n_estimators']} ensemble members and no fine-tuning. Model inputs contain "
              "only the baseline feature matrix; outcomes serve as context for earlier TRAIN games and "
              "as separate evaluation targets for query games.", "",
              "Comparator: " + manifest["baseline_comparison"] + ". "
              "DEV is exploratory: the saved booster's settings were selected on DEV. "
              "The bootstrap is conditional on these fitted models and does not include training uncertainty.", "",
              f"Local device: {config['model']['device']}; recorded elapsed time: {result['runtime_seconds']:.1f} s. "
              "The source manifest, checkpoint, packages, cohort and per-game forecasts are pinned in the run artifacts.", ""]
    (report / "findings.md").write_text("\n".join(lines))


def fit_and_predict(config, stage):
    output = output_directory(config, stage)
    if (output / "run.json").exists():
        raise ValueError("Predictions already exist; use the validator instead of rerunning")
    if (output / "predictions.csv").exists() and not (output / "prediction_progress.json").exists():
        raise ValueError("Unbound prediction file exists without resumable progress")
    if stage == "dev":
        require_smoke(config)
    manifest, inputs, labels = load_prepared(config, stage)
    count = len(manifest["fit_game_ids"])
    names = manifest["feature_names"]
    x = np.asarray([[r["features"][n] for n in names] for r in inputs], dtype=float)
    train_y = np.asarray([r["home_win"] for r in labels[:count]], dtype=int)
    if set(train_y.tolist()) != {0, 1} or count > config["model"]["max_context_rows"]:
        raise ValueError("Invalid model context classes or size")
    started = time.monotonic()
    versions = {p: version(p) for p in ("tabpfn", "torch", "numpy", "scikit-learn")}
    progress_path = output / "prediction_progress.json"
    manifest_hash = sha256(output / "manifest.json")
    progress = load_progress(progress_path, manifest_hash, inputs[count:], versions)
    previous_runtime = progress["runtime_seconds"]
    progress["attempts"] += 1
    completed = len(progress["predictions"])
    batch_size = config["query_batch_size"]
    if completed < len(inputs) - count:
        print(f"{stage}: fitting context {count:,} x {len(names)} on {config['model']['device']}; "
              f"resuming at {completed}/{len(inputs) - count} queries", flush=True)
        model = model_for(config)
        model.fit(x[:count], train_y)
        if (model.n_train_samples_ != count or model.inference_config_.SUBSAMPLE_SAMPLES is not None
                or model.ensemble_preprocessor_.subsample_row_indices is not None):
            raise ValueError("TabPFN silently subsampled the fixed TRAIN context")
        progress["context_audit"] = {"submitted_rows": count, "model_n_train_samples": int(model.n_train_samples_),
                                     "subsample_samples": None, "row_subsampling_applied": False}
        progress["predictions_sha256"] = hashlib.sha256(json.dumps(progress["predictions"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        atomic_json(progress_path, progress)
        print(f"Context prepared in {time.monotonic() - started:.1f}s", flush=True)
        for start in range(count + completed, len(x), batch_size):
            if time.monotonic() - started > config["max_runtime_seconds"]:
                raise TimeoutError("Attempt runtime limit reached; completed query batches are resumable")
            print(f"Predicting queries {start - count + 1}–{min(start + batch_size, len(x)) - count} "
                  f"of {len(inputs) - count}; all {count:,} TRAIN rows in context", flush=True)
            p = positive_probabilities(model, x[start:start + batch_size])
            progress["predictions"].extend({"game_id": row["game_id"], "game_date_et": row["game_date_et"], "tabpfn": float(probability)}
                                           for row, probability in zip(inputs[start:start + batch_size], p))
            progress["runtime_seconds"] = previous_runtime + time.monotonic() - started
            progress["predictions_sha256"] = hashlib.sha256(json.dumps(progress["predictions"], sort_keys=True, allow_nan=False).encode()).hexdigest()
            atomic_json(progress_path, progress)
            print(f"Saved {len(progress['predictions'])}/{len(inputs) - count} query forecasts "
                  f"({time.monotonic() - started:.1f}s this attempt); outcomes not scored", flush=True)
    p_tabpfn = np.asarray([row["tabpfn"] for row in progress["predictions"]])
    if stage == "train-smoke":
        baseline = manifest["baseline_config"]
        comparator = new_model(baseline, baseline["candidates"][0]).fit(x[:count], train_y)
        p_boost = positive_probabilities(comparator, x[count:])
    else:
        with (output / "baseline_predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if [r["game_id"] for r in rows] != manifest["query_game_ids"]:
            raise ValueError("Frozen comparator membership changed")
        p_boost = np.asarray([float(r["rich_gradient_boosting"]) for r in rows])
    # Query outcomes become evaluation targets only after both forecasts exist.
    query_y = np.asarray([r["home_win"] for r in labels[count:]], dtype=int)
    probabilities = {"tabpfn": p_tabpfn, "rich_gradient_boosting": p_boost}
    metrics = {name: score_predictions(query_y, p) for name, p in probabilities.items()}
    dates = [r["game_date_et"] for r in inputs[count:]]
    intervals = bootstrap_brier(query_y, probabilities, dates, config["bootstrap_replicates"], config["seed"], "rich_gradient_boosting")
    write_csv(output / "predictions.csv", [{"game_id": row["game_id"], "game_date_et": row["game_date_et"],
               "tabpfn": float(p_tabpfn[i]), "rich_gradient_boosting": float(p_boost[i])}
              for i, row in enumerate(inputs[count:])])
    result = {"schema_version": 1, "stage": stage, "status": "complete", "created_at_utc": utc_now(),
              "execution_platform": config["compute_platform"], "runtime_seconds": previous_runtime + time.monotonic() - started,
              "attempts": progress["attempts"],
              "context_audit": progress["context_audit"],
              "manifest_sha256": sha256(output / "manifest.json"),
              "artifact_sha256": {"predictions.csv": sha256(output / "predictions.csv")},
              "model": config["model"], "model_parameters": {"random_state": config["seed"], "fit_mode": "fit_preprocessors", "n_preprocessing_jobs": 1, "memory_saving_mode": "auto", "inference_config": {"SUBSAMPLE_SAMPLES": None}},
              "versions": versions,
              "metrics": metrics, "uncertainty": intervals,
              "primary_comparison": {"brier_difference": metrics["tabpfn"]["brier"] - metrics["rich_gradient_boosting"]["brier"],
                                     "difference_95_interval": intervals["difference_vs_reference"]["tabpfn"]},
              "interpretation": "TRAIN diagnostic only; not generalization evidence" if stage == "train-smoke" else "Exploratory DEV comparison; baseline settings previously selected on DEV; TEST remains excluded"}
    write_json(output / "run.json", result)
    write_report(config, manifest, result)
    print(json.dumps({"stage": stage, "metrics": metrics, "primary_comparison": result["primary_comparison"]}, indent=2), flush=True)
    return result


def adaptation_config():
    config = json.loads(ADAPTATION_CONFIG_PATH.read_text())
    if (config["baseline_revision"] != BASELINE_REVISION
            or config["baseline_config"] != "config/tabpfn.json"
            or config["output_dir"] != "data/processed/tabpfn/adaptation"
            or config["inner_validation_season"] != 2023):
        raise ValueError("Adaptation must use the fixed baseline and chronological TRAIN split")
    return config


def adaptation_arrays(manifest, inputs, labels):
    count = len(manifest["fit_game_ids"])
    names = manifest["feature_names"]
    x = np.asarray([[row["features"][name] for name in names] for row in inputs], dtype=float)
    y_train = np.asarray([row["home_win"] for row in labels[:count]], dtype=np.int64)
    earlier = np.asarray([row["season"] < 2023 for row in inputs[:count]])
    validation = np.asarray([row["season"] == 2023 for row in inputs[:count]])
    if not earlier.any() or not validation.any() or not np.all(earlier | validation):
        raise ValueError("Missing chronological inner TRAIN partitions")
    return x[:count], y_train, x[count:], earlier, validation


def prepare_adaptation():
    config = adaptation_config()
    output = repo_path(config["output_dir"])
    if (output / "manifest.json").exists():
        raise ValueError("Adaptation protocol already frozen; validate instead of replacing")
    baseline = load_config(repo_path(config["baseline_config"]))
    from validate_tabpfn import validate_run
    validate_run(baseline, "dev")
    manifest, inputs, labels = load_prepared(baseline, "dev", source_revision=BASELINE_REVISION)
    count = len(manifest["fit_game_ids"])
    adaptation_arrays(manifest, inputs, labels)
    paths = ["scripts/run_tabpfn.py", "scripts/validate_tabpfn.py",
             "scripts/tabpfn_synthetic.py", "scripts/tabpfn_finetuning.py",
             "config/tabpfn_adaptation.json", "config/tabpfn.json", "requirements-tabpfn.txt",
             "data/processed/tabpfn/dev/manifest.json", "data/processed/tabpfn/dev/run.json",
             "data/processed/tabpfn/dev/predictions.csv"]
    protocol = {"schema_version": 1, "created_at_utc": utc_now(), "config": config,
                "baseline_revision": BASELINE_REVISION,
                "baseline_manifest_sha256": sha256(output_directory(baseline, "dev") / "manifest.json"),
                "baseline_config": manifest["baseline_config"],
                "fit_game_ids": manifest["fit_game_ids"], "query_game_ids": manifest["query_game_ids"],
                "inner_fit_game_ids": [row["game_id"] for row in inputs[:count] if row["season"] < 2023],
                "inner_validation_game_ids": [row["game_id"] for row in inputs[:count] if row["season"] == 2023],
                "feature_names": manifest["feature_names"],
                "source_sha256": {path: sha256(repo_path(path)) for path in paths},
                "protocol": "Three prespecified exploratory DEV comparisons. Generator and weight updates use only "
                            "2012-22 TRAIN seasons; 2023-24 TRAIN selects checkpoints. Final context uses all "
                            "original TRAIN games. Same synthetic artifact for both augmentation arms. TEST excluded."}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", protocol)
    print(json.dumps({"status": "prepared", "inner_fit": len(protocol["inner_fit_game_ids"]),
                      "inner_validation": len(protocol["inner_validation_game_ids"]),
                      "dev": len(protocol["query_game_ids"])}), flush=True)
    return protocol


def load_adaptation():
    config = adaptation_config()
    output = repo_path(config["output_dir"])
    protocol = json.loads((output / "manifest.json").read_text())
    if protocol["config"] != config or protocol["baseline_revision"] != BASELINE_REVISION:
        raise ValueError("Adaptation protocol changed")
    verify_hashes(protocol["source_sha256"])
    baseline = load_config(repo_path(config["baseline_config"]))
    manifest, inputs, labels = load_prepared(baseline, "dev", source_revision=BASELINE_REVISION)
    if protocol["baseline_manifest_sha256"] != sha256(output_directory(baseline, "dev") / "manifest.json"):
        raise ValueError("Frozen baseline manifest changed")
    count = len(manifest["fit_game_ids"])
    expected = {"fit_game_ids": manifest["fit_game_ids"], "query_game_ids": manifest["query_game_ids"],
                "feature_names": manifest["feature_names"],
                "inner_fit_game_ids": [row["game_id"] for row in inputs[:count] if row["season"] < 2023],
                "inner_validation_game_ids": [row["game_id"] for row in inputs[:count] if row["season"] == 2023]}
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError("Adaptation partition or feature metadata changed")
    return protocol, manifest, inputs, labels


def adaptation_synthetic(config, names):
    output = repo_path(config["output_dir"]) / "synthetic_data"
    manifest = json.loads((output / "manifest.json").read_text())
    verify_hashes(manifest["artifact_sha256"], output)
    x = np.load(output / "features.npy", allow_pickle=False)
    y = np.load(output / "labels.npy", allow_pickle=False)
    if (x.shape != (config["generator"]["n_samples"], len(names)) or y.shape != (len(x),)
            or np.isinf(x).any() or not np.isin(y, [0, 1]).all()):
        raise ValueError("Malformed synthetic augmentation")
    return x, y, manifest


def run_adaptation(stage, variant):
    protocol, manifest, inputs, labels = load_adaptation()
    config = protocol["config"]
    baseline = load_config(repo_path(config["baseline_config"]))
    settings = {**baseline, **config, "model": baseline["model"]}
    root = repo_path(config["output_dir"])
    x_train, y_train, x_dev, earlier, validation = adaptation_arrays(manifest, inputs, labels)
    if stage == "adapt-generate":
        from tabpfn_synthetic import generate
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(config["generator"]["max_runtime_seconds"] + 120)
        try:
            return generate(settings, x_train[earlier], y_train[earlier], manifest["feature_names"],
                            protocol["inner_fit_game_ids"], root / "synthetic_data")
        finally:
            signal.alarm(0)
    if variant not in ("synthetic", "finetune", "combined"):
        raise ValueError("Choose --variant synthetic, finetune or combined")
    output = root / variant
    output.mkdir(parents=True, exist_ok=True)
    x_fake, y_fake, synthetic_manifest = None, None, None
    if variant != "finetune":
        x_fake, y_fake, synthetic_manifest = adaptation_synthetic(config, manifest["feature_names"])
        if (synthetic_manifest["input_partition_game_ids"] != protocol["inner_fit_game_ids"]
                or not set(synthetic_manifest["source_game_ids"]).issubset(protocol["inner_fit_game_ids"])):
            raise ValueError("Synthetic generation used forbidden source games")
    if stage == "adapt-fit":
        if variant == "synthetic":
            raise ValueError("Synthetic-only has no gradient training")
        if (output / "training.json").exists():
            raise ValueError("Completed fine-tuning is immutable")
        from tabpfn_finetuning import train
        x_fit, y_fit = x_train[earlier], y_train[earlier]
        if x_fake is not None:
            x_fit, y_fit = np.concatenate([x_fit, x_fake]), np.concatenate([y_fit, y_fake])
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(config["finetuning"]["max_runtime_seconds"] + 120)
        try:
            training = train(settings, x_fit, y_fit, x_train[validation], y_train[validation], output / "training")
        finally:
            signal.alarm(0)
        training.update({"fit_game_ids": protocol["inner_fit_game_ids"],
                         "validation_game_ids": protocol["inner_validation_game_ids"],
                         "synthetic_rows": 0 if x_fake is None else len(x_fake),
                         "manifest_sha256": sha256(root / "manifest.json")})
        write_json(output / "training.json", training)
        print(json.dumps({key: training[key] for key in ("runtime_seconds", "optimizer_steps",
                          "epochs_completed", "initial_inner_log_loss", "best_inner_log_loss",
                          "selected_pretrained_weights")}), flush=True)
        return training
    if stage != "adapt-dev":
        raise ValueError("Unknown adaptation stage")
    if (output / "run.json").exists():
        raise ValueError("Completed DEV predictions are immutable; use the validator")
    training = None
    if variant != "synthetic":
        training = json.loads((output / "training.json").read_text())
        if (training["manifest_sha256"] != sha256(root / "manifest.json")
                or training["checkpoint_sha256"] != sha256(repo_path(training["checkpoint_path"]))):
            raise ValueError("Training protocol or selected checkpoint changed")
    context_x, context_y = x_train, y_train
    if x_fake is not None:
        context_x, context_y = np.concatenate([x_train, x_fake]), np.concatenate([y_train, y_fake])
    prediction_file = output / "prediction_progress.json"
    started = time.monotonic()
    binding = {"manifest_sha256": sha256(root / "manifest.json"), "variant": variant,
               "checkpoint_sha256": training["checkpoint_sha256"] if training else baseline["model"]["sha256"],
               "synthetic_manifest_sha256": sha256(root / "synthetic_data/manifest.json") if x_fake is not None else None}
    if prediction_file.exists():
        progress = json.loads(prediction_file.read_text())
        if progress["binding"] != binding:
            raise ValueError("Saved prediction provenance changed")
        p = np.asarray(progress["probabilities"])
        expected_hash = hashlib.sha256(json.dumps(progress["probabilities"], allow_nan=False).encode()).hexdigest()
        if (progress.get("probabilities_sha256") != expected_hash or p.shape != (len(x_dev),)
                or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
            raise ValueError("Saved prediction probabilities changed or are malformed")
        audit = progress["context_audit"]
        runtime = progress["runtime_seconds"]
    else:
        if training:
            from tabpfn_finetuning import classifier_from_checkpoint
            model = classifier_from_checkpoint(settings, repo_path(training["checkpoint_path"]))
        else:
            model = model_for(baseline)
        print(f"{variant}: context {len(context_x)} x {context_x.shape[1]}; {len(x_dev)} DEV queries", flush=True)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(1800)
        try:
            model.fit(context_x, context_y)
            if (model.n_train_samples_ != len(context_x) or model.inference_config_.SUBSAMPLE_SAMPLES is not None
                    or model.ensemble_preprocessor_.subsample_row_indices is not None):
                raise ValueError("Adaptation silently subsampled context")
            p = positive_probabilities(model, x_dev)
        finally:
            signal.alarm(0)
        runtime = time.monotonic() - started
        audit = {"original_train_rows": len(x_train), "synthetic_rows": 0 if x_fake is None else len(x_fake),
                 "submitted_rows": len(context_x), "model_n_train_samples": int(model.n_train_samples_),
                 "row_subsampling_applied": False}
        atomic_json(prediction_file, {"binding": binding, "probabilities": p.tolist(),
                                     "probabilities_sha256": hashlib.sha256(json.dumps(p.tolist(), allow_nan=False).encode()).hexdigest(),
                                     "context_audit": audit, "runtime_seconds": runtime})
        del model
    # Evaluation outcomes are accessed only after forecasts have been persisted.
    y_dev = np.asarray([row["home_win"] for row in labels[len(x_train):]], dtype=int)
    with (root.parent / "dev/predictions.csv").open() as stream:
        original = list(csv.DictReader(stream))
    if [row["game_id"] for row in original] != manifest["query_game_ids"]:
        raise ValueError("Frozen comparator query membership changed")
    probabilities = {variant: p, **{name: np.asarray([float(row[name]) for row in original])
                                  for name in ("tabpfn", "rich_gradient_boosting")}}
    metrics = {name: score_predictions(y_dev, values) for name, values in probabilities.items()}
    uncertainty = bootstrap_brier(y_dev, probabilities, [row["game_date_et"] for row in original],
                                 config["bootstrap_replicates"], config["seed"], "tabpfn")
    write_csv(output / "predictions.csv", [{"game_id": row["game_id"], "game_date_et": row["game_date_et"],
              **{name: float(values[index]) for name, values in probabilities.items()}} for index, row in enumerate(original)])
    result = {"schema_version": 1, "stage": stage, "status": "complete", "variant": variant,
              "created_at_utc": utc_now(), "execution_platform": "local", "runtime_seconds": runtime,
              "manifest_sha256": binding["manifest_sha256"],
              "artifact_sha256": {"predictions.csv": sha256(output / "predictions.csv")},
              "context_audit": audit, "training": training,
              "synthetic_manifest_sha256": binding["synthetic_manifest_sha256"],
              "model": baseline["model"], "versions": {p: version(p) for p in ("tabpfn", "tabpfn-extensions", "torch", "numpy", "scikit-learn")},
              "metrics": metrics, "uncertainty": uncertainty,
              "primary_comparison": {"brier_difference": metrics[variant]["brier"] - metrics["tabpfn"]["brier"],
                                     "difference_95_interval": uncertainty["difference_vs_reference"][variant]}}
    write_json(output / "run.json", result)
    print(json.dumps({"variant": variant, "metrics": metrics, "primary_comparison": result["primary_comparison"]}, indent=2), flush=True)
    return result


def report_adaptation():
    config = adaptation_config()
    root = repo_path(config["output_dir"])
    baseline = json.loads((root.parent / "dev/run.json").read_text())
    variants = {}
    from validate_tabpfn import validate_adaptation
    for variant in ("synthetic", "finetune", "combined"):
        if (root / variant / "run.json").exists():
            validate_adaptation(variant)
            variants[variant] = json.loads((root / variant / "run.json").read_text())
    report = ROOT / "reports/tabpfn"
    write_json(report / "metrics.json", {"baseline": baseline, "adaptation": variants})
    lines = ["# TabPFN statistical experiments", "",
             "All comparisons use the same 1,123 DEV games (2024–25) and 252 frozen pregame features. "
             "TRAIN is 2012–24; TEST is excluded. Lower Brier/log loss is better.", "",
             "| Model | Brier | Log loss | AUC | Accuracy | Brier difference vs frozen TabPFN (95% interval) |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name in ("rich_gradient_boosting", "tabpfn"):
        m = baseline["metrics"][name]
        lines.append(f"| {name} | {m['brier']:.6f} | {m['log_loss']:.6f} | {m['roc_auc']:.4f} | {m['accuracy']:.1%} | — |")
    for name, run in variants.items():
        m, comparison = run["metrics"][name], run["primary_comparison"]
        low, high = comparison["difference_95_interval"]
        lines.append(f"| {name} | {m['brier']:.6f} | {m['log_loss']:.6f} | {m['roc_auc']:.4f} | {m['accuracy']:.1%} | "
                     f"{comparison['brier_difference']:+.6f} [{low:+.6f}, {high:+.6f}] |")
    lines += ["", "## Protocol", "",
              "The original TabPFN-2.5 result used fixed pretrained weights and all 13,457 TRAIN games as context, "
              "with four estimators on Apple MPS. Its Brier difference versus boosting was -0.002500 "
              "(paired 95% interval [-0.005122, +0.000148]); this did not establish an advantage.", "",
              "The adaptation study uses the official TabPFN unsupervised generation and fine-tuning tools. "
              "Synthetic-only and combined share 1,024 generated rows, using a deterministic sample of "
              "1,024 real 2012–22 TRAIN games as the generator context. The continuous regressor checkpoint "
              "could not be downloaded (HuggingFace TLS timeouts; official fallback HTTP404), so the official "
              "classifier-only joint sampler uses up to eight TRAIN-fitted quantile bins per primitive feature. "
              "Bin medians are decoded and engineered arithmetic/missingness identities reconstructed. This is "
              "a coarse binned generation experiment, not an evaluation of the continuous-regressor generator. "
              "Fine-tuning uses the earlier TRAIN seasons for weight updates and "
              "the 2023–24 TRAIN season for checkpoint selection by log loss. Final prediction context includes "
              "all original TRAIN games, plus the same synthetic rows in augmentation arms. "
              "Both fine-tuning arms use the same fixed training settings and budget.", "",
              "Fine-tuning is bounded to five epochs, learning rate 1e-5, 256-row meta-batches, one estimator "
              "during updates/inner validation, patience two, and four estimators for final prediction. "
              "The selected checkpoint may remain the original pretrained weights if no epoch improves "
              "inner-TRAIN log loss; per-run metadata records that explicitly.", "",
              "Official documentation: [data generation](https://docs.priorlabs.ai/capabilities/data-generation), "
              "[fine-tuning](https://docs.priorlabs.ai/capabilities/fine-tuning).", "",
              "Intervals use 2,000 paired ET-calendar-day bootstrap replicates. These are exploratory, "
              "unadjusted comparisons of three variants, conditional on the fitted models; they do not include "
              "training/generation seed uncertainty or dependence across days. DEV had already been inspected "
              "and used to select the booster. Synthetic rows are model-generated examples, not new observed games.", "",
              "Frozen baseline artifacts remain in `data/processed/tabpfn/dev/`; experiment manifests, "
              "separate synthetic features/labels, training checkpoints and per-game predictions are under "
              "`data/processed/tabpfn/adaptation/`. Recompute scores with `validate_tabpfn.py --variant <name>`.", ""]
    (report / "findings.md").write_text("\n".join(lines))
    print(json.dumps({name: run["primary_comparison"] for name, run in variants.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", choices=("download-model", "prepare", *STAGES,
                                           "adapt-prepare", "adapt-generate", "adapt-fit", "adapt-dev", "adapt-report"), default="prepare")
    parser.add_argument("--variant", choices=("synthetic", "finetune", "combined"))
    parser.add_argument("--cohort", choices=STAGES, default="train-smoke", help="Cohort for --stage prepare")
    args = parser.parse_args()
    if args.stage.startswith("adapt-"):
        if args.stage == "adapt-prepare":
            prepare_adaptation()
        elif args.stage == "adapt-report":
            report_adaptation()
        else:
            run_adaptation(args.stage, args.variant)
        return
    config = load_config(args.config)
    if args.stage == "download-model":
        download_model(config)
    elif args.stage == "prepare":
        prepare_dataset(config, args.config, stage=args.cohort)
    else:
        # Default OS signal action enforces the wall limit even during a long
        # native/MPS operation; a Python signal handler could wait for that call.
        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("The bounded local CLI requires POSIX SIGALRM support")
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(math.ceil(config["max_runtime_seconds"]))
        print(f"Attempt wall-clock cutoff: {config['max_runtime_seconds']}s; complete batches survive a timeout", flush=True)
        try:
            fit_and_predict(config, args.stage)
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()
