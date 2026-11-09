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
import json
import math
import os
from pathlib import Path
import signal
import time
import urllib.request

import numpy as np

from collect_pilot import ROOT, write_json
from run_gradient_boosting import (FEATURE_NAMES, bootstrap_brier, new_model,
                                   score_predictions, season_splits)

CONFIG_PATH = ROOT / "config/tabpfn.json"
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


def verify_hashes(expected, directory=None):
    for name, expected_hash in expected.items():
        if directory is None:
            path = repo_path(name)
        else:
            if Path(name).name != name:
                raise ValueError("Bundle artifact names must be plain filenames")
            path = directory / name
        if sha256(path) != expected_hash:
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
    allowed = (ROOT / "reports/tabpfn").resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("TabPFN run records must stay inside reports/tabpfn")
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


def shared_dataset(config):
    from nba_dataset import Dataset
    return Dataset(ROOT, ROOT / config.get("dataset_dir", "data/processed"))


def read_source_rows(dataset, allowed_splits):
    """Open only explicitly allowed label partitions; TEST has no TabPFN stage."""
    if not allowed_splits or not set(allowed_splits) <= {"train", "dev"}:
        raise ValueError("TEST is excluded")
    rows = dataset.inputs(splits=allowed_splits)
    outcomes = dataset.labels(allowed_splits)
    labels = {r["game_id"]: r for r in outcomes}
    ids = [r["game_id"] for r in rows]
    if len(ids) != len(set(ids)) or len(labels) != len(outcomes) or set(ids) != set(labels):
        raise ValueError("Shared inputs and labels must align uniquely")
    return rows, labels


def record_directory(config, stage):
    """Read staged metadata during validation, installed reports otherwise."""
    output = output_directory(config, stage)
    staging = shared_dataset(config).processed_dir / "exports" / config["output_dir"] / stage
    return staging if staging.exists() else output


def prediction_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def load_comparison_rows(config, stage, manifest):
    dataset = shared_dataset(config)
    ids = manifest["query_game_ids"]
    forecasts = {}
    for model in ("tabpfn", "tabpfn_comparator"):
        rows = dataset.predictions(model, stage=stage)
        by_id = {r["game_id"]: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != set(ids):
            raise ValueError("Saved prediction membership differs from query cohort")
        forecasts[model] = by_id
    result = []
    for gid in ids:
        left, right = forecasts["tabpfn"][gid], forecasts["tabpfn_comparator"][gid]
        if left["game_date_et"] != right["game_date_et"]:
            raise ValueError("Comparator prediction dates differ")
        result.append({"game_id": gid, "game_date_et": left["game_date_et"],
                       "tabpfn": left["p_home_win"], "rich_gradient_boosting": right["p_home_win"]})
    return result


def verify_imported_record(dataset, stage, name, value):
    key = f"data/processed/tabpfn/{stage}/{name}"
    if dataset.manifest["legacy_records"].get(key) != value:
        raise ValueError("Changed imported run record: " + key)


def experiment_settings(config):
    paths = {"source_manifest", "source_inputs", "source_labels", "source_predictions", "baseline_config",
             "dataset_dir", "output_dir", "report_dir"}
    return {k: v for k, v in config.items() if k not in paths}

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
    from validate_tabpfn import validate_run
    validate_run(config, "train-smoke")

def prepare_dataset(config, config_path, *, stage="train-smoke"):
    output = output_directory(config, stage)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared output already exists; never overwrite a completed experiment")
    if stage == "dev":
        require_smoke(config)
    dataset = shared_dataset(config)
    dataset.verify(label_splits={"train"} if stage == "train-smoke" else {"train", "dev"})
    baseline = json.loads(repo_path(config["baseline_config"]).read_text())
    names = dataset.manifest["feature_names"]
    rows, labels = read_source_rows(dataset, {"train"} if stage == "train-smoke" else {"train", "dev"})
    inputs, outcomes, fit_ids, query_ids = select_cohort(rows, labels, stage, config["smoke_games"], config["smoke_context_games"])
    validate_rows(inputs, outcomes, names, baseline)
    if len(fit_ids) > config["model"]["max_context_rows"] or set(labels[i]["home_win"] for i in fit_ids) != {0, 1}:
        raise ValueError("Invalid complete TRAIN context size/classes")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 2, "created_at_utc": utc_now(), "stage": stage, "config": config,
                "implementation_sha256": sha256(Path(__file__)), "config_sha256": sha256(config_path),
                "dataset_id": dataset.manifest["dataset_id"], "feature_names": names,
                "baseline_config": baseline, "fit_game_ids": fit_ids, "query_game_ids": query_ids,
                "protocol": "Shared pregame tables; separate labels; context before query ET dates; TEST excluded",
                "baseline_comparison": "same-context fixed first booster candidate" if stage == "train-smoke" else "preserved full-TRAIN DEV-selected booster forecasts",
                "diagnostic_limitation": "TRAIN inference diagnostic, not generalization or gradient learning."}
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {stage}: {len(fit_ids)} context, {len(query_ids)} query; shared dataset {manifest['dataset_id']}", flush=True)
    return manifest

def load_prepared(config, stage):
    dataset = shared_dataset(config)
    dataset.verify(label_splits={"train"} if stage == "train-smoke" else {"train", "dev"})
    output = record_directory(config, stage)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["stage"] != stage or experiment_settings(manifest["config"]) != experiment_settings(config):
        raise ValueError("Configuration/stage changed after preparation")
    if manifest.get("schema_version") == 1:
        verify_imported_record(dataset, stage, "manifest.json", manifest)
    else:
        if manifest["dataset_id"] != dataset.manifest["dataset_id"]:
            raise ValueError("Shared dataset identity changed")
        if manifest["implementation_sha256"] != sha256(Path(__file__)):
            raise ValueError("TabPFN implementation changed after preparation")
    rows, outcomes = read_source_rows(dataset, {"train"} if stage == "train-smoke" else {"train", "dev"})
    _, _, expected_fit, expected_query = select_cohort(
        rows, outcomes, stage, config["smoke_games"], config["smoke_context_games"])
    if manifest["fit_game_ids"] != expected_fit or manifest["query_game_ids"] != expected_query:
        raise ValueError("Prepared fit/query membership or order differs from the complete fixed cohort")
    by_id = {row["game_id"]: row for row in rows}
    ids = manifest["fit_game_ids"] + manifest["query_game_ids"]
    if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        raise ValueError("Prepared membership missing or duplicated")
    inputs = [by_id[gid] for gid in ids]
    labels = [outcomes[gid] for gid in ids]
    validate_rows(inputs, labels, manifest["feature_names"], manifest["baseline_config"])
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
    if (ROOT / config.get("dataset_dir", "data/processed")).resolve() != (ROOT / "data/processed").resolve():
        raise ValueError("Inference writes to the published shared dataset; staged datasets are validation-only")
    output = output_directory(config, stage)
    if (output / "run.json").exists():
        raise ValueError("Predictions already exist; use the validator instead of rerunning")
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
        saved = {r["game_id"]: r for r in shared_dataset(config).predictions("gradient_boosting") if r["split"] == "dev"}
        if set(saved) != set(manifest["query_game_ids"]) or any(r["produced_by"] != "train_fit" for r in saved.values()):
            raise ValueError("Frozen comparator membership changed")
        p_boost = np.asarray([saved[gid]["p_boost"] for gid in manifest["query_game_ids"]])
    # Query outcomes become evaluation targets only after both forecasts exist.
    query_y = np.asarray([r["home_win"] for r in labels[count:]], dtype=int)
    probabilities = {"tabpfn": p_tabpfn, "rich_gradient_boosting": p_boost}
    metrics = {name: score_predictions(query_y, p) for name, p in probabilities.items()}
    dates = [r["game_date_et"] for r in inputs[count:]]
    intervals = bootstrap_brier(query_y, probabilities, dates, config["bootstrap_replicates"], config["seed"], "rich_gradient_boosting")
    from nba_dataset import store_predictions
    comparison = [{"game_id": row["game_id"], "game_date_et": row["game_date_et"],
                   "tabpfn": float(p_tabpfn[i]), "rich_gradient_boosting": float(p_boost[i])}
                  for i, row in enumerate(inputs[count:])]
    for model_id, probabilities in (("tabpfn", p_tabpfn), ("tabpfn_comparator", p_boost)):
        store_predictions(ROOT, [{**{k: row[k] for k in ("game_id", "game_date_et", "season", "split")},
                                  "p_home_win": float(probabilities[i]), "produced_by": "train_context"}
                                 for i, row in enumerate(inputs[count:])], model_id=model_id, stage=stage)
    result = {"schema_version": 2, "stage": stage, "status": "complete", "created_at_utc": utc_now(),
              "execution_platform": config["compute_platform"], "runtime_seconds": previous_runtime + time.monotonic() - started,
              "attempts": progress["attempts"],
              "context_audit": progress["context_audit"],
              "manifest_sha256": sha256(output / "manifest.json"),
              "predictions_sha256": prediction_digest(comparison),
              "model": config["model"], "model_parameters": {"random_state": config["seed"], "fit_mode": "fit_preprocessors", "n_preprocessing_jobs": 1, "memory_saving_mode": "auto", "inference_config": {"SUBSAMPLE_SAMPLES": None}},
              "versions": versions,
              "metrics": metrics, "uncertainty": intervals,
              "primary_comparison": {"brier_difference": metrics["tabpfn"]["brier"] - metrics["rich_gradient_boosting"]["brier"],
                                     "difference_95_interval": intervals["difference_vs_reference"]["tabpfn"]},
              "interpretation": "TRAIN diagnostic only; not generalization evidence" if stage == "train-smoke" else "Exploratory DEV comparison; baseline settings previously selected on DEV; TEST remains excluded"}
    write_json(output / "run.json", result)
    progress_path.unlink(missing_ok=True)
    write_report(config, manifest, result)
    print(json.dumps({"stage": stage, "metrics": metrics, "primary_comparison": result["primary_comparison"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", choices=("download-model", "prepare", *STAGES), default="prepare")
    parser.add_argument("--cohort", choices=STAGES, default="train-smoke", help="Cohort for --stage prepare")
    args = parser.parse_args()
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
