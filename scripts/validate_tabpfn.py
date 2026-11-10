"""Validate a saved TRAIN diagnostic or DEV comparison without model inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from collect_pilot import ROOT
from run_gradient_boosting import FEATURE_NAMES, bootstrap_brier, score_predictions, sha256


def check_equal(actual, expected, name):
    """Compare recomputed scores, allowing JSON roundoff but rejecting omissions."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected).issubset(actual):
            raise ValueError(f"Missing or malformed {name}")
        for key, value in expected.items():
            check_equal(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Mismatched {name}")
        for index, value in enumerate(expected):
            check_equal(actual[index], value, f"{name}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not np.isfinite(actual)
                or not np.isclose(actual, expected, rtol=0, atol=1e-12)):
            raise ValueError(f"Mismatched {name}: expected {expected}, got {actual}")
    elif actual != expected:
        raise ValueError(f"Mismatched {name}")


def validate_prediction_rows(manifest, inputs, labels, rows):
    """Reject mismatched cohorts before computing any paired comparison."""
    names = list(FEATURE_NAMES)
    if manifest["feature_names"] != names:
        raise ValueError("Feature order differs from the frozen gradient-boosting baseline")
    input_ids = [row["game_id"] for row in inputs]
    label_ids = [row["game_id"] for row in labels]
    if len(set(input_ids)) != len(input_ids) or label_ids != input_ids:
        raise ValueError("Inputs and separate labels must align uniquely")
    if any(set(row) != {"game_id", "home_win"} for row in labels):
        raise ValueError("Unexpected outcome record schema")
    if any(type(row["home_win"]) is not int or row["home_win"] not in (0, 1) for row in labels):
        raise ValueError("Expected binary home-win labels")
    fit_ids, query_ids = manifest["fit_game_ids"], manifest["query_game_ids"]
    if (not fit_ids or not query_ids or len(set(fit_ids + query_ids)) != len(fit_ids + query_ids)
            or set(fit_ids + query_ids) != set(input_ids)):
        raise ValueError("Fit and query cohorts must be distinct and cover the prepared data")
    by_id = {row["game_id"]: row for row in inputs}
    if max(by_id[gid]["game_date_et"] for gid in fit_ids) >= min(by_id[gid]["game_date_et"] for gid in query_ids):
        raise ValueError("Fit outcomes must come from earlier calendar dates than every query")
    stage = manifest["stage"]
    if stage not in ("train-smoke", "dev"):
        raise ValueError("Only TRAIN diagnostics and DEV comparisons are supported")
    expected_query_split = "train" if stage == "train-smoke" else "dev"
    if (any(by_id[gid]["split"] != "train" for gid in fit_ids)
            or any(by_id[gid]["split"] != expected_query_split for gid in query_ids)):
        raise ValueError("Cohort split is forbidden for this stage; TEST is excluded")
    if any(set(row["features"]) != set(names) for row in inputs):
        raise ValueError("Input feature schema differs from the baseline")
    if any(set(row) & {"home_win", "home_score", "away_score", "outcome"} for row in inputs):
        raise ValueError("Outcomes must remain separate from pregame inputs")
    columns = {"game_id", "game_date_et", "tabpfn", "rich_gradient_boosting"}
    if not rows or any(set(row) != columns for row in rows):
        raise ValueError("Prediction CSV must contain only metadata and the two forecasts")
    if [row["game_id"] for row in rows] != query_ids:
        raise ValueError("Prediction membership/order differs from the query cohort")
    if any(row["game_date_et"] != by_id[row["game_id"]]["game_date_et"] for row in rows):
        raise ValueError("Prediction dates differ from prepared inputs")
    outcomes = {row["game_id"]: row["home_win"] for row in labels}
    y = np.asarray([outcomes[gid] for gid in query_ids], dtype=int)
    predictions = {name: np.asarray([float(row[name]) for row in rows])
                   for name in ("tabpfn", "rich_gradient_boosting")}
    # Shared scorer checks shape, finiteness and range for both models.
    metrics = {name: score_predictions(y, values) for name, values in predictions.items()}
    return y, predictions, metrics


def validate_run(config, stage):
    # Import lazily so pure artifact checks do not import TabPFN or load weights.
    from run_tabpfn import (load_prepared, record_directory, shared_dataset, verify_imported_record,
                           load_comparison_rows, prediction_digest)

    manifest, inputs, labels = load_prepared(config, stage)
    output = record_directory(config, stage)
    run = json.loads((output / "run.json").read_text())
    if run.get("stage") != stage or run.get("status") != "complete":
        raise ValueError("Expected a completed run for the requested stage")
    if run.get("model") != config["model"] or run.get("versions", {}).get("tabpfn") != config["model"]["package_version"]:
        raise ValueError("Recorded model/package differs from the experiment pin")
    if run["manifest_sha256"] != sha256(output / "manifest.json"):
        raise ValueError("Prepared manifest changed after the run")
    rows = load_comparison_rows(config, stage, manifest)
    if run.get("schema_version", 1) == 1:
        dataset = shared_dataset(config)
        verify_imported_record(dataset, stage, "run.json", run)
        original_path = f"data/processed/tabpfn/{stage}/predictions.csv"
        if run.get("artifact_sha256", {}).get("predictions.csv") != dataset.manifest["source_sha256"].get(original_path):
            raise ValueError("Imported run does not bind the prediction artifact")
        if prediction_digest(rows) != dataset.manifest["imported_prediction_sha256"][stage]:
            raise ValueError("Changed imported per-game predictions")
    elif not run.get("predictions_sha256"):
        raise ValueError("Run does not bind the prediction artifact")
    elif run["predictions_sha256"] != prediction_digest(rows):
        raise ValueError("Changed run artifact: shared predictions")
    y, predictions, metrics = validate_prediction_rows(manifest, inputs, labels, rows)
    uncertainty = bootstrap_brier(
        y, predictions, [row["game_date_et"] for row in rows],
        config["bootstrap_replicates"], config["seed"], "rich_gradient_boosting",
    )
    primary = {
        "brier_difference": metrics["tabpfn"]["brier"] - metrics["rich_gradient_boosting"]["brier"],
        "difference_95_interval": uncertainty["difference_vs_reference"]["tabpfn"],
    }
    check_equal(run["metrics"], metrics, "metrics")
    check_equal(run["uncertainty"], uncertainty, "uncertainty")
    check_equal(run["primary_comparison"], primary, "primary_comparison")
    return {"stage": stage, "fit_games": len(manifest["fit_game_ids"]),
            "query_games": len(rows), "features": len(manifest["feature_names"]),
            "metrics": metrics, "primary_comparison": primary}


ADAPTATION_VARIANTS = ("synthetic", "finetune", "combined")


def validate_adaptation_protocol(protocol, baseline_manifest, inputs):
    """Every arm shares the frozen DEV cohort and chronological TRAIN partition."""
    for field in ("feature_names", "fit_game_ids", "query_game_ids", "baseline_config"):
        if protocol.get(field) != baseline_manifest[field]:
            raise ValueError(f"Adaptation {field} differs from the frozen baseline")
    last_train_season = baseline_manifest["baseline_config"]["split_seasons"]["train"][1]
    earlier = [row["game_id"] for row in inputs
               if row["split"] == "train" and row["season"] < last_train_season]
    validation = [row["game_id"] for row in inputs
                  if row["split"] == "train" and row["season"] == last_train_season]
    if not earlier or not validation:
        raise ValueError("Adaptation needs earlier TRAIN and last-season TRAIN validation")
    if protocol.get("inner_fit_game_ids") != earlier or protocol.get("inner_validation_game_ids") != validation:
        raise ValueError("Inner fitting and validation must use the fixed chronological TRAIN seasons")
    if earlier + validation != protocol["fit_game_ids"]:
        raise ValueError("Inner partitions must cover every original TRAIN game")


def validate_adaptation_rows(variant, manifest, inputs, labels, rows, baseline_rows):
    """Recompute an arm on exactly the original games and original comparators."""
    if variant not in ADAPTATION_VARIANTS:
        raise ValueError("Expected synthetic, finetune or combined; TEST is excluded")
    columns = {"game_id", "game_date_et", variant, "tabpfn", "rich_gradient_boosting"}
    if not rows or any(set(row) != columns for row in rows):
        raise ValueError("Adaptation predictions must contain only matching IDs, dates and three forecasts")
    original_columns = columns - {variant}
    preserved = [{name: row[name] for name in original_columns} for row in rows]
    y, predictions, metrics = validate_prediction_rows(manifest, inputs, labels, preserved)
    _, baseline_predictions, _ = validate_prediction_rows(manifest, inputs, labels, baseline_rows)
    for name in ("tabpfn", "rich_gradient_boosting"):
        if not np.array_equal(predictions[name], baseline_predictions[name]):
            raise ValueError(f"Frozen {name} comparison probabilities changed")
    predictions = {variant: np.asarray([float(row[variant]) for row in rows]), **predictions}
    metrics = {variant: score_predictions(y, predictions[variant]), **metrics}
    return y, predictions, metrics


def validate_adaptation_evidence(variant, protocol, run, directory):
    """Bind generated examples and selected weights to earlier TRAIN provenance."""
    synthetic_rows = run["context_audit"]["synthetic_rows"]
    if variant != "finetune":
        synthetic_dir = directory / "synthetic_data"
        manifest_path = synthetic_dir / "manifest.json"
        if run["synthetic_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("Synthetic manifest changed after inference")
        synthetic = json.loads(manifest_path.read_text())
        if synthetic.get("input_partition_game_ids") != protocol["inner_fit_game_ids"]:
            raise ValueError("Generator input partition must be earlier TRAIN only")
        source_ids = synthetic.get("source_game_ids", [])
        if (not source_ids or len(source_ids) != len(set(source_ids))
                or not set(source_ids).issubset(protocol["inner_fit_game_ids"])):
            raise ValueError("Generator fitting rows must be unique earlier TRAIN games")
        if synthetic.get("feature_names") != protocol["feature_names"]:
            raise ValueError("Synthetic feature order differs from real inputs")
        artifacts = synthetic.get("artifact_sha256", {})
        for name in ("features.npy", "labels.npy"):
            if artifacts.get(name) != sha256(synthetic_dir / name):
                raise ValueError(f"Synthetic artifact changed: {name}")
        features = np.load(synthetic_dir / "features.npy", allow_pickle=False)
        labels = np.load(synthetic_dir / "labels.npy", allow_pickle=False)
        if features.shape != (synthetic_rows, len(protocol["feature_names"])) or np.isinf(features).any():
            raise ValueError("Synthetic feature dimensions or values are invalid")
        if (labels.shape != (synthetic_rows,) or not np.issubdtype(labels.dtype, np.integer)
                or not np.isin(labels, [0, 1]).all()):
            raise ValueError("Synthetic labels must be a separate binary vector")
    training = run.get("training")
    if variant != "synthetic":
        if not isinstance(training, dict):
            raise ValueError("Missing fine-tuning audit")
        if (training.get("fit_game_ids") != protocol["inner_fit_game_ids"]
                or training.get("validation_game_ids") != protocol["inner_validation_game_ids"]):
            raise ValueError("Fine-tuning must use the fixed chronological TRAIN partitions")
        if training.get("synthetic_rows") != synthetic_rows:
            raise ValueError("Fine-tuning augmentation differs from inference context")
        checkpoint = Path(training.get("checkpoint_path", ""))
        if not str(checkpoint) or str(checkpoint) == "." or checkpoint.is_absolute() or ".." in checkpoint.parts:
            raise ValueError("Selected checkpoint must be a repository-relative file")
        if training.get("checkpoint_sha256") != sha256(ROOT / checkpoint):
            raise ValueError("Selected fine-tuning checkpoint changed")
        for field in ("weights_before_sha256", "weights_after_sha256"):
            digest = training.get(field, "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Fine-tuning must record before/after weight checksums")


def validate_adaptation(variant):
    """Validate stored adaptation evidence without importing or running a model."""
    from run_tabpfn import load_adaptation

    if variant not in ADAPTATION_VARIANTS:
        raise ValueError("Expected synthetic, finetune or combined; TEST is excluded")
    protocol, manifest, inputs, labels = load_adaptation()
    validate_adaptation_protocol(protocol, manifest, inputs)
    directory = ROOT / "data/processed/tabpfn/adaptation"
    output = directory / variant
    run = json.loads((output / "run.json").read_text())
    if run.get("stage") != "adapt-dev" or run.get("status") != "complete" or run.get("variant") != variant:
        raise ValueError("Expected a completed adaptation run for the requested variant")
    if run.get("manifest_sha256") != sha256(directory / "manifest.json"):
        raise ValueError("Adaptation manifest changed after inference")
    artifacts = run.get("artifact_sha256", {})
    if "predictions.csv" not in artifacts:
        raise ValueError("Adaptation run does not bind its prediction artifact")
    for name, expected in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Adaptation artifact paths must stay within its output directory")
        if sha256(output / relative) != expected:
            raise ValueError(f"Changed adaptation artifact: {name}")
    audit = run.get("context_audit", {})
    original = len(manifest["fit_game_ids"])
    synthetic = audit.get("synthetic_rows")
    if type(synthetic) is not int or synthetic < 0 or (synthetic > 0) != (variant != "finetune"):
        raise ValueError("Context augmentation count differs from the experiment arm")
    if variant != "finetune" and synthetic != protocol["config"]["generator"]["n_samples"]:
        raise ValueError("Synthetic row count differs from the frozen augmentation size")
    expected_context = {"original_train_rows": original, "synthetic_rows": synthetic,
                        "submitted_rows": original + synthetic,
                        "model_n_train_samples": original + synthetic,
                        "row_subsampling_applied": False}
    check_equal(audit, expected_context, "context_audit")
    if (run.get("training") is not None) != (variant != "synthetic"):
        raise ValueError("Fine-tuning evidence differs from the experiment arm")
    if (run.get("synthetic_manifest_sha256") is not None) != (variant != "finetune"):
        raise ValueError("Synthetic provenance differs from the experiment arm")
    validate_adaptation_evidence(variant, protocol, run, directory)
    with (output / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    from run_tabpfn import CONFIG_PATH, load_comparison_rows, load_config
    # adaptation_config() pins the baseline to config/tabpfn.json; its DEV forecasts are in the shared table.
    baseline_rows = load_comparison_rows(load_config(CONFIG_PATH), "dev", manifest)
    y, predictions, metrics = validate_adaptation_rows(variant, manifest, inputs, labels, rows, baseline_rows)
    settings = protocol["config"]
    uncertainty = bootstrap_brier(y, predictions, [row["game_date_et"] for row in rows],
                                 settings["bootstrap_replicates"], settings["seed"], "tabpfn")
    primary = {"brier_difference": metrics[variant]["brier"] - metrics["tabpfn"]["brier"],
               "difference_95_interval": uncertainty["difference_vs_reference"][variant]}
    check_equal(run["metrics"], metrics, "metrics")
    check_equal(run["uncertainty"], uncertainty, "uncertainty")
    check_equal(run["primary_comparison"], primary, "primary_comparison")
    return {"stage": "adapt-dev", "variant": variant, "fit_games": original,
            "synthetic_rows": synthetic, "query_games": len(rows),
            "features": len(manifest["feature_names"]), "metrics": metrics,
            "primary_comparison": primary}


def main():
    from run_tabpfn import CONFIG_PATH, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", choices=("train-smoke", "dev"), default="train-smoke")
    parser.add_argument("--dataset", type=Path, help="Validate a staged shared dataset")
    parser.add_argument("--variant", choices=ADAPTATION_VARIANTS,
                        help="Validate a saved adaptation arm instead of the original baseline")
    args = parser.parse_args()
    if args.variant:
        result = validate_adaptation(args.variant)
    else:
        config = load_config(args.config.resolve())
        if args.dataset:
            config["dataset_dir"] = str(args.dataset.resolve())
        result = validate_run(config, args.stage)
    print(json.dumps({"status": "PASS", **result}, indent=2))


if __name__ == "__main__":
    main()
