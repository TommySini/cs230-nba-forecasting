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
    from run_tabpfn import load_prepared

    manifest, inputs, labels = load_prepared(config, stage)
    output = ROOT / config["output_dir"] / stage
    run = json.loads((output / "run.json").read_text())
    if run.get("stage") != stage or run.get("status") != "complete":
        raise ValueError("Expected a completed run for the requested stage")
    if run.get("model") != config["model"] or run.get("versions", {}).get("tabpfn") != config["model"]["package_version"]:
        raise ValueError("Recorded model/package differs from the experiment pin")
    if run["manifest_sha256"] != sha256(output / "manifest.json"):
        raise ValueError("Prepared manifest changed after the run")
    artifacts = run["artifact_sha256"]
    if "predictions.csv" not in artifacts:
        raise ValueError("Run does not bind the prediction artifact")
    for name, expected in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Run artifact paths must stay within its output directory")
        if sha256(output / relative) != expected:
            raise ValueError(f"Changed run artifact: {name}")
    with (output / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
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


def main():
    from run_tabpfn import CONFIG_PATH, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", choices=("train-smoke", "dev"), default="train-smoke")
    args = parser.parse_args()
    result = validate_run(load_config(args.config.resolve()), args.stage)
    print(json.dumps({"status": "PASS", **result}, indent=2))


if __name__ == "__main__":
    main()
