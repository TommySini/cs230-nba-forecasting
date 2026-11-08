"""Compare prior-only richer box-score features against the preserved baseline."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from collect_pilot import ROOT, write_json, write_jsonl
from run_gradient_boosting import (FEATURE_NAMES as ORIGINAL_FEATURES, load_verified_history,
                                   prepare_examples, read_rows, score_predictions, sha256,
                                   write_csv)

CONFIG_PATH = ROOT / "config/rich_gradient_boosting.json"
ARMS = ("original", "team", "team_player")
SPLITS = ("train", "validation", "test")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprints():
    paths = [ROOT / "config/gradient_boosting.json", ROOT / "scripts/run_gradient_boosting.py"]
    paths += sorted((ROOT / "reports/gradient_boosting").glob("*"))
    paths += sorted((ROOT / "data/processed/gradient_boosting").glob("*"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}


def verify_hashes(expected):
    for relative, digest in expected.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Changed frozen artifact: {relative}")


def load_config(path):
    config = json.loads(path.read_text())
    if tuple(config["arms"]) != ARMS or config["selection_metric"] != "brier":
        raise ValueError("Expected the declared feature arms and Brier selection")
    if config["model_parameters"]["n_iter_no_change"] is not None:
        raise ValueError("Internal early stopping would change the declared protocol")
    return config


def numeric_features(values, player_names):
    """Strict JSON null for unknown player statistics, never missing-as-zero."""
    result = {}
    for name, value in values.items():
        if value is None or (isinstance(value, (int, float, np.number)) and np.isnan(value)):
            if name not in player_names:
                raise ValueError(f"Missing mandatory numeric feature: {name}")
            result[name] = None
        elif not np.isfinite(value):
            raise ValueError(f"Non-finite feature: {name}")
        else:
            result[name] = float(value)
    return result


def verify_box_manifest(config, baseline):
    manifest_path = ROOT / config["box_manifest"]
    box_path = ROOT / config["box_records"]
    box_manifest = json.loads(manifest_path.read_text())
    for name, expected in box_manifest["artifact_sha256"].items():
        if sha256(manifest_path.parent / name) != expected:
            raise ValueError(f"Box-score artifact changed: {name}")
    if box_path.resolve() != (manifest_path.parent / "games.jsonl").resolve():
        raise ValueError("Configured box records are not the manifest's normalized games")
    if box_manifest["history_sha256"] != sha256(ROOT / baseline["history_path"]):
        raise ValueError("Box-score history differs from the verified baseline history")
    if box_manifest["normalizer_sha256"] != sha256(ROOT / "scripts/collect_rich_stats.py"):
        raise ValueError("Rebuild the box-score cache with the current normalizer")
    return box_manifest


def prepare_dataset(config, config_path):
    from rich_stats_features import (RichStatsIndex, TEAM_FEATURE_NAMES, PLAYER_FEATURE_NAMES,
                                      WINDOWS, FEATURE_DEFINITIONS)

    baseline = json.loads((ROOT / config["baseline_config"]).read_text())
    if config["split_seasons"] != baseline["split_seasons"]:
        raise ValueError("Comparison must use the original chronological seasons")
    if config["windows"] != [name for name, _ in WINDOWS] or config["min_previous_games"] != baseline["min_previous_games"]:
        raise ValueError("Configured windows/warmup differ from the feature implementation")
    history = load_verified_history(baseline)
    original_inputs, labels, warmup = prepare_examples(history, baseline)
    by_id = {g["game_id"]: g for g in history}
    label_by_id = {r["game_id"]: r for r in labels}
    box_path = ROOT / config["box_records"]
    box_manifest = ROOT / config["box_manifest"]
    if not box_path.exists() or not box_manifest.exists():
        raise FileNotFoundError("Collect and validate rich historical statistics before preparing features")
    source_manifest = verify_box_manifest(config, baseline)
    index = RichStatsIndex(history, read_rows(box_path), min_previous_games=config["min_previous_games"],
                           cutoff_minutes=baseline["cutoff_minutes"])
    feature_names = {
        "original": list(ORIGINAL_FEATURES),
        "team": list(ORIGINAL_FEATURES) + list(TEAM_FEATURE_NAMES),
        "team_player": list(ORIGINAL_FEATURES) + list(TEAM_FEATURE_NAMES) + list(PLAYER_FEATURE_NAMES),
    }
    if len(set(feature_names["team_player"])) != len(feature_names["team_player"]):
        raise ValueError("Feature groups must have disjoint names")
    inputs, output_labels, exclusions = [], [], []
    team_eligible = {name: 0 for name in SPLITS}
    for original in original_inputs:
        game = by_id[original["game_id"]]
        rich = index.for_game(game, include_players=True)
        team_only = index.for_game(game, include_players=False)
        team_eligible[original["split"]] += int(team_only["eligible"])
        if not rich["eligible"]:
            exclusions.append({"game_id": original["game_id"], "split": original["split"],
                               "game_date_et": original["game_date_et"],
                               "reasons": rich["reasons"], "coverage": rich["coverage"]})
            continue
        values = dict(original["features"])
        values.update(rich["features"])
        if set(values) != set(feature_names["team_player"]):
            raise ValueError(f"Unexpected feature schema for {original['game_id']}")
        values = numeric_features(values, set(PLAYER_FEATURE_NAMES))
        record = dict(original, features=values, rich_source_lineage=rich["lineage"],
                      rich_coverage=rich["coverage"])
        inputs.append(record)
        output_labels.append(label_by_id[original["game_id"]])
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prediction_lock.json").exists():
        raise ValueError("Existing prediction lock: use a new experiment directory for changed data")
    for split in SPLITS:
        if not any(r["split"] == split for r in inputs):
            raise ValueError(f"No complete-source examples in {split}")
    write_jsonl(output / "inputs.jsonl", inputs)
    write_jsonl(output / "labels.jsonl", output_labels)
    write_jsonl(output / "excluded.jsonl", exclusions)
    write_csv(output / "split_manifest.csv", [
        {k: r[k] for k in ("game_id", "season", "split", "game_date_et", "cutoff_utc")}
        for r in inputs])
    source_paths = [config_path, ROOT / config["baseline_config"], box_path, box_manifest,
                    ROOT / baseline["history_path"], ROOT / "scripts/run_gradient_boosting.py",
                    ROOT / "scripts/collect_pilot.py", ROOT / "scripts/collect_rich_stats.py",
                    ROOT / "scripts/rich_stats_features.py", Path(__file__).resolve()]
    manifest = {
        "experiment_id": config["experiment_id"], "created_at_utc": utc_now(),
        "config": config, "feature_names": feature_names,
        "feature_definitions": FEATURE_DEFINITIONS,
        "box_source_summary": {k: source_manifest[k] for k in
                               ("provider", "provider_counts", "requested_games", "eligible_history_games",
                                "excluded_non_franchise_games", "complete_team_games", "complete_player_games", "failed_games")},
        "history_games": len(history), "warmup_exclusions": len(warmup),
        "coverage": {split: {"original_eligible": sum(r["split"] == split for r in original_inputs),
                              "team_eligible": team_eligible[split],
                              "common_eligible": sum(r["split"] == split for r in inputs),
                              "games_with_missing_player_features": sum(
                                  r["split"] == split and any(v is None for v in r["features"].values()) for r in inputs)}
                     for split in SPLITS},
        "feature_missingness": {split: {name: sum(r["features"][name] is None for r in inputs if r["split"] == split)
                                           for name in feature_names["team_player"]}
                                for split in SPLITS},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                            (output / n for n in ("inputs.jsonl", "labels.jsonl", "excluded.jsonl", "split_manifest.csv"))},
        "original_artifacts": fingerprints(),
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "joblib")},
    }
    write_json(output / "dataset_manifest.json", manifest)
    print("Prepared common cohort:", json.dumps(manifest["coverage"]), flush=True)
    return manifest


def model_for(config, candidate):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", GradientBoostingClassifier(**config["model_parameters"], **candidate,
                                             random_state=config["seed"])),
    ])


def choose_on_dev(train_x, train_y, dev_x, dev_y, config):
    """There is deliberately no TEST argument to feature/model selection."""
    trials, fitted = [], []
    for number, candidate in enumerate(config["candidates"]):
        model = model_for(config, candidate).fit(train_x, train_y)
        probabilities = model.predict_proba(dev_x)[:, 1]
        trials.append({"candidate_index": number, **candidate,
                       **score_predictions(dev_y, probabilities)})
        fitted.append((model, probabilities))
    selected = min(trials, key=lambda r: (r["brier"], r["candidate_index"]))
    return selected["candidate_index"], trials, fitted


def fit_and_lock(config):
    output = ROOT / config["output_dir"]
    lock_path = output / "prediction_lock.json"
    if lock_path.exists():
        raise ValueError("Predictions are already locked; evaluate the saved run without retraining")
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Configuration changed since preparation")
    verify_hashes(manifest["source_sha256"] | manifest["artifact_sha256"] | manifest["original_artifacts"])
    inputs, labels = read_rows(output / "inputs.jsonl"), read_rows(output / "labels.jsonl")
    if [r["game_id"] for r in inputs] != [r["game_id"] for r in labels]:
        raise ValueError("Inputs and labels are not aligned")
    y = np.array([r["home_win"] for r in labels])
    masks = {name: np.array([r["split"] == name for r in inputs]) for name in SPLITS}
    fit_mask = masks["train"] | masks["validation"]
    tests = [r for r in inputs if r["split"] == "test"]
    devs = [r for r in inputs if r["split"] == "validation"]
    test_predictions = [{k: r[k] for k in ("game_id", "game_date_et", "home_abbr", "away_abbr")}
                        for r in tests]
    dev_predictions = [{k: r[k] for k in ("game_id", "game_date_et")} for r in devs]
    selections, model_hashes, fit_details = {}, {}, {}
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)
    for arm in ARMS:
        names = manifest["feature_names"][arm]
        x = np.asarray([[r["features"][n] for n in names] for r in inputs], dtype=float)
        selected, trials, dev_models = choose_on_dev(x[masks["train"]], y[masks["train"]],
                                                     x[masks["validation"]], y[masks["validation"]], config)
        selections[arm] = {"selected_candidate_index": selected, "trials": trials,
                           "selected_dev_brier": trials[selected]["brier"], "feature_count": len(names)}
        fixed_index = next(i for i, c in enumerate(config["candidates"]) if c == config["fixed_candidate"])
        fitted_candidates = {}
        for role, index in (("fixed", fixed_index), ("selected", selected)):
            name = f"{arm}_{role}"
            candidate = config["candidates"][index]
            if index not in fitted_candidates:
                fitted_candidates[index] = model_for(config, candidate).fit(x[fit_mask], y[fit_mask])
            model = fitted_candidates[index]
            path = models_dir / f"{name}.joblib"
            joblib.dump({"model": model, "feature_names": names,
                         "fit_game_ids": [r["game_id"] for r in inputs if r["split"] != "test"],
                         "selection_game_ids": [r["game_id"] for r in devs],
                         "candidate": candidate, "experiment_id": config["experiment_id"]}, path)
            model_hashes[str(path.relative_to(ROOT))] = sha256(path)
            # Save probabilities without TEST outcomes; evaluation is a separate stage.
            for row, probability in zip(test_predictions, model.predict_proba(x[masks["test"]])[:, 1]):
                row[name] = float(probability)
            for row, probability in zip(dev_predictions, dev_models[index][1]):
                row[name] = float(probability)
            fit_details[name] = {"arm": arm, "candidate": candidate, "feature_count": len(names),
                                 "feature_importance": sorted(
                                     [{"feature": n, "importance": float(v)} for n, v in zip(names, model.named_steps["model"].feature_importances_)],
                                     key=lambda r: (-r["importance"], r["feature"]))}
        print(f"DEV {arm}: {len(names)} features, Brier {trials[selected]['brier']:.6f}, {config['candidates'][selected]}", flush=True)
    selected_richer = min(("team", "team_player"), key=lambda a: (selections[a]["selected_dev_brier"], ARMS.index(a)))
    write_json(output / "selection.json", {"arms": selections, "selected_richer_arm": selected_richer,
                                           "fit_details": fit_details, "selected_using": "TRAIN fit / DEV Brier only"})
    write_csv(output / "test_predictions_locked.csv", test_predictions)
    write_csv(output / "dev_predictions.csv", dev_predictions)
    locked_files = [output / n for n in ("dataset_manifest.json", "selection.json", "test_predictions_locked.csv", "dev_predictions.csv")]
    lock = {"experiment_id": config["experiment_id"], "created_at_utc": utc_now(),
            "primary_richer_model": selected_richer + "_selected", "model_sha256": model_hashes,
            "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in locked_files},
            "test_game_ids": [r["game_id"] for r in tests],
            "selection_access": "No TEST outcomes passed to candidate or feature-arm selection"}
    write_json(lock_path, lock)
    verify_hashes(manifest["original_artifacts"])
    print(f"Locked {len(tests)} TEST forecasts; selected richer arm {selected_richer}.", flush=True)
    return lock


def paired_intervals(y, predictions, dates, repeats, seed):
    unique, inverse = np.unique(dates, return_inverse=True)
    counts = np.bincount(inverse)
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(repeats, len(unique)))
    denominators = counts[sampled].sum(axis=1)
    draws = {}
    for name, p in predictions.items():
        sums = np.bincount(inverse, weights=(np.asarray(p) - y) ** 2)
        draws[name] = sums[sampled].sum(axis=1) / denominators
    reference = draws["original_fixed"]
    result = {"method": "Paired ET-calendar-date cluster bootstrap, percentile 95% interval",
            "replicates": repeats, "dates": len(unique), "seed": seed,
            "brier_intervals": {name: np.quantile(v, [.025, .975]).tolist() for name, v in draws.items()},
            "brier_difference_vs_original": {name: np.quantile(v - reference, [.025, .975]).tolist()
                                               for name, v in draws.items()},
            "limitations": "Conditional on fitted models; does not capture training uncertainty or dependence across dates."}
    if "original_selected" in draws:
        result["brier_difference_vs_original_selected"] = {
            name: np.quantile(v - draws["original_selected"], [.025, .975]).tolist()
            for name, v in draws.items()}
    return result


def evaluate(config):
    output, report = ROOT / config["output_dir"], ROOT / config["report_dir"]
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    lock = json.loads((output / "prediction_lock.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Configuration changed after prediction lock")
    verify_hashes(manifest["source_sha256"] | manifest["artifact_sha256"] | manifest["original_artifacts"])
    verify_hashes(lock["artifact_sha256"] | lock["model_sha256"])
    selection = json.loads((output / "selection.json").read_text())
    labels = {r["game_id"]: r["home_win"] for r in read_rows(output / "labels.jsonl")}
    with (output / "test_predictions_locked.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if [r["game_id"] for r in rows] != lock["test_game_ids"]:
        raise ValueError("Prediction membership differs from locked TEST IDs")
    y = np.array([labels[r["game_id"]] for r in rows])
    names = list(selection["fit_details"])
    predictions = {name: np.array([float(r[name]) for r in rows]) for name in names}
    metrics = {name: score_predictions(y, p) for name, p in predictions.items()}
    intervals = paired_intervals(y, predictions, [r["game_date_et"] for r in rows],
                                 config["bootstrap_replicates"], config["seed"])
    original_report = json.loads((ROOT / "reports/gradient_boosting/metrics.json").read_text())
    original_same_ids = False
    with (ROOT / "data/processed/gradient_boosting/test_predictions.csv").open() as stream:
        old_rows = list(csv.DictReader(stream))
    original_same_ids = [r["game_id"] for r in old_rows] == [r["game_id"] for r in rows]
    exact_original_reproduction = None
    complete_fit = all(c["original_eligible"] == c["common_eligible"] for c in manifest["coverage"].values())
    if original_same_ids and complete_fit:
        old_p = np.array([float(r["gradient_boosting"]) for r in old_rows])
        np.testing.assert_allclose(predictions["original_fixed"], old_p, rtol=0, atol=1e-12)
        exact_original_reproduction = True
    primary = lock["primary_richer_model"]
    result = {"experiment_id": config["experiment_id"], "config": config,
              "evaluated_at_utc": utc_now(), "coverage": manifest["coverage"],
              "box_source_summary": manifest["box_source_summary"],
              "feature_missingness": manifest["feature_missingness"],
              "feature_definitions": manifest["feature_definitions"],
              "feature_names": manifest["feature_names"], "selection": selection,
              "test": metrics, "uncertainty": intervals,
              "primary_comparison": {"richer_model": primary, "reference": "original_fixed",
                                      "brier_difference": metrics[primary]["brier"] - metrics["original_fixed"]["brier"],
                                      "difference_95_interval": intervals["brier_difference_vs_original"][primary]},
              "original_full_test": original_report["test"]["gradient_boosting"],
              "original_full_test_same_ids": original_same_ids,
              "exact_original_reproduction": exact_original_reproduction,
              "prediction_lock_sha256": sha256(output / "prediction_lock.json"),
              "source_manifest_sha256": sha256(ROOT / config["box_manifest"]),
              "versions": manifest["versions"]}
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "metrics.json", result)
    write_report(report, result)
    write_csv(report / "comparison.csv", [
        {"model": name, "features": selection["fit_details"][name]["feature_count"],
         **metrics[name], "brier_difference_vs_original": metrics[name]["brier"] - metrics["original_fixed"]["brier"],
         "difference_ci_low": intervals["brier_difference_vs_original"][name][0],
         "difference_ci_high": intervals["brier_difference_vs_original"][name][1]}
        for name in names])
    write_csv(report / "feature_definitions.csv", [{"metric": name, "definition": definition}
                                                  for name, definition in manifest["feature_definitions"].items()])
    write_csv(report / "feature_missingness.csv", [
        {"split": split, "feature": name, "missing_games": count,
         "total_games": manifest["coverage"][split]["common_eligible"]}
        for split, counts in manifest["feature_missingness"].items() for name, count in counts.items()])
    verify_hashes(manifest["original_artifacts"])
    print(json.dumps(result["primary_comparison"], indent=2), flush=True)
    return result


def write_report(report, result):
    primary = result["primary_comparison"]
    test = result["test"]
    selection = result["selection"]
    delta = primary["brier_difference"]
    ci = primary["difference_95_interval"]
    lines = ["# Rich-statistics gradient boosting", "",
             f"Primary DEV-selected richer model: **{primary['richer_model']}**. On the same "
             f"{test['original_fixed']['games']:,} held-out games, Brier is "
             f"**{test[primary['richer_model']]['brier']:.6f}** versus "
             f"**{test['original_fixed']['brier']:.6f}** for the original features. "
             f"Difference (richer minus original): **{delta:+.6f}**, paired date-bootstrap "
             f"95% interval **[{ci[0]:+.6f}, {ci[1]:+.6f}]**. Negative favors richer features.", "",
             "## Same-capacity feature comparison", "",
             "All rows below use the original fixed 100 depth-1 trees and otherwise identical training settings.", "",
             "| Features | Count | Brier | Log loss | AUC | Accuracy |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        name = arm + "_fixed"
        row = test[name]
        lines.append(f"| {arm} | {selection['fit_details'][name]['feature_count']} | {row['brier']:.6f} | {row['log_loss']:.6f} | {row['roc_auc']:.4f} | {row['accuracy']:.1%} |")
    lines += ["", "## Matched DEV-selected comparison", "",
              "Every feature arm receives the same four candidates (depth 1/2 × 100/200 trees). "
              "The richer arm and its settings were selected using DEV, before TEST scoring.", "",
              "| Features | Selected depth / trees | DEV Brier | TEST Brier | TEST accuracy |",
              "| --- | --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        name = arm + "_selected"
        candidate = selection["fit_details"][name]["candidate"]
        lines.append(f"| {arm} | {candidate['max_depth']} / {candidate['n_estimators']} | {selection['arms'][arm]['selected_dev_brier']:.6f} | {test[name]['brier']:.6f} | {test[name]['accuracy']:.1%} |")
    lines += ["", "## Data and temporal boundaries", "",
              "| Split | Original eligible | Team-source eligible | Common paired cohort | Any missing player window |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for split, counts in result["coverage"].items():
        lines.append(f"| {split} | {counts['original_eligible']} | {counts['team_eligible']} | {counts['common_eligible']} | {counts['games_with_missing_player_features']} |")
    lines += ["", "TRAIN: 2023–24; DEV: 2024–25; TEST: 2025–26, through April 10. "
              "Final models refit TRAIN + DEV with settings fixed. Within TEST, earlier completed "
              "games update rolling features but model weights are not refit.", "",
              f"Source coverage: {result['box_source_summary']['complete_team_games']:,} complete team "
              f"box scores, {result['box_source_summary']['complete_player_games']:,} complete player "
              f"box scores. Providers: {json.dumps(result['box_source_summary']['provider_counts'], sort_keys=True)}. "
              "Audited official NBA pages repair invalid ESPN records; the original raw records and "
              "replacement-source mapping remain preserved.", "",
              f"The source audit excludes {result['box_source_summary']['excluded_non_franchise_games']} "
              "All-Star games involving non-franchise teams that ESPN labeled with regular-season "
              "type metadata. They were already excluded by the original five-game warmup and do "
              "not change the original eligible forecasting cohort. Original history and baseline "
              "artifacts remain intact.", "",
              "Features are calculated only from earlier Eastern-calendar dates in the same season. "
              "Full prior TEAM box-score coverage is required. Team ratios use pooled box-score counts; "
              "possession-derived ratings and pace are estimates, not official NBA ratings. "
              "Player features summarize prior workload and rotation continuity; actual current-game "
              "participants, final lineups and injury outcomes are excluded.", "",
              "Missing player windows retain their actual game membership and carry explicit coverage "
              "indicators. Their missing numeric values use medians learned only from TRAIN during "
              "DEV selection, then TRAIN+DEV for final fitting. No TEST data informs imputation. "
              "Ambiguous minute placeholders are not treated as zero minutes.", "",
              "Historical public box scores were retrieved retrospectively. Source hashes and dates "
              "are preserved, but original publication times and later stat corrections are not "
              "independently archived. Only normalized numeric fields reach the feature builder.", "",
              "Current-game injury availability was not added: the existing report sample covers "
              "only 140/3,375 original eligible games and has additional timestamp/coverage issues. "
              "A missing injury report is not interpreted as a healthy roster.", "",
              f"Original full-test prediction reproduction on the complete cohort: **{result['exact_original_reproduction']}**. "
              "If the matched cohort differs, the original-feature arms are refit on its same TRAIN/DEV games; "
              "the published full-cohort result is separately retained in metrics.json.", "",
              "## Interpretation limits", "",
              "This TEST season has already been examined in prior experiments. Treat this as an "
              "exploratory backtest. The paired interval is conditional on fitted models and does not "
              "include training uncertainty or correlation across calendar days. Added features can "
              "fail to improve prediction; all declared arms are reported. No text model was trained "
              "or evaluated by this run.", "",
              "## Reproduce", "", "```bash",
              ".venv/bin/python scripts/collect_rich_stats.py",
              ".venv/bin/python scripts/run_rich_gradient_boosting.py --stage all",
              ".venv/bin/python scripts/validate_rich_gradient_boosting.py", "```", "",
              "An existing prediction lock prevents accidental retraining. Use `--stage evaluate` "
              "to reproduce metrics from the saved run. A changed experiment requires a new output "
              "directory and protocol. See `protocol.md`, `metrics.json`, and `comparison.csv`.", ""]
    (report / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", choices=("prepare", "fit", "evaluate", "all"), default="all")
    args = parser.parse_args()
    path = args.config.resolve()
    config = load_config(path)
    if args.stage in ("prepare", "all"):
        prepare_dataset(config, path)
    if args.stage in ("fit", "all"):
        fit_and_lock(config)
    if args.stage in ("evaluate", "all"):
        evaluate(config)


if __name__ == "__main__":
    main()
