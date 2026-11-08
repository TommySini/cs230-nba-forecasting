"""Reconstruct rich features and verify temporal lineage, fitted models and metrics."""
import csv
import json
from pathlib import Path
import warnings

import joblib
import numpy as np

from collect_pilot import ROOT
from collect_rich_stats import rebuild_from_record
from rich_stats_features import RichStatsIndex, PLAYER_FEATURE_NAMES
from run_gradient_boosting import load_verified_history, prepare_examples, read_rows, score_predictions, sha256
from run_rich_gradient_boosting import (CONFIG_PATH, load_config, numeric_features, paired_intervals,
                                        verify_box_manifest, verify_hashes)


def main():
    config = load_config(CONFIG_PATH)
    output, report = ROOT / config["output_dir"], ROOT / config["report_dir"]
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    lock = json.loads((output / "prediction_lock.json").read_text())
    results = json.loads((report / "metrics.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    assert manifest["config"] == results["config"] == config
    verify_hashes(manifest["source_sha256"] | manifest["artifact_sha256"] | manifest["original_artifacts"])
    verify_hashes(lock["artifact_sha256"] | lock["model_sha256"])
    assert results["prediction_lock_sha256"] == sha256(output / "prediction_lock.json")
    baseline = json.loads((ROOT / config["baseline_config"]).read_text())
    verify_box_manifest(config, baseline)
    history = load_verified_history(baseline)
    history_by_id = {g["game_id"]: g for g in history}
    original_inputs, all_labels, _ = prepare_examples(history, baseline)
    box_records = read_rows(ROOT / config["box_records"])
    for record in box_records:
        rebuilt = rebuild_from_record(history_by_id[record["game_id"]], record, root=ROOT)
        assert rebuilt == record, f"Normalized record not bound to source: {record['game_id']}"
    index = RichStatsIndex(history, box_records, min_previous_games=config["min_previous_games"],
                           cutoff_minutes=baseline["cutoff_minutes"])
    inputs, labels = read_rows(output / "inputs.jsonl"), read_rows(output / "labels.jsonl")
    expected_inputs, expected_excluded = [], []
    for original in original_inputs:
        rich = index.for_game(history_by_id[original["game_id"]], include_players=True)
        if not rich["eligible"]:
            expected_excluded.append(original["game_id"])
            continue
        values = numeric_features(original["features"] | rich["features"], set(PLAYER_FEATURE_NAMES))
        expected_inputs.append(dict(original, features=values, rich_source_lineage=rich["lineage"],
                                    rich_coverage=rich["coverage"]))
    assert inputs == expected_inputs, "Feature/source-lineage reconstruction differs"
    assert expected_excluded == [r["game_id"] for r in read_rows(output / "excluded.jsonl")]
    ids = {r["game_id"] for r in inputs}
    assert labels == [r for r in all_labels if r["game_id"] in ids]
    assert [r["game_id"] for r in inputs] == [r["game_id"] for r in labels]
    assert len(ids) == len(inputs)
    y = np.array([r["home_win"] for r in labels])
    test_mask = np.array([r["split"] == "test" for r in inputs])
    fit_mask = ~test_mask
    train_ids = [r["game_id"] for r in inputs if r["split"] != "test"]
    dev_ids = [r["game_id"] for r in inputs if r["split"] == "validation"]
    test_ids = [r["game_id"] for r in inputs if r["split"] == "test"]
    assert test_ids == lock["test_game_ids"]
    assert not set(train_ids).intersection(test_ids)
    for row in inputs:
        for side in ("home", "away"):
            for game_id in row["prior_game_ids"][side]:
                prior = history_by_id[game_id]
                assert prior["season"] == row["season"]
                assert prior["game_date_et"] < row["game_date_et"]
    with (output / "test_predictions_locked.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    assert [r["game_id"] for r in saved] == test_ids
    assert "home_win" not in saved[0], "Prediction lock must not include target outcomes"
    probabilities = {}
    for name, detail in selection["fit_details"].items():
        bundle = joblib.load(output / "models" / f"{name}.joblib")
        feature_names = manifest["feature_names"][detail["arm"]]
        assert bundle["feature_names"] == feature_names
        assert bundle["fit_game_ids"] == train_ids
        assert bundle["selection_game_ids"] == dev_ids
        assert bundle["candidate"] == detail["candidate"]
        model = bundle["model"]
        for key, value in (config["model_parameters"] | detail["candidate"] | {"random_state": config["seed"]}).items():
            assert model.named_steps["model"].get_params()[key] == value
        x = np.asarray([[r["features"][n] for n in feature_names] for r in inputs], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected_medians = np.nanmedian(x[fit_mask], axis=0)
        expected_medians = np.nan_to_num(expected_medians, nan=0)
        np.testing.assert_allclose(model.named_steps["imputer"].statistics_, expected_medians, atol=0, rtol=0)
        p = model.predict_proba(x[test_mask])[:, 1]
        np.testing.assert_allclose(p, [float(r[name]) for r in saved], atol=1e-15, rtol=0)
        actual = score_predictions(y[test_mask], p)
        for key, value in actual.items():
            np.testing.assert_allclose(value, results["test"][name][key], atol=1e-12, rtol=0)
        probabilities[name] = p
    richer = min(("team", "team_player"),
                 key=lambda a: (selection["arms"][a]["selected_dev_brier"], ("team", "team_player").index(a)))
    assert lock["primary_richer_model"] == richer + "_selected"
    for arm, entry in selection["arms"].items():
        chosen = min(entry["trials"], key=lambda r: (r["brier"], r["candidate_index"]))
        assert entry["selected_candidate_index"] == chosen["candidate_index"]
    uncertainty = paired_intervals(y[test_mask], probabilities, [r["game_date_et"] for r in saved],
                                   config["bootstrap_replicates"], config["seed"])
    assert uncertainty == results["uncertainty"]
    print(f"PASS: {len(box_records)} raw-bound box scores; {len(inputs)} reconstructed prior-only rows; "
          f"six saved model variants reproduce {len(saved)} paired TEST forecasts and metrics. "
          "Imputation uses TRAIN+DEV only; original baseline hashes unchanged.")


if __name__ == "__main__":
    main()
