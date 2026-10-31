"""Verify baseline lineage, chronological separation, saved model and metrics."""
import csv
import json

import joblib
import numpy as np

from run_gradient_boosting import (CONFIG_PATH, DATA_DIR, FEATURE_NAMES, MODEL_NAMES, REPORT_DIR,
                                   ROOT, arrays, load_verified_history, prepare_examples, read_rows,
                                   score_predictions, sha256)


def main():
    config = json.loads(CONFIG_PATH.read_text())
    results = json.loads((REPORT_DIR / "metrics.json").read_text())
    assert config == results["config"]
    for name, expected in {**results["source_sha256"], **results["artifact_sha256"]}.items():
        assert sha256(ROOT / name) == expected, f"Hash changed: {name}"
    history = load_verified_history(config)
    expected_inputs, expected_labels, expected_excluded = prepare_examples(history, config)
    inputs, labels, excluded = [read_rows(DATA_DIR / name) for name in ("inputs.jsonl", "labels.jsonl", "excluded.jsonl")]
    assert inputs == expected_inputs and labels == expected_labels and excluded == expected_excluded
    assert len({r["game_id"] for r in inputs}) == len(inputs)
    game_by_id = {g["game_id"]: g for g in history}
    for row in inputs:
        assert set(row["features"]) == set(FEATURE_NAMES)
        assert not {"home_win", "home_score", "away_score"}.intersection(row)
        for side in ("home", "away"):
            for prior_id in row["prior_game_ids"][side]:
                prior = game_by_id[prior_id]
                assert prior["season"] == row["season"]
                assert prior["game_date_et"] < row["game_date_et"] and prior_id != row["game_id"]
    ordered = [[r for r in inputs if r["split"] == split] for split in ("train", "validation", "test")]
    assert ordered[0][-1]["game_date_et"] < ordered[1][0]["game_date_et"]
    assert ordered[1][-1]["game_date_et"] < ordered[2][0]["game_date_et"]
    bundle = joblib.load(DATA_DIR / "model.joblib")  # Only this locally generated artifact.
    assert list(bundle["feature_names"]) == list(FEATURE_NAMES) == results["feature_names"]
    assert bundle["fit_game_ids"] == [r["game_id"] for r in inputs if r["split"] != "test"]
    assert bundle["selection_game_ids"] == [r["game_id"] for r in ordered[1]]
    assert not set(bundle["fit_game_ids"]) & {r["game_id"] for r in ordered[2]}
    for parameter, value in results["selected_parameters"].items():
        assert bundle["model"].get_params()[parameter] == value
    selected = min(results["validation_candidates"], key=lambda r: (r["brier"], r["candidate_index"]))
    for name, value in config["candidates"][selected["candidate_index"]].items():
        assert results["selected_parameters"][name] == value
    x, y = arrays(inputs, labels)
    test = np.array([r["split"] == "test" for r in inputs])
    predictions = {"gradient_boosting": bundle["model"].predict_proba(x[test])[:, 1],
                   "historical_home_rate": np.full(test.sum(), y[~test].mean()),
                   "constant_50_percent": np.full(test.sum(), 0.5)}
    with (DATA_DIR / "test_predictions.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    assert [r["game_id"] for r in saved] == [r["game_id"] for r in ordered[2]]
    np.testing.assert_array_equal([int(r["home_win"]) for r in saved], y[test])
    for name in MODEL_NAMES:
        np.testing.assert_allclose([float(r[name]) for r in saved], predictions[name], rtol=0, atol=1e-15)
        actual = score_predictions(y[test], predictions[name])
        for key, value in actual.items():
            np.testing.assert_allclose(value, results["test"][name][key], rtol=0, atol=1e-12)
    print(f"PASS: {len(inputs)} prior-only examples; disjoint chronological splits; no test IDs in fit/selection; saved model reproduces {test.sum()} held-out predictions and all metrics.")


if __name__ == "__main__":
    main()
