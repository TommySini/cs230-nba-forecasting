"""Replay source lineage, paired selection and saved LM predictions offline."""
import argparse
import csv
import gc
import json

import joblib
import numpy as np

from run_text_ablation import (ARMS, CONFIG_PATH, DATA_DIR, FEATURE_NAMES, REPORT_DIR,
                               ROOT, encode, get_runtime, load_prepared, make_cohort,
                               make_model, paired_brier_interval, predict, score_predictions,
                               select_shared_epoch, sha256, trainable_digest, verify_selection)


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    protocol, records, labels = load_prepared(config)
    recreated, recreated_labels = make_cohort(config)
    assert recreated == records and recreated_labels == labels
    assert len(records) == len({r["game_id"] for r in records}) == 140
    assert all(not {"home_win", "home_score", "away_score"} & set(r) for r in records)
    subsets = {s: [i for i, r in enumerate(records) if r["split"] == s] for s in ("train", "dev", "test")}
    assert {s: len(indices) for s, indices in subsets.items()} == config["split_counts"]
    for early, later in (("train", "dev"), ("dev", "test")):
        assert max(records[i]["game_date_et"] for i in subsets[early]) < min(records[i]["game_date_et"] for i in subsets[later])
    selection = json.loads((REPORT_DIR / "selection.json").read_text())
    result = json.loads((REPORT_DIR / "metrics.json").read_text())
    verify_selection(selection)
    assert result["protocol_sha256"] == sha256(REPORT_DIR / "protocol.json")
    assert result["selection_sha256"] == sha256(REPORT_DIR / "selection.json")
    assert result["test_predictions_sha256"] == sha256(DATA_DIR / "test_predictions.csv")
    assert set(selection["initial_trainable_sha256"]) == set(ARMS)
    assert len(set(selection["initial_trainable_sha256"].values())) == 1
    for split in ("train", "dev"):
        assert selection[f"{split}_game_ids"] == [records[i]["game_id"] for i in subsets[split]]
    train_ids = selection["train_game_ids"]
    assert selection["training_order_game_ids"] == [[train_ids[i] for i in np.random.default_rng(config["seed"] + epoch).permutation(len(train_ids))]
                                                   for epoch in range(1, config["epochs"] + 1)]
    history = [{"arm": r["arm"], "epoch": int(r["epoch"]), "dev_brier": float(r["dev_brier"])} for r in csv_rows(REPORT_DIR / "learning_curves.csv")]
    choice, scores = select_shared_epoch(history)
    assert choice["epoch"] == selection["selected_epoch"] == result["selected_epoch"]
    assert scores == selection["selection_scores"]
    saved_dev = csv_rows(DATA_DIR / "dev_predictions.csv")
    saved_test = csv_rows(DATA_DIR / "test_predictions.csv")
    test = [records[i] for i in subsets["test"]]
    dev = [records[i] for i in subsets["dev"]]
    y_test = np.asarray([labels[i]["home_win"] for i in subsets["test"]])
    y_dev = np.asarray([labels[i]["home_win"] for i in subsets["dev"]])
    assert [r["game_id"] for r in saved_test] == [r["game_id"] for r in test]
    np.testing.assert_array_equal([int(r["home_win"]) for r in saved_test], y_test)
    assert not set(r["game_id"] for r in test) & (set(train_ids) | set(selection["dev_game_ids"]))
    device, tokenizer = get_runtime(config, args.device)
    initial = make_model(config, tokenizer)
    assert trainable_digest(initial) == selection["initial_trainable_sha256"][ARMS[0]]
    del initial
    probabilities = {}
    for arm in ARMS:
        model = make_model(config, tokenizer, DATA_DIR / arm / f"epoch_{choice['epoch']}").to(device)
        p_dev = predict(model, encode(dev, arm, tokenizer, config), tokenizer, device, config["batch_size"])
        saved_subset = [r for r in saved_dev if r["arm"] == arm and int(r["epoch"]) == choice["epoch"]]
        assert [r["game_id"] for r in saved_subset] == [r["game_id"] for r in dev]
        np.testing.assert_allclose(p_dev, [float(r["probability"]) for r in saved_subset], atol=1e-6, rtol=0)
        expected_dev = next(r["dev_brier"] for r in history if r["arm"] == arm and r["epoch"] == choice["epoch"])
        np.testing.assert_allclose(score_predictions(y_dev, p_dev)["brier"], expected_dev, atol=1e-6, rtol=0)
        probabilities[arm] = predict(model, encode(test, arm, tokenizer, config), tokenizer, device, config["batch_size"])
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    original = joblib.load(ROOT / "data/processed/gradient_boosting/model.joblib")
    assert not set(original["fit_game_ids"]) & {r["game_id"] for r in test}
    probabilities["original_boosting_reference"] = original["model"].predict_proba(np.array([[r["features"][n] for n in FEATURE_NAMES] for r in test]))[:, 1]
    for name, p in probabilities.items():
        np.testing.assert_allclose(p, [float(r[name]) for r in saved_test], atol=1e-6, rtol=0)
        actual = score_predictions(y_test, p)
        for metric, value in actual.items():
            np.testing.assert_allclose(value, result["test"][name][metric], atol=1e-6, rtol=0)
    interval = paired_brier_interval(y_test, probabilities[ARMS[0]], probabilities[ARMS[1]], [r["game_date_et"] for r in test], config)
    for field in ("text_minus_stats", "paired_date_bootstrap_ci95"):
        np.testing.assert_allclose(interval[field], result["paired_comparison"][field], atol=1e-6, rtol=0)
    print("PASS: 140 source-verified paired games; chronological 108/14/18; identical initialization/order/budget; dev-only common epoch; both saved models reproduce dev/test predictions and Brier comparison; original baseline unchanged.")


if __name__ == "__main__":
    main()
