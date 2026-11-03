"""Replay source, chronology, boosting lineage and saved hybrid predictions."""
import argparse
import csv
import io
import json

import joblib
import numpy as np
from pypdf import PdfReader

from audit_game_notes import decode_archive_body
from collect_pilot import ROOT, write_json
from review_game_notes import content_checks
from run_gradient_boosting import (FEATURE_NAMES, arrays, load_verified_history, new_model,
                                   prepare_examples, score_predictions, sha256)
from run_text_ablation import get_runtime, make_model, paired_brier_interval
from run_boosting_notes import (CONFIG, DATA, REPORT, encode, evidence_text, load_prepared,
                                predict, verify_note, verify_selection, whole_date_splits)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    protocol, rows, labels = load_prepared(config)
    base = json.loads((ROOT / config["baseline_config"]).read_text())
    history = load_verified_history(base)
    games = {g["game_id"]: g for g in history}
    numeric, original_labels, _ = prepare_examples(history, base)
    numeric_by_id = {r["game_id"]: r for r in numeric}
    y_by_id = {r["game_id"]: r for r in original_labels}
    original = joblib.load(ROOT / "data/processed/gradient_boosting/model.joblib")
    fit_ids = set(original["fit_game_ids"])
    if len({r["game_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate cohort games")
    if labels != [y_by_id[r["game_id"]] for r in rows]:
        raise ValueError("Outcome labels do not reproduce from normalized schedule")
    reassigned = whole_date_splits(rows, fit_ids, config)
    if [(r["game_id"], r["split"]) for r in rows] != [(r["game_id"], r["split"]) for r in reassigned]:
        raise ValueError("Frozen split does not follow the declared date/fitting constraints")
    device, tokenizer = get_runtime(config, "cpu" if args.data_only else args.device)
    review_config_path = ROOT / "config/game_notes_content_review.json"
    review_config = json.loads(review_config_path.read_text())
    for row in rows:
        numeric_row = numeric_by_id[row["game_id"]]
        for key in ("features", "prior_game_ids", "cutoff_utc", "game_date_et", "home_abbr", "away_abbr"):
            if row[key] != numeric_row[key]:
                raise ValueError(f"Numeric lineage changed: {key}")
        note = row["note_source"]
        verify_note(note, numeric_row)
        if (note["team"] not in review_config["reviewed_publisher_templates"]
                or note["source"]["sha256"] in review_config["excluded_source_hashes"]
                or note["review"]["review_config_sha256"] != sha256(review_config_path)):
            raise ValueError("Manual/template review decision does not authorize this source")
        if row["game_date_et"] < config["minimum_game_date"]:
            raise ValueError("A cohort game predates the allowed checkpoint window")
        raw = decode_archive_body((ROOT / note["source"]["raw_path"]).read_bytes())
        front = PdfReader(io.BytesIO(raw)).pages[0].extract_text() or ""
        reasons, _, clean = content_checks(front, games[row["game_id"]], note["team"], history)
        if reasons or clean != note["text"]:
            raise ValueError("Reviewed source does not reproduce")
        evidence, choice = evidence_text(clean, tokenizer, config["evidence_token_budget"])
        if evidence != row["note_evidence"] or choice != row["evidence_selection"]:
            raise ValueError("Evidence selection changed")
        if "home_win" in row or "home_score" in row or "away_score" in row:
            raise ValueError("Target label in pregame inputs")
    encode(rows, tokenizer, config)
    lineage = json.loads((DATA / "boosting_lineage.json").read_text())
    for fold in sorted({r["boosting_fold"] for r in rows}):
        target = [r for r in rows if r["boosting_fold"] == fold]
        saved = joblib.load(DATA / f"boosting_folds/{fold}.joblib")
        expected = [r for r in numeric if r["game_id"] in fit_ids] if fold == "original_frozen" else [r for r in numeric if r["game_date_et"] < fold]
        ids = [r["game_id"] for r in expected]
        if saved["fit_game_ids"] != ids or lineage[fold]["fit_game_ids"] != ids:
            raise ValueError("Boosting fold did not fit exactly the declared earlier games")
        if set(ids) & {r["game_id"] for r in target} or max(r["game_date_et"] for r in expected) >= min(r["game_date_et"] for r in target):
            raise ValueError("Target or future labels in boosting fit")
        x, y = arrays(expected, [y_by_id[r["game_id"]] for r in expected])
        rebuilt = original["model"] if fold == "original_frozen" else new_model(base, config["boosting_parameters"]).fit(x, y)
        features = [[r["features"][n] for n in FEATURE_NAMES] for r in target]
        np.testing.assert_allclose(rebuilt.predict_proba(features)[:, 1], [r["p_boost"] for r in target], atol=1e-12)
        np.testing.assert_allclose(saved["model"].predict_proba(features)[:, 1], [r["p_boost"] for r in target], atol=1e-12)
    result = {"games": len(rows), "boosting_folds": len(lineage), "source_and_chronology": "passed", "data_only": args.data_only}
    if not args.data_only:
        selection = verify_selection()
        curves = read_csv(REPORT / "learning_curves.csv")
        choice = min(curves, key=lambda r: (float(r["dev_brier"]), int(r["epoch"])))
        if int(choice["epoch"]) != selection["selected_epoch"]:
            raise ValueError("Selected epoch not the best predeclared dev checkpoint")
        for split in ("train", "dev"):
            if selection[split + "_game_ids"] != [r["game_id"] for r in rows if r["split"] == split]:
                raise ValueError("Training/selection membership changed")
        model = make_model(config, tokenizer, DATA / f"adapters/epoch_{selection['selected_epoch']}").to(device)
        dev = [r for r in rows if r["split"] == "dev"]
        dev_p = predict(model, dev, encode(dev, tokenizer, config), tokenizer, device, config)
        dev_csv = [r for r in read_csv(DATA / "dev_predictions.csv") if int(r["epoch"]) == selection["selected_epoch"]]
        if [r["game_id"] for r in dev_csv] != [r["game_id"] for r in dev]:
            raise ValueError("Dev prediction membership changed")
        np.testing.assert_allclose(dev_p, [float(r["p_hybrid"]) for r in dev_csv], atol=1e-6, rtol=0)
        all_dev = read_csv(DATA / "dev_predictions.csv")
        dev_y = np.asarray([y_by_id[r["game_id"]]["home_win"] for r in dev])
        for curve in curves:
            saved_dev = [r for r in all_dev if int(r["epoch"]) == int(curve["epoch"])]
            if [r["game_id"] for r in saved_dev] != [r["game_id"] for r in dev]:
                raise ValueError("Incomplete epoch dev predictions")
            brier = score_predictions(dev_y, np.asarray([float(r["p_hybrid"]) for r in saved_dev]))["brier"]
            np.testing.assert_allclose(brier, float(curve["dev_brier"]), atol=1e-12)
        test = [r for r in rows if r["split"] == "test"]
        test_p = predict(model, test, encode(test, tokenizer, config), tokenizer, device, config)
        test_csv = read_csv(DATA / "test_predictions.csv")
        if [r["game_id"] for r in test_csv] != [r["game_id"] for r in test]:
            raise ValueError("Test prediction membership changed")
        np.testing.assert_allclose(test_p, [float(r["boosting_plus_notes"]) for r in test_csv], atol=1e-6, rtol=0)
        y = np.asarray([y_by_id[r["game_id"]]["home_win"] for r in test])
        boost = np.asarray([r["p_boost"] for r in test])
        np.testing.assert_array_equal(y, [int(r["home_win"]) for r in test_csv])
        np.testing.assert_allclose(boost, [float(r["original_boosting"]) for r in test_csv], atol=1e-12)
        metrics = json.loads((REPORT / "metrics.json").read_text())
        if metrics["test_predictions_sha256"] != sha256(DATA / "test_predictions.csv") or metrics["selection_sha256"] != sha256(REPORT / "selection.json"):
            raise ValueError("Evaluation artifact changed")
        for arm, p in (("original_boosting", boost), ("boosting_plus_notes", test_p)):
            replay = score_predictions(y, p)
            for name, value in replay.items():
                if value is not None:
                    np.testing.assert_allclose(value, metrics["test"][arm][name], atol=1e-6, rtol=0)
        paired = paired_brier_interval(y, boost, test_p, [r["game_date_et"] for r in test], config)
        np.testing.assert_allclose(paired["text_minus_stats"], metrics["paired_comparison"]["hybrid_minus_boosting"], atol=1e-6)
        np.testing.assert_allclose(paired["paired_date_bootstrap_ci95"], metrics["paired_comparison"]["paired_date_bootstrap_ci95"], atol=1e-6)
        result.update(prediction_replay="passed within absolute tolerance 1e-6", selected_epoch=selection["selected_epoch"])
    write_json(REPORT / ("data_validation.json" if args.data_only else "validation.json"), result)
    print("PASS: " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
