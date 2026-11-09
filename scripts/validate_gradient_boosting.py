"""Rebuild the boosting inputs from sources; verify splits, prior-only lineage, models, predictions and metrics."""
import argparse
import hashlib
import json
import re
import subprocess
import warnings

import joblib
import numpy as np

from run_gradient_boosting import (CONFIG_PATH, DATA_DIR, FEATURE_NAMES, METRICS_PATH, ROOT, SIDES, SPLITS, arrays,
                                   evaluate, load_box_scores, load_verified_history, predict, prepare_examples,
                                   read_rows, score_predictions, season_splits, sha256)


def verify_hashes(results, source_revision=None, root=ROOT):
    """Check original code in a pinned commit; data and artifacts always use current bytes.

    The rest of this validator replays the frozen predictions with today's code.
    Historical source verification never substitutes for that equivalence check.
    """
    if source_revision and not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full, immutable Git commit ID")
    for name, expected in results["source_sha256"].items():
        if source_revision and name.startswith("scripts/"):
            body = subprocess.run(["git", "-C", str(root), "cat-file", "blob", f"{source_revision}:{name}"],
                                  check=True, capture_output=True).stdout
            actual = hashlib.sha256(body).hexdigest()
        else:
            actual = sha256(root / name)
        assert actual == expected, f"Source hash changed: {name}"
    for name, expected in results["artifact_sha256"].items():
        assert sha256(root / name) == expected, f"Artifact hash changed: {name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", help="Full Git commit for the original source hashes; data is always checked live")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    results = json.loads(METRICS_PATH.read_text())
    assert results["config"] == config
    verify_hashes(results, args.source_revision)
    history = load_verified_history(config)
    box_records, box_manifest = load_box_scores(config)
    assert box_manifest["histories"] == [{"path": s["path"], "sha256": sha256(ROOT / s["path"])} for s in config["histories"]], \
        "Box scores were built from other schedules"
    inputs, labels, excluded = [read_rows(DATA_DIR / name) for name in ("inputs.jsonl", "labels.jsonl", "excluded.jsonl")]
    assert (inputs, labels, excluded) == prepare_examples(history, box_records, config), "Saved rows do not rebuild from sources"
    by_id, season_split = {g["game_id"]: g for g in history}, season_splits(config)
    for row in inputs:
        assert row["split"] == season_split[row["season"]] and set(row["features"]) == set(FEATURE_NAMES)
        assert not {"home_win", "home_score", "away_score"} & set(row)
        for side in SIDES:
            assert set(row["missing_box_score_game_ids"][side]) <= set(row["prior_game_ids"][side])
            for prior in row["prior_game_ids"][side]:
                assert by_id[prior]["season"] == row["season"] and by_id[prior]["game_date_et"] < row["game_date_et"]
    dates = {name: [r["game_date_et"] for r in inputs if r["split"] == name] for name in SPLITS}
    assert max(dates["train"]) < min(dates["dev"]) and max(dates["dev"]) < min(dates["test"])

    x, y = arrays(inputs, labels)
    splits, seasons = np.array([r["split"] for r in inputs]), np.array([r["season"] for r in inputs])
    ids, train = np.array([r["game_id"] for r in inputs]), splits == "train"
    bundle = joblib.load(DATA_DIR / "model.joblib")  # Only this locally generated, hash-checked artifact.
    best = min(results["dev_candidates"], key=lambda r: (r["brier"], r["candidate_index"]))["candidate_index"]
    settings = {**config["model_parameters"], **config["candidates"][best], "random_state": config["seed"]}
    assert bundle["candidate_index"] == results["selected_candidate_index"] == best
    assert bundle["settings"] == results["selected_parameters"] == settings
    assert bundle["feature_names"] == list(FEATURE_NAMES) == results["feature_names"]
    dev = splits == "dev"
    dev_score = score_predictions(y[dev], bundle["final"]["model"].predict_proba(x[dev])[:, 1])
    assert dev_score == {k: results["dev_candidates"][best][k] for k in dev_score}, "Selected model's DEV score differs"
    assert bundle["final"]["fit_game_ids"] == ids[train].tolist()
    assert bundle["final"]["selection_game_ids"] == ids[splits == "dev"].tolist()
    assert sorted(bundle["prior_seasons"]) == sorted(set(seasons[train].tolist()))[1:]
    fits = [(bundle["final"], train)]
    for season, entry in bundle["prior_seasons"].items():
        assert entry["fit_game_ids"] == ids[train & (seasons < season)].tolist(), f"Season {season} fit membership"
        fits.append((entry, train & (seasons < season)))
    for entry, mask in fits:
        model = entry["model"]
        assert all(model.named_steps["model"].get_params()[k] == v for k, v in settings.items())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nan_to_num(np.nanmedian(x[mask], axis=0), nan=0)
        np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, medians)

    p, produced = predict(bundle["final"]["model"], {s: e["model"] for s, e in bundle["prior_seasons"].items()}, x, splits, seasons)
    saved = read_rows(DATA_DIR / "predictions.jsonl")
    kept = [i for i in range(len(inputs)) if produced[i] is not None]
    assert [r["game_id"] for r in saved] == ids[kept].tolist(), "Every eligible game needs one prediction row"
    for i, row in zip(kept, saved):
        assert set(row) == {"game_id", "season", "game_date_et", "split", "p_boost", "produced_by"}
        assert (row["season"], row["game_date_et"], row["split"], row["produced_by"]) == (
            inputs[i]["season"], inputs[i]["game_date_et"], inputs[i]["split"], produced[i])
    np.testing.assert_allclose([r["p_boost"] for r in saved], p[kept], rtol=0, atol=1e-15)
    assert len(kept) == len(inputs) - results["train_games_without_forecast"]
    home_rate, evaluation, by_season = evaluate(p, y, splits, seasons, np.array([r["game_date_et"] for r in inputs]), config)
    assert home_rate == results["train_home_win_rate"] and evaluation == results["evaluation"] and by_season == results["train_forecasts"]
    print(f"PASS: {len(inputs)} rebuilt prior-only rows ({len(excluded)} exclusions); TRAIN/DEV/TEST chronological; "
          f"final model fitted on {train.sum()} TRAIN games only, {len(bundle['prior_seasons'])} TRAIN-season models fitted on "
          f"earlier seasons only; {len(saved)} predictions and all metrics reproduce.")


if __name__ == "__main__":
    main()
