"""Reproducible statistics-only gradient boosting with season-held-out evaluation."""
import argparse
from contextlib import redirect_stdout
import csv
from datetime import timedelta
import hashlib
from importlib.metadata import version
import io
import json
import os
import platform

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from collect_pilot import ROOT, Cache, collect_games, iso, prior_features, stamp, write_json, write_jsonl

CONFIG_PATH = ROOT / "config/gradient_boosting.json"
DATA_DIR = ROOT / "data/processed/gradient_boosting"
REPORT_DIR = ROOT / "reports/gradient_boosting"
BASE_FEATURES = ("previous_games", "season_win_rate", "last_5_win_rate",
                 "last_5_point_differential", "days_since_previous_game", "back_to_back")
FEATURE_NAMES = tuple(f"{side}_{name}" for side in ("home", "away", "difference") for name in BASE_FEATURES)
MODEL_NAMES = ("gradient_boosting", "historical_home_rate", "constant_50_percent")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_verified_history(config):
    history = read_rows(ROOT / config["history_path"])
    schedule_config = json.loads((ROOT / config["history_config"]).read_text())
    # Verify all schedule days and raw checksums, including games outside the
    # 160-game text audit. Only normalized numeric history reaches feature code.
    with redirect_stdout(io.StringIO()):
        verified = collect_games(Cache(offline=True), schedule_config)
    if history != verified:
        raise ValueError("History does not match the complete checksum-verified schedule cache")
    if len({game["game_id"] for game in history}) != len(history):
        raise ValueError("Duplicate historical game IDs")
    return history


def prepare_examples(history, config):
    seasons = config["split_seasons"]
    ordered = [seasons[name] for name in ("train", "validation", "test")]
    if ordered != sorted(set(ordered)):
        raise ValueError("Training, validation and test seasons must be distinct and chronological")
    if config["min_previous_games"] < 1 or config["cutoff_minutes"] <= 0:
        raise ValueError("Warmup and cutoff must be positive")
    history = sorted(history, key=lambda g: (g["scheduled_start_utc"], g["game_id"]))
    inputs, labels, excluded = [], [], []
    for game in history:
        if game["season"] not in ordered:
            raise ValueError(f"Unassigned season: {game['season']}")
        features = {side: prior_features(game, history, side) for side in ("home", "away")}
        if any(f["previous_games"] < config["min_previous_games"] for f in features.values()):
            excluded.append({"game_id": game["game_id"], "season": game["season"],
                             "reason": "fewer_than_minimum_prior_games",
                             "home_previous_games": features["home"]["previous_games"],
                             "away_previous_games": features["away"]["previous_games"]})
            continue
        values = {f"{side}_{name}": float(features[side][name])
                  for side in ("home", "away") for name in BASE_FEATURES}
        values.update({f"difference_{name}": values[f"home_{name}"] - values[f"away_{name}"] for name in BASE_FEATURES})
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite features: {game['game_id']}")
        split = next(name for name, season in seasons.items() if game["season"] == season)
        record = {key: game[key] for key in ("game_id", "season", "scheduled_start_utc", "game_date_et", "home_abbr", "away_abbr")}
        record.update(split=split, cutoff_utc=iso(stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"])),
                      features=values, prior_game_ids={side: features[side]["prior_game_ids"] for side in ("home", "away")})
        inputs.append(record)
        labels.append({"game_id": game["game_id"], "home_win": int(game["home_score"] > game["away_score"])})
    return inputs, labels, excluded


def arrays(inputs, labels):
    if [r["game_id"] for r in inputs] != [r["game_id"] for r in labels]:
        raise ValueError("Inputs and outcome labels must align by game ID")
    x = np.array([[r["features"][name] for name in FEATURE_NAMES] for r in inputs], dtype=float)
    y = np.array([r["home_win"] for r in labels], dtype=int)
    return x, y


def score_predictions(y, probability):
    p = np.asarray(probability, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected one finite probability in [0, 1] per outcome")
    return {"games": len(y), "brier": float(brier_score_loss(y, p, pos_label=1, scale_by_half=True)),
            "log_loss": float(log_loss(y, np.column_stack((1 - p, p)), labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "mean_home_probability": float(p.mean()), "observed_home_win_rate": float(y.mean())}


def new_model(config, candidate):
    return GradientBoostingClassifier(**config["model_parameters"], **candidate, random_state=config["seed"])


def select_candidate(x_train, y_train, x_validation, y_validation, config):
    # There is deliberately no test-data argument to selection. No random
    # validation split, early stopping, calibration fit or threshold search.
    if config["selection_metric"] != "brier" or config["model_parameters"]["n_iter_no_change"] is not None:
        raise ValueError("This protocol requires Brier selection and disabled internal early stopping")
    results = []
    for index, candidate in enumerate(config["candidates"]):
        model = new_model(config, candidate).fit(x_train, y_train)
        metrics = score_predictions(y_validation, model.predict_proba(x_validation)[:, 1])
        results.append({"candidate_index": index, **candidate, **metrics})
    best = min(results, key=lambda row: (row["brier"], row["candidate_index"]))
    return config["candidates"][best["candidate_index"]], results


def bootstrap_brier(y, predictions, dates, repeats, seed):
    # Resample ET calendar days together; preserve paired predictions and
    # outcomes. This is conditional on the fitted model, not training uncertainty.
    unique, inverse = np.unique(dates, return_inverse=True)
    counts = np.bincount(inverse)
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(repeats, len(unique)))
    denominators = counts[sampled].sum(axis=1)
    draws = {}
    for name, probability in predictions.items():
        sums = np.bincount(inverse, weights=(probability - y) ** 2)
        draws[name] = sums[sampled].sum(axis=1) / denominators
    interval = lambda values: np.quantile(values, [0.025, 0.975]).tolist()
    return {"method": "paired ET-calendar-day cluster bootstrap, percentile 95% interval",
            "replicates": repeats, "days": len(unique), "seed": seed,
            "brier_intervals": {name: interval(value) for name, value in draws.items()},
            "gb_minus_home_rate_interval": interval(draws["gradient_boosting"] - draws["historical_home_rate"]),
            "limitations": "Conditional on this fitted model and season; does not capture training uncertainty or serial dependence across days"}


def calibration_rows(y, probability, bins):
    bucket = np.minimum((probability * bins).astype(int), bins - 1)
    return [{"bin_low": i / bins, "bin_high": (i + 1) / bins, "games": int(np.sum(bucket == i)),
             "mean_probability": float(probability[bucket == i].mean()),
             "observed_home_win_rate": float(y[bucket == i].mean())}
            for i in range(bins) if np.any(bucket == i)]


def write_plot(results, calibration):
    os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    colors = ["#176b65", "#8795a2", "#bec6ce"]
    names = list(MODEL_NAMES)
    values = [results["test"][name]["brier"] for name in names]
    left.barh(["Gradient boosting", "Historical home rate", "50% every game"], values, color=colors)
    left.invert_yaxis()
    left.set_xlim(0, max(values) * 1.22)
    left.set_xlabel("Brier score · lower is better")
    left.set_title("Held-out probability accuracy", loc="left", weight="bold")
    for i, value in enumerate(values):
        left.text(value + 0.005, i, f"{value:.4f}", va="center")
    right.plot([0, 1], [0, 1], color="#aeb6be", linestyle="--", label="Perfect calibration")
    px = [r["mean_probability"] for r in calibration]
    py = [r["observed_home_win_rate"] for r in calibration]
    right.plot(px, py, color=colors[0], linewidth=1.5)
    right.scatter(px, py, s=[15 + r["games"] * 0.4 for r in calibration], color=colors[0], zorder=3)
    right.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted home-win probability", ylabel="Observed home-win rate")
    right.set_title("Calibration · marker area reflects games", loc="left", weight="bold")
    right.legend(loc="upper left", frameon=False, fontsize=9)
    season = results["config"]["split_seasons"]["test"]
    fig.suptitle(f'NBA statistics baseline | {season}–{str(season + 1)[-2:]} holdout · {results["test"][names[0]]["games"]:,} games', fontsize=14, weight="bold")
    fig.savefig(REPORT_DIR / "baseline.png", dpi=180)
    plt.close(fig)


def write_report(result):
    gb, reference = result["test"]["gradient_boosting"], result["test"]["historical_home_rate"]
    skill = 1 - gb["brier"] / reference["brier"]
    lines = ["# Gradient-boosting statistics baseline", "",
             f'Held-out Brier score: **{gb["brier"]:.4f}**, versus **{reference["brier"]:.4f}** for the historical home-win-rate reference ({skill:.1%} relative reduction).', "",
             "This is a retrospective statistics-only result. It is not a text-model result or a contamination-cleared LLM benchmark.", "",
             "## Protocol", "",
             f'- Source: {result["history_games"]:,} checksum-verified historical regular-season results, through April 10 in each season. No additional network requests.',
             f'- Eligibility: both teams have at least {result["config"]["min_previous_games"]} previous games in the same season; {result["excluded_games"]} warmup games excluded.',
             '- Prediction cutoff: 60 minutes before the retrieved scheduled start. Features use only earlier ET calendar dates in the same season.',
             '- Features: prior-game count, season win rate, last-five win rate, last-five average point differential, rest days and back-to-back status, for each side and home-minus-away differences (18 numeric features).',
             '- Model selection: four predetermined shallow gradient-boosting classifiers, selected by validation Brier score. No random splits, internal early stopping, post-hoc probability calibration or test-driven tuning.',
             '- Final fit: training plus validation seasons, with the selected parameters fixed. The historical-rate reference also uses only these fitted seasons.',
             '- During the test season, earlier completed test games update rolling features. Their outcomes are available before later predictions; the model itself is not refitted.',
             '- No injury text, narratives, current standings, odds, player data, team identifiers or target-game outcomes enter the feature matrix.', "",
             "| Role | Season | Games | First game | Last game |", "| --- | --- | ---: | --- | --- |"]
    for name, split in result["splits"].items():
        lines.append(f'| {name} | {split["season"]}–{str(split["season"] + 1)[-2:]} | {split["games"]} | {split["first_date"]} | {split["last_date"]} |')
    lines.extend(["", "## Validation model selection", "", "| Depth | Trees | Brier | Log loss | ROC AUC | Accuracy |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in result["validation_candidates"]:
        lines.append(f'| {row["max_depth"]} | {row["n_estimators"]} | {row["brier"]:.4f} | {row["log_loss"]:.4f} | {row["roc_auc"]:.4f} | {row["accuracy"]:.1%} |')
    lines.extend(["", f'Selected parameters: `{json.dumps(result["selected_parameters"], sort_keys=True)}`.', "",
                  "## Held-out test results", "", "| Model | Brier ↓ | Log loss ↓ | ROC AUC ↑ | Accuracy ↑ |", "| --- | ---: | ---: | ---: | ---: |"])
    for name, row in result["test"].items():
        lines.append(f'| {name} | {row["brier"]:.4f} | {row["log_loss"]:.4f} | {row["roc_auc"]:.4f} | {row["accuracy"]:.1%} |')
    ci = result["uncertainty"]
    lo, hi = ci["brier_intervals"]["gradient_boosting"]
    delta_lo, delta_hi = ci["gb_minus_home_rate_interval"]
    lines.extend(["", f'Gradient-boosting Brier 95% interval: **[{lo:.4f}, {hi:.4f}]**. Paired Brier difference (boosting minus historical rate): **[{delta_lo:.4f}, {delta_hi:.4f}]**; negative favors boosting.',
                  f'Intervals use {ci["replicates"]:,} paired bootstrap resamples of {ci["days"]} ET game dates. {ci["limitations"]}.', "",
                  f'Test home-win frequency: {gb["observed_home_win_rate"]:.1%}. Historical-rate prediction: {reference["mean_home_probability"]:.1%}. Accuracy uses a fixed 0.5 threshold; ties predict home.', "",
                  "![Brier comparison and calibration](baseline.png)", "",
                  "## Limits and next use", "",
                  "The five-game warmup excludes season openers, and coverage ends April 10 rather than including every regular-season game. Scheduled starts were retrieved retrospectively; rescheduling and exact historical result-availability times are not independently archived. Previous-date results are used conservatively, but unusual delayed completion remains a limitation.", "",
                  "The test season has now been examined for this baseline. Future model comparisons should preserve this protocol and disclose repeated evaluation; choose a later untouched period after the eventual language model's training cutoff for the final LLM benchmark. Narrative/injury experiments must compare all models on exactly the same eligible games, not compare their small text sample directly with this larger statistics cohort.", "",
                  "## Reproduce", "", "```bash", "uv pip install --python .venv/bin/python -r requirements-baseline.txt",
                  ".venv/bin/python scripts/run_gradient_boosting.py", ".venv/bin/python scripts/validate_gradient_boosting.py",
                  ".venv/bin/python -m unittest discover -s tests -v", "```", "",
                  "Model, separated inputs/labels, split membership and validation/test predictions are under `data/processed/gradient_boosting/` (git-ignored). Configuration, source/code hashes, library versions and full-precision metrics are in `metrics.json`; calibration bins are in `calibration.csv`.", "",
                  "Implementation references: [scikit-learn gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [binary Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html), and [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).", ""])
    (REPORT_DIR / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    if not config["refit_train_and_validation"]:
        raise ValueError("This protocol requires refitting on training plus validation")
    history = load_verified_history(config)
    inputs, labels, excluded = prepare_examples(history, config)
    x, y = arrays(inputs, labels)
    masks = {name: np.array([row["split"] == name for row in inputs]) for name in config["split_seasons"]}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Each chronological split must contain eligible games")
    chosen, selection = select_candidate(x[masks["train"]], y[masks["train"]], x[masks["validation"]], y[masks["validation"]], config)
    print("Validation selection:", json.dumps(selection), flush=True)
    # Freeze the selection before fitting the final model or evaluating test labels.
    fitted = masks["train"] | masks["validation"]
    model = new_model(config, chosen).fit(x[fitted], y[fitted])
    predictions = {"gradient_boosting": model.predict_proba(x[masks["test"]])[:, 1],
                   "historical_home_rate": np.full(masks["test"].sum(), y[fitted].mean()),
                   "constant_50_percent": np.full(masks["test"].sum(), 0.5)}
    test_inputs = [row for row in inputs if row["split"] == "test"]
    test_y = y[masks["test"]]
    result = {"config": config, "feature_names": list(FEATURE_NAMES), "history_games": len(history),
              "excluded_games": len(excluded), "selected_parameters": {**config["model_parameters"], **chosen, "random_state": config["seed"]},
              "validation_candidates": selection, "splits": {},
              "test": {name: score_predictions(test_y, p) for name, p in predictions.items()},
              "uncertainty": bootstrap_brier(test_y, predictions, [row["game_date_et"] for row in test_inputs], config["bootstrap_replicates"], config["seed"]),
              "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib")},
              "python_version": platform.python_version(),
              "source_sha256": {name: sha256(ROOT / name) for name in (config["history_path"], config["history_config"], "config/gradient_boosting.json", "scripts/collect_pilot.py", "scripts/run_gradient_boosting.py")}}
    for name, mask in masks.items():
        subset = [row for row in inputs if row["split"] == name]
        result["splits"][name] = {"season": config["split_seasons"][name], "games": len(subset),
                                  "first_date": subset[0]["game_date_et"], "last_date": subset[-1]["game_date_et"]}
    write_jsonl(DATA_DIR / "inputs.jsonl", inputs)
    write_jsonl(DATA_DIR / "labels.jsonl", labels)
    write_jsonl(DATA_DIR / "excluded.jsonl", excluded)
    write_csv(DATA_DIR / "split_manifest.csv", [{k: row[k] for k in ("game_id", "season", "split", "game_date_et", "cutoff_utc")} for row in inputs])
    write_csv(DATA_DIR / "test_predictions.csv", [{"game_id": row["game_id"], "game_date_et": row["game_date_et"], "home_win": int(test_y[i]),
                                                  **{name: float(p[i]) for name, p in predictions.items()}} for i, row in enumerate(test_inputs)])
    validation_model = new_model(config, chosen).fit(x[masks["train"]], y[masks["train"]])
    val_probabilities = validation_model.predict_proba(x[masks["validation"]])[:, 1]
    validation_inputs = [row for row in inputs if row["split"] == "validation"]
    write_csv(DATA_DIR / "validation_predictions.csv", [{"game_id": row["game_id"], "home_win": int(y[masks["validation"]][i]),
                                                        "gradient_boosting": float(val_probabilities[i]), "historical_home_rate": float(y[masks["train"]].mean())}
                                                       for i, row in enumerate(validation_inputs)])
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                 "fit_game_ids": [row["game_id"] for row in inputs if row["split"] != "test"],
                 "selection_game_ids": [row["game_id"] for row in validation_inputs]}, DATA_DIR / "model.joblib")
    result["artifact_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(DATA_DIR.glob("*.jsonl"))}
    write_json(REPORT_DIR / "metrics.json", result)
    calibration = calibration_rows(test_y, predictions["gradient_boosting"], config["calibration_bins"])
    write_csv(REPORT_DIR / "calibration.csv", calibration)
    write_plot(result, calibration)
    write_report(result)
    print(json.dumps({"splits": result["splits"], "test": result["test"], "uncertainty": result["uncertainty"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
