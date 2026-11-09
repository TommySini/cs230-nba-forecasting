"""Gradient-boosting statistics baseline: fit on TRAIN, choose on DEV, score TEST once.

Features are the 18 basic team features (collect_pilot.prior_features) plus team and
player box-score windows (box_score_features), all from earlier ET dates of the same
season. Each model imputes missing box-score statistics with medians fitted on its own
training rows. A fixed candidate grid is fitted on TRAIN and chosen by DEV Brier; DEV and
TEST forecasts both come from that TRAIN fit. Each TRAIN season after the first is
forecast by the selected settings fitted on the earlier TRAIN seasons only, so every
p_boost (which BAM uses as an input) is fitted on strictly earlier games; only the choice
among the fixed settings used DEV. Splits are defined once, in config/gradient_boosting.json.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import timedelta
import hashlib
from importlib.metadata import version
import io
import json
import math
import platform
import time

import joblib
from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from box_score_features import AVAILABILITY_LIMITATION, BOX_SCORE_FEATURE_NAMES, BoxScoreIndex
from collect_pilot import ABBR, ROOT, Cache, collect_games, iso, prior_features, stamp, write_json, write_jsonl
from collect_rich_stats import NBA_FRANCHISES

CONFIG_PATH = ROOT / "config/gradient_boosting.json"
DATA_DIR = ROOT / "data/processed/gradient_boosting"
METRICS_PATH = ROOT / "results/gradient_boosting.json"
SPLITS = ("train", "dev", "test")
SIDES = ("home", "away")
BASE_FEATURES = ("previous_games", "season_win_rate", "last_5_win_rate",
                 "last_5_point_differential", "days_since_previous_game", "back_to_back")
BASIC_FEATURE_NAMES = tuple(f"{side}_{name}" for side in ("home", "away", "difference") for name in BASE_FEATURES)
FEATURE_NAMES = BASIC_FEATURE_NAMES + BOX_SCORE_FEATURE_NAMES
CODE = ("scripts/run_gradient_boosting.py", "scripts/box_score_features.py", "scripts/collect_pilot.py")
DATA_FILES = ("inputs.jsonl", "labels.jsonl", "excluded.jsonl", "predictions.jsonl", "model.joblib")


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


def season_label(season):
    return f"{season}–{str(season + 1)[-2:]}"


def season_splits(config):
    """Season start year -> split, from inclusive [first, last] ranges."""
    bounds = [config["split_seasons"][name] for name in SPLITS]
    if any(first > last for first, last in bounds) or any(a[1] >= b[0] for a, b in zip(bounds, bounds[1:])):
        raise ValueError("TRAIN, DEV and TEST seasons must be non-empty, disjoint and chronological")
    return {season: name for name, (first, last) in zip(SPLITS, bounds) for season in range(first, last + 1)}


def franchise_game(game):
    return all(ABBR.get(game[f"{side}_abbr"], game[f"{side}_abbr"]) in NBA_FRANCHISES for side in SIDES)


def load_verified_history(config):
    """Every configured schedule, each equal to its checksum-verified raw scoreboard cache.

    ESPN labels All-Star exhibitions as regular season; a saved history may already omit
    them (history_2012 does). They are dropped here in any case.
    """
    history = []
    for source in config["histories"]:
        rows = read_rows(ROOT / source["path"])
        with redirect_stdout(io.StringIO()):
            verified = collect_games(Cache(offline=True), json.loads((ROOT / source["config"]).read_text()))
        if rows != verified and rows != [g for g in verified if franchise_game(g)]:
            raise ValueError(f"{source['path']} does not match its checksum-verified schedule cache")
        history += rows
    if len({game["game_id"] for game in history}) != len(history):
        raise ValueError("Duplicate historical game IDs")
    return [game for game in history if franchise_game(game)]


def load_box_scores(config):
    directory = ROOT / config["box_scores"]
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "games.jsonl") != manifest["artifact_sha256"]["games.jsonl"]:
        raise ValueError("Box-score records differ from their manifest")
    return read_rows(directory / "games.jsonl"), manifest


def prepare_examples(history, box_records, config):
    """Prior-only feature rows, separate outcome labels, and exclusions with reasons."""
    splits = season_splits(config)
    index = BoxScoreIndex(history, box_records, min_previous_games=config["min_previous_games"],
                          min_box_score_games=config["min_box_score_games"], cutoff_minutes=config["cutoff_minutes"])
    history = sorted(history, key=lambda g: (g["scheduled_start_utc"], g["game_id"]))
    by_season = {}
    for game in history:
        if game["season"] not in splits:
            raise ValueError(f"Unassigned season: {game['season']}")
        by_season.setdefault(game["season"], []).append(game)
    inputs, labels, excluded = [], [], []
    for game in history:
        # prior_features keeps same-season games on earlier ET dates; one season suffices.
        basic = {side: prior_features(game, by_season[game["season"]], side) for side in SIDES}
        box = index.for_game(game)
        if any(basic[side]["previous_games"] != box["coverage"][side]["previous_games"] for side in SIDES):
            raise ValueError(f"Basic and box-score windows disagree: {game['game_id']}")
        record = {key: game[key] for key in ("game_id", "season", "scheduled_start_utc", "game_date_et", "home_abbr", "away_abbr")}
        record["split"] = splits[game["season"]]
        if not box["eligible"]:
            excluded.append({**{key: record[key] for key in ("game_id", "season", "split", "game_date_et")},
                             "reason": box["reason"], "details": box["reasons"], "coverage": box["coverage"]})
            continue
        values = {f"{side}_{name}": float(basic[side][name]) for side in SIDES for name in BASE_FEATURES}
        values.update({f"difference_{name}": values[f"home_{name}"] - values[f"away_{name}"] for name in BASE_FEATURES})
        values.update(box["features"])
        record.update(cutoff_utc=iso(stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"])),
                      features={name: None if math.isnan(values[name]) else float(values[name]) for name in FEATURE_NAMES},
                      prior_game_ids={side: basic[side]["prior_game_ids"] for side in SIDES},
                      missing_box_score_game_ids={side: [m["game_id"] for m in box["missing_box_scores"][side]] for side in SIDES},
                      box_score_coverage=box["coverage"])
        inputs.append(record)
        labels.append({"game_id": game["game_id"], "home_win": int(game["home_score"] > game["away_score"])})
    return inputs, labels, excluded


def arrays(inputs, labels):
    if [r["game_id"] for r in inputs] != [r["game_id"] for r in labels]:
        raise ValueError("Inputs and outcome labels must align by game ID")
    x = np.array([[r["features"][name] for name in FEATURE_NAMES] for r in inputs], dtype=float)  # None -> NaN
    return x, np.array([r["home_win"] for r in labels], dtype=int)


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
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("model", GradientBoostingClassifier(**config["model_parameters"], **candidate,
                                                          random_state=config["seed"]))])


def fit(config, candidate, x, y):
    return new_model(config, candidate).fit(x, y)


def select_and_fit(x, y, splits, seasons, config, jobs=-1):
    """Fit each candidate on TRAIN, choose by DEV Brier, then refit on the TRAIN seasons before each later TRAIN season.

    There is deliberately no TEST argument: TEST rows never reach fitting or selection.
    """
    if config["selection_metric"] != "brier" or config["model_parameters"]["n_iter_no_change"] is not None:
        raise ValueError("The protocol requires DEV Brier selection and no internal early stopping")
    train, dev = splits == "train", splits == "dev"
    if not train.any() or not dev.any():
        raise ValueError("TRAIN and DEV need eligible games")
    parallel = Parallel(n_jobs=jobs)
    models = parallel(delayed(fit)(config, candidate, x[train], y[train]) for candidate in config["candidates"])
    trials = [{"candidate_index": i, **candidate, **score_predictions(y[dev], model.predict_proba(x[dev])[:, 1])}
              for i, (candidate, model) in enumerate(zip(config["candidates"], models))]
    best = min(trials, key=lambda r: (r["brier"], r["candidate_index"]))["candidate_index"]
    later = sorted(set(seasons[train].tolist()))[1:]
    prior = parallel(delayed(fit)(config, config["candidates"][best], x[train & (seasons < s)], y[train & (seasons < s)])
                     for s in later)
    return best, trials, models[best], dict(zip(later, prior))


def predict(final, prior, x, splits, seasons):
    """DEV/TEST from the TRAIN fit; each later TRAIN season from the model fitted on earlier TRAIN seasons."""
    train = splits == "train"
    p, produced = np.full(len(x), np.nan), np.full(len(x), None, dtype=object)
    p[~train], produced[~train] = final.predict_proba(x[~train])[:, 1], "train_fit"
    for season, model in prior.items():
        mask = train & (seasons == season)
        p[mask], produced[mask] = model.predict_proba(x[mask])[:, 1], "prior_seasons"
    return p, produced


def evaluate(p, y, splits, seasons, dates, config):
    """DEV/TEST scores against the TRAIN home-win rate, and prior-season TRAIN forecast scores by season."""
    train = splits == "train"
    home_rate = float(y[train].mean())
    evaluation = {}
    for name in ("dev", "test"):
        mask = splits == name
        compared = {"gradient_boosting": p[mask], "historical_home_rate": np.full(mask.sum(), home_rate),
                    "constant_50_percent": np.full(mask.sum(), 0.5)}
        evaluation[name] = {"metrics": {k: score_predictions(y[mask], v) for k, v in compared.items()},
                            "uncertainty": bootstrap_brier(y[mask], compared, dates[mask], config["bootstrap_replicates"],
                                                           config["seed"], "historical_home_rate")}
    scored = train & np.isfinite(p)
    by_season = {str(s): score_predictions(y[scored & (seasons == s)], p[scored & (seasons == s)])
                 for s in sorted(set(seasons[scored].tolist()))}
    if by_season:
        by_season["all"] = score_predictions(y[scored], p[scored])
    return home_rate, evaluation, by_season


def bootstrap_brier(y, predictions, dates, repeats, seed, reference):
    """Paired ET-calendar-day cluster bootstrap of Brier, conditional on the fitted models."""
    unique, inverse = np.unique(dates, return_inverse=True)
    counts = np.bincount(inverse)
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(repeats, len(unique)))
    denominators = counts[sampled].sum(axis=1)
    draws = {name: np.bincount(inverse, weights=(np.asarray(p) - y) ** 2)[sampled].sum(axis=1) / denominators
             for name, p in predictions.items()}
    interval = lambda values: np.quantile(values, [0.025, 0.975]).tolist()
    return {"method": "paired ET-calendar-day cluster bootstrap, percentile 95% interval",
            "replicates": repeats, "days": len(unique), "seed": seed, "reference": reference,
            "brier_intervals": {name: interval(v) for name, v in draws.items()},
            "difference_vs_reference": {name: interval(v - draws[reference]) for name, v in draws.items() if name != reference},
            "limitations": "Conditional on the fitted models; no training uncertainty or dependence across days"}


def calibration_rows(y, probability, bins):
    bucket = np.minimum((probability * bins).astype(int), bins - 1)
    return [{"bin_low": i / bins, "bin_high": (i + 1) / bins, "games": int(np.sum(bucket == i)),
             "mean_probability": float(probability[bucket == i].mean()),
             "observed_home_win_rate": float(y[bucket == i].mean())}
            for i in range(bins) if np.any(bucket == i)]


def coverage_by_season(history, box_records, inputs, excluded, config):
    splits = season_splits(config)
    boxes = Counter(r["season"] for r in box_records)
    result = {}
    for season in sorted({g["season"] for g in history}):
        rows = [r for r in inputs if r["season"] == season]
        gap = lambda key: sum(any(r["box_score_coverage"][s][key] < r["box_score_coverage"][s]["previous_games"]
                                  for s in SIDES) for r in rows)
        result[str(season)] = {"split": splits[season], "games": sum(g["season"] == season for g in history),
                               "box_scores": boxes.get(season, 0), "eligible": len(rows),
                               "excluded": dict(Counter(r["reason"] for r in excluded if r["season"] == season)),
                               "eligible_with_team_box_gaps": gap("valid_box_scores"),
                               "eligible_with_player_box_gaps": gap("valid_player_box_scores")}
    return result


def run(config, jobs):
    started = time.time()
    history = load_verified_history(config)
    box_records, box_manifest = load_box_scores(config)
    inputs, labels, excluded = prepare_examples(history, box_records, config)
    print(f"Prepared {len(inputs)} eligible games ({len(excluded)} excluded) in {time.time() - started:.0f}s", flush=True)
    x, y = arrays(inputs, labels)
    splits = np.array([r["split"] for r in inputs])
    seasons = np.array([r["season"] for r in inputs])
    if not (splits == "test").any():
        raise ValueError("TEST needs eligible games")
    best, trials, final, prior = select_and_fit(x, y, splits, seasons, config, jobs)
    candidate = config["candidates"][best]
    print(f"DEV-selected candidate {best}: {candidate}", flush=True)
    # TEST is scored once, by the DEV-selected TRAIN model.
    train = splits == "train"
    p, produced = predict(final, prior, x, splits, seasons)
    ids = np.array([r["game_id"] for r in inputs])
    predictions = [{"game_id": r["game_id"], "season": r["season"], "game_date_et": r["game_date_et"], "split": r["split"],
                    "p_boost": float(p[i]), "produced_by": produced[i]}
                   for i, r in enumerate(inputs) if produced[i] is not None]
    settings = {**config["model_parameters"], **candidate, "random_state": config["seed"]}
    bundle = {"feature_names": list(FEATURE_NAMES), "settings": settings, "candidate_index": best,
              "final": {"model": final, "fit_game_ids": ids[train].tolist(), "selection_game_ids": ids[splits == "dev"].tolist()},
              "prior_seasons": {season: {"model": model, "fit_game_ids": ids[train & (seasons < season)].tolist()}
                                for season, model in prior.items()}}
    for name, rows in (("inputs.jsonl", inputs), ("labels.jsonl", labels), ("excluded.jsonl", excluded),
                       ("predictions.jsonl", predictions)):
        write_jsonl(DATA_DIR / name, rows)
    joblib.dump(bundle, DATA_DIR / "model.joblib")
    dates = np.array([r["game_date_et"] for r in inputs])
    home_rate, evaluation, train_metrics = evaluate(p, y, splits, seasons, dates, config)
    test = splits == "test"
    box_dir = ROOT / config["box_scores"]
    sources = [CONFIG_PATH, box_dir / "manifest.json", box_dir / "games.jsonl", *(ROOT / name for name in CODE)]
    sources += [ROOT / s[key] for s in config["histories"] for key in ("path", "config")]
    result = {"config": config, "feature_names": list(FEATURE_NAMES), "history_games": len(history),
              "eligible_games": len(inputs), "excluded_games": len(excluded),
              "split_games": {name: int((splits == name).sum()) for name in SPLITS},
              "coverage_by_season": coverage_by_season(history, box_records, inputs, excluded, config),
              "box_score_source": {"path": config["box_scores"], **{k: box_manifest.get(k) for k in (
                  "provider_counts", "season_counts", "complete_team_games", "complete_player_games", "failed_games")}},
              "dev_candidates": trials, "selected_candidate_index": best, "selected_parameters": settings,
              "train_home_win_rate": home_rate, "evaluation": evaluation, "train_forecasts": train_metrics,
              "train_games_without_forecast": int((train & ~np.isfinite(p)).sum()),
              "test_calibration": calibration_rows(y[test], p[test], config["calibration_bins"]),
              "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")},
              "python_version": platform.python_version(), "runtime_seconds": round(time.time() - started, 1),
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
              "artifact_sha256": {f"data/processed/gradient_boosting/{name}": sha256(DATA_DIR / name) for name in DATA_FILES}}
    write_json(METRICS_PATH, result)
    print(json.dumps({name: {k: round(v["brier"], 4) for k, v in e["metrics"].items()} for name, e in evaluation.items()}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=-1, help="parallel model fits (joblib n_jobs)")
    args = parser.parse_args()
    run(json.loads(CONFIG_PATH.read_text()), args.jobs)


if __name__ == "__main__":
    main()
