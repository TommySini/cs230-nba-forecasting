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
DATA_DIR = ROOT / "data/processed"
MODEL_PATH = ROOT / "data/models/gradient_boosting/model.joblib"
REPORT_DIR = ROOT / "reports/gradient_boosting"
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
    from nba_dataset import Dataset, remap_paths
    dataset = Dataset(ROOT, ROOT / config.get("dataset_dir", "data/processed"))
    history = []
    for config_name in config["history_configs"]:
        source_config = json.loads((ROOT / config_name).read_text())
        seasons = [s["start_year"] for s in source_config["seasons"]]
        rows = dataset.history(seasons=seasons)
        with redirect_stdout(io.StringIO()):
            verified = collect_games(Cache(offline=True), source_config)
        if sorted(rows, key=lambda r: r["game_id"]) != sorted(
                [remap_paths(g) for g in verified if franchise_game(g)], key=lambda r: r["game_id"]):
            raise ValueError(f"Shared games do not match the checksum-verified schedule cache: {config_name}")
        history += rows
    if len({game["game_id"] for game in history}) != len(history):
        raise ValueError("Duplicate historical game IDs")
    return [game for game in history if franchise_game(game)]


def load_box_scores(config):
    from nba_dataset import Dataset
    dataset = Dataset(ROOT, ROOT / config.get("dataset_dir", "data/processed"))
    return dataset.box_scores(), dataset.manifest["legacy_records"]["data/processed/box_scores/manifest.json"]


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
    from nba_dataset import Dataset, store_predictions
    if (ROOT / config["dataset_dir"]).resolve() != (ROOT / "data/processed").resolve():
        raise ValueError("Training writes to the published shared dataset; staged datasets are validation-only")
    if MODEL_PATH.exists() or (REPORT_DIR / "metrics.json").exists():
        raise ValueError("A completed boosting run already exists; validate it instead of overwriting it")
    started = time.time()
    dataset = Dataset(ROOT, ROOT / config["dataset_dir"])
    dataset.verify()
    history = dataset.history()
    box_records, box_manifest = load_box_scores(config)
    inputs = dataset.inputs()
    labels = dataset.labels(SPLITS)
    excluded = dataset.excluded()
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
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    store_predictions(ROOT, [{**r, "p_home_win": r["p_boost"]} for r in predictions],
                      model_id="gradient_boosting", stage="baseline")
    dates = np.array([r["game_date_et"] for r in inputs])
    home_rate, evaluation, train_metrics = evaluate(p, y, splits, seasons, dates, config)
    test = splits == "test"
    sources = [CONFIG_PATH, *(ROOT / name for name in CODE)]
    result = {"config": config, "feature_names": list(FEATURE_NAMES), "history_games": len(history),
              "eligible_games": len(inputs), "excluded_games": len(excluded),
              "split_games": {name: int((splits == name).sum()) for name in SPLITS},
              "coverage_by_season": coverage_by_season(history, box_records, inputs, excluded, config),
              "dataset_id": dataset.manifest["dataset_id"],
              "box_score_source": {"path": config["dataset_dir"], **{k: box_manifest.get(k) for k in (
                  "provider_counts", "season_counts", "complete_team_games", "complete_player_games", "failed_games")}},
              "dev_candidates": trials, "selected_candidate_index": best, "selected_parameters": settings,
              "train_home_win_rate": home_rate, "evaluation": evaluation, "train_forecasts": train_metrics,
              "train_games_without_forecast": int((train & ~np.isfinite(p)).sum()),
              "test_calibration": calibration_rows(y[test], p[test], config["calibration_bins"]),
              "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")},
              "python_version": platform.python_version(), "runtime_seconds": round(time.time() - started, 1),
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
              "artifact_sha256": {str(MODEL_PATH.relative_to(ROOT)): sha256(MODEL_PATH)}}
    write_json(REPORT_DIR / "metrics.json", result)
    write_report(result)
    print(json.dumps({name: {k: round(v["brier"], 4) for k, v in e["metrics"].items()} for name, e in evaluation.items()}), flush=True)
    return result


def write_report(result):
    config, evaluation = result["config"], result["evaluation"]
    first, last = config["split_seasons"]["train"]
    dev, test = (season_label(config["split_seasons"][name][0]) for name in ("dev", "test"))
    gb, ref = evaluation["test"]["metrics"]["gradient_boosting"], evaluation["test"]["metrics"]["historical_home_rate"]
    selected = result["dev_candidates"][result["selected_candidate_index"]]
    coverage = result["coverage_by_season"]
    lacking = [season_label(int(s)) for s, row in coverage.items() if row["split"] == "train" and not row["eligible"]]
    lines = ["# Gradient-boosting statistics baseline", "",
             f"TEST {test}: Brier **{gb['brier']:.4f}**, log loss {gb['log_loss']:.4f}, accuracy {gb['accuracy']:.1%} "
             f"on {gb['games']:,} games, versus {ref['brier']:.4f} for the TRAIN home-win rate. "
             f"Settings chosen on DEV {dev}: depth {selected['max_depth']}, {selected['n_estimators']} trees.", "",
             "## Protocol", "",
             f"- Splits by season: TRAIN {season_label(first)} to {season_label(last)}, DEV {dev}, TEST {test} "
             "(`config/gradient_boosting.json`). Regular-season games from checksum-verified ESPN schedules.",
             f"- Features ({len(result['feature_names'])}): 18 basic team features (games played, season and last-5 win rate, "
             "last-5 point differential, rest, back-to-back) plus team and player box-score windows (season, last 5, last 10) "
             "with team-coverage indicators; home, away and home-minus-away. All use earlier ET dates of the same season only.",
             f"- Eligibility: both teams have at least {config['min_previous_games']} earlier games and "
             f"{config['min_box_score_games']} valid earlier box scores. Other missing box scores reduce coverage: team "
             "statistics pool the window's valid games, coverage features record the gap, and player statistics are missing "
             "unless the whole window has player box scores.",
             "- Models: gradient boosting with median imputation fitted on each model's own training rows. "
             f"{len(config['candidates'])} fixed candidates are fitted on TRAIN and chosen by DEV Brier. DEV and TEST forecasts both "
             "come from that TRAIN fit; TEST was scored once, after selection. No early stopping, calibration or TEST-driven tuning.",
             "- TRAIN forecasts (`produced_by: prior_seasons`): each TRAIN season after the first is predicted by the selected "
             "settings fitted on the earlier TRAIN seasons only, so every forecast BAM receives is fitted on strictly earlier "
             f"games (the choice among the {len(config['candidates'])} settings used DEV). "
             f"The first TRAIN season ({season_label(first)}) has none.", "",
             "## DEV selection", "", "| Depth | Trees | DEV Brier | Log loss | Accuracy |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in result["dev_candidates"]:
        lines.append(f"| {row['max_depth']} | {row['n_estimators']} | {row['brier']:.4f} | {row['log_loss']:.4f} | {row['accuracy']:.1%} |")
    lines += ["", "## Results", "", "| Split | Forecast | Games | Brier | 95% interval | Log loss | Accuracy |",
              "| --- | --- | ---: | ---: | --- | ---: | ---: |"]
    for name in ("dev", "test"):
        u = evaluation[name]["uncertainty"]
        for model, m in evaluation[name]["metrics"].items():
            lo, hi = u["brier_intervals"][model]
            lines.append(f"| {name.upper()} | {model} | {m['games']:,} | {m['brier']:.4f} | [{lo:.4f}, {hi:.4f}] | {m['log_loss']:.4f} | {m['accuracy']:.1%} |")
    lines.append("")
    for name in ("dev", "test"):
        lo, hi = evaluation[name]["uncertainty"]["difference_vs_reference"]["gradient_boosting"]
        lines.append(f"{name.upper()} paired Brier difference, boosting minus home rate: [{lo:+.4f}, {hi:+.4f}] "
                     f"({evaluation[name]['uncertainty']['days']} ET dates resampled).")
    lines += ["", "## TRAIN forecasts from earlier seasons", ""]
    if result["train_forecasts"]:
        lines += ["| Season | Games | Brier | Log loss | Accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
        lines += [f"| {season if season == 'all' else season_label(int(season))} | {m['games']:,} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['accuracy']:.1%} |"
                  for season, m in result["train_forecasts"].items()]
        lines.append("")
    lines.append(f"{result['train_games_without_forecast']:,} eligible TRAIN games have no earlier TRAIN season and no forecast.")
    lines += ["", "## Coverage by season", "",
              f"Box scores: `{result['box_score_source']['path']}`."
              + (f" TRAIN seasons without eligible games: {', '.join(lacking)}." if lacking else ""), "",
              "| Season | Split | Games | Box records | Eligible | Excluded: warmup | Excluded: box coverage | With team-box gaps | With player-box gaps |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for season, row in coverage.items():
        ex = row["excluded"]
        lines.append(f"| {season_label(int(season))} | {row['split']} | {row['games']:,} | {row['box_scores']:,} | {row['eligible']:,} | "
                     f"{ex.get('fewer_than_min_previous_games', 0):,} | {ex.get('fewer_than_min_box_scores', 0) + ex.get('invalid_window', 0):,} | "
                     f"{row['eligible_with_team_box_gaps']:,} | {row['eligible_with_player_box_gaps']:,} |")
    lines += ["", "## Limits", "",
              f"- {AVAILABILITY_LIMITATION} Possessions and ratings are box-score estimates, not official NBA figures.",
              "- No injury, lineup, odds or text inputs. Prior-game workload is not current-game availability.",
              f"- The {test} TEST season was examined in earlier experiments, so this is an exploratory backtest. The previous protocol "
              "(TRAIN 2023–24 only, final refit on 2023–25) scored 0.2099 (basic 18 features) and 0.2059 (rich features) on it: "
              "[basic](../past/basic_boosting_2023/findings.md), [rich](../past/rich_boosting_2023/findings.md).",
              "- Bootstrap intervals are conditional on the fitted models.", "",
              "## Reproduce", "", "```bash", ".venv/bin/python scripts/run_gradient_boosting.py",
              ".venv/bin/python scripts/validate_gradient_boosting.py", "```", "",
              f"Runtime {result['runtime_seconds']:.0f} s. Shared tables and forecasts are in `data/processed/`; "
              "the saved model is in `data/models/gradient_boosting/model.joblib`. "
              "Full metrics, hashes and versions are in `metrics.json`.", ""]
    (REPORT_DIR / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=-1, help="parallel model fits (joblib n_jobs)")
    args = parser.parse_args()
    run(json.loads(CONFIG_PATH.read_text()), args.jobs)


if __name__ == "__main__":
    main()
