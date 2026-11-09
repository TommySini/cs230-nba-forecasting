"""Score Polymarket cutoff prices against the gradient-boosting baseline on identical games."""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from collect_pilot import ROOT, write_json
from collect_polymarket import CONFIG_PATH, DATA_DIR, load_games, outcome_side, read_rows, summarize
from run_gradient_boosting import calibration_rows, score_predictions, write_csv

METRICS_PATH = ROOT / "results/polymarket.json"

SPLITS = {"validation": "dev", "test": "test"}  # Polymarket report name -> boosting split


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def resolved_side(row, game):
    """Winning side according to the market's final resolution, for label consistency only."""
    market = row.get("market") or {}
    try:
        prices = [float(p) for p in json.loads(market.get("outcome_prices") or "[]")]
    except ValueError:
        return None
    if market.get("resolution") != "resolved" or sorted(prices) != [0.0, 1.0]:
        return None
    return outcome_side(market["outcomes"][prices.index(1.0)], game)


def paired_bootstrap(y, predictions, dates, pairs, repeats, seed):
    # Same design as the boosting baseline: resample ET calendar days together.
    unique, inverse = np.unique(dates, return_inverse=True)
    counts = np.bincount(inverse)
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(repeats, len(unique)))
    denominators = counts[sampled].sum(axis=1)
    draw = lambda values: np.bincount(inverse, weights=values, minlength=len(unique))[sampled].sum(axis=1) / denominators
    brier = {name: draw((p - y) ** 2) for name, p in predictions.items()}
    accuracy = {name: draw(((p >= 0.5) == y).astype(float)) for name, p in predictions.items()}
    interval = lambda values: np.quantile(values, [0.025, 0.975]).tolist()
    return {"method": "paired ET-calendar-day cluster bootstrap, percentile 95% interval",
            "replicates": repeats, "days": len(unique), "seed": seed,
            "brier_intervals": {name: interval(values) for name, values in brier.items()},
            "brier_differences": {f"{a}_minus_{b}": interval(brier[a] - brier[b]) for a, b in pairs},
            "accuracy_differences": {f"{a}_minus_{b}": interval(accuracy[a] - accuracy[b]) for a, b in pairs}}


def joined_split(name, config, markets, games):
    """Boosting forecasts of one split (TRAIN-only fit), outcomes from history, joined to market rows."""
    comparison = config["comparison"]
    home_rate = json.loads((ROOT / comparison["boosting_metrics"]).read_text())["train_home_win_rate"]
    rows = [dict(row, gradient_boosting=row["p_boost"], historical_home_rate=home_rate,
                 home_win=int(games[row["game_id"]]["home_score"] > games[row["game_id"]]["away_score"]))
            for row in read_rows(ROOT / comparison["boosting_predictions"]) if row["split"] == SPLITS[name]]
    return rows, [(row, markets[row["game_id"]], games[row["game_id"]]) for row in rows if row["game_id"] in markets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    comparison = config["comparison"]
    games = {g["game_id"]: g for g in load_games(config)}
    markets = {r["game_id"]: r for r in read_rows(DATA_DIR / "market_probs.jsonl")}
    audit = {r["game_id"]: r for r in read_rows(DATA_DIR / "audit.jsonl")}
    if (DATA_DIR / "listing_failures.json").exists() or set(audit) != set(games):
        raise ValueError("Incomplete market collection; validate its saved records before evaluation")
    result = {"config": config, "splits": {}}
    arrays = {}
    for name in SPLITS:
        rows, joined = joined_split(name, config, markets, games)
        y = np.array([int(r["home_win"]) for r, _, _ in joined])
        predictions = {"polymarket_cutoff": np.array([m["home_probability"] for _, m, _ in joined]),
                       "gradient_boosting": np.array([float(r["gradient_boosting"]) for r, _, _ in joined]),
                       "historical_home_rate": np.array([float(r["historical_home_rate"]) for r, _, _ in joined]),
                       "polymarket_tipoff_reference": np.array([m["tipoff_home_probability"] for _, m, _ in joined])}
        dates = [g["game_date_et"] for _, _, g in joined]
        volumes = np.array([float(audit[g["game_id"]]["market"]["volume"] or 0) for _, _, g in joined])
        arrays[name] = (y, predictions, dates, joined)
        liquid = volumes >= comparison["liquidity_threshold_usd"]
        result["splits"][name] = {
            "season": joined[0][2]["season"] if joined else None, "baseline_games": len(rows), "matched_games": len(joined),
            "first_date": min(dates) if dates else None, "last_date": max(dates) if dates else None,
            "metrics": {k: score_predictions(y, p) for k, p in predictions.items()},
            "uncertainty": paired_bootstrap(y, predictions, dates,
                                            [("polymarket_cutoff", "gradient_boosting"), ("polymarket_cutoff", "historical_home_rate")],
                                            comparison["bootstrap_replicates"], comparison["seed"]),
            "liquid_subset": {"threshold_usd": comparison["liquidity_threshold_usd"], "games": int(liquid.sum()),
                              "metrics": {k: score_predictions(y[liquid], p[liquid]) for k, p in predictions.items()} if liquid.sum() > 1 else None}}
    # Does boosting add information the market lacks? Fit a two-feature stack on
    # the validation season only, then score it once on the test season.
    y_val, p_val, _, _ = arrays["validation"]
    y_test, p_test, dates_test, joined_test = arrays["test"]
    features = lambda p: np.column_stack((logit(p["polymarket_cutoff"]), logit(p["gradient_boosting"])))
    stack = LogisticRegression(C=1e6, max_iter=1000).fit(features(p_val), y_val)
    stacked = stack.predict_proba(features(p_test))[:, 1]
    test_with_stack = {"polymarket_cutoff": p_test["polymarket_cutoff"], "market_plus_boosting_stack": stacked}
    result["stack"] = {"fit_on": "validation season (boosting model trained on TRAIN seasons only)",
                       "coefficients": {"intercept": float(stack.intercept_[0]), "market_logit": float(stack.coef_[0][0]),
                                        "boosting_logit": float(stack.coef_[0][1])},
                       "test_metrics": score_predictions(y_test, stacked),
                       "uncertainty": paired_bootstrap(y_test, test_with_stack, dates_test,
                                                       [("market_plus_boosting_stack", "polymarket_cutoff")],
                                                       comparison["bootstrap_replicates"], comparison["seed"])}
    consistency = {"agree": 0, "disagree": [], "unresolved": 0}
    for game_id, market in markets.items():
        game = games[game_id]
        side = resolved_side(audit[game_id], game)
        if side is None:
            consistency["unresolved"] += 1
        elif (side == "home") == (game["home_score"] > game["away_score"]):
            consistency["agree"] += 1
        else:
            consistency["disagree"].append(game_id)
    result["resolution_consistency"] = consistency
    result["coverage"] = summarize(list(markets.values()), list(audit.values()), list(games.values()), [], config)
    result["test_calibration"] = calibration_rows(y_test, p_test["polymarket_cutoff"], comparison["calibration_bins"])
    write_csv(DATA_DIR / "comparison_test.csv", [
        {"game_id": g["game_id"], "game_date_et": g["game_date_et"], "home_win": int(y_test[i]),
         **{k: float(p[i]) for k, p in p_test.items()}, "market_plus_boosting_stack": float(stacked[i])}
        for i, (_, _, g) in enumerate(joined_test)])
    write_json(METRICS_PATH, result)
    print(json.dumps({name: {k: v["brier"] for k, v in split["metrics"].items()} for name, split in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
