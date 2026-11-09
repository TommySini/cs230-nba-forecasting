"""Score Polymarket cutoff prices against the gradient-boosting baseline on identical games."""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from collect_pilot import ROOT, write_json
from collect_polymarket import (CONFIG_PATH, REPORT_DIR, outcome_side, load_games,
                                load_market_records, load_market_audit)
from nba_dataset import Dataset
from run_gradient_boosting import calibration_rows, score_predictions, write_csv

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
            for row in Dataset(ROOT, ROOT / config["dataset_dir"]).predictions(comparison["boosting_model"])
            if row["split"] == SPLITS[name]]
    return rows, [(row, markets[row["game_id"]], games[row["game_id"]]) for row in rows if row["game_id"] in markets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    comparison = config["comparison"]
    games = {g["game_id"]: g for g in load_games(config)}
    markets = {r["game_id"]: r for r in load_market_records()}
    audit = {r["game_id"]: r for r in load_market_audit()}
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
    result["coverage"] = json.loads((REPORT_DIR / "coverage.json").read_text())
    calibration = calibration_rows(y_test, p_test["polymarket_cutoff"], comparison["calibration_bins"])
    write_csv(REPORT_DIR / "calibration.csv", calibration)
    write_csv(REPORT_DIR / "comparison_test.csv", [
        {"game_id": g["game_id"], "game_date_et": g["game_date_et"], "home_win": int(y_test[i]),
         **{k: float(p[i]) for k, p in p_test.items()}, "market_plus_boosting_stack": float(stacked[i])}
        for i, (_, _, g) in enumerate(joined_test)])
    write_json(REPORT_DIR / "metrics.json", result)
    write_report(result)
    print(json.dumps({name: {k: v["brier"] for k, v in split["metrics"].items()} for name, split in result["splits"].items()}, indent=2))


def fmt_interval(values, digits=4):
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def write_report(result):
    coverage = result["coverage"]
    lines = ["# Polymarket market baseline", "",
             "Market-implied home-win probability at the same prediction cutoff as the boosting baseline "
             "(scheduled tipoff minus 60 minutes), scored on identical games.", ""]
    test = result["splits"]["test"]
    if test["matched_games"]:
        m, g = test["metrics"]["polymarket_cutoff"], test["metrics"]["gradient_boosting"]
        diff = test["uncertainty"]["brier_differences"]["polymarket_cutoff_minus_gradient_boosting"]
        lines.extend([f'Held-out 2025–26 games with both forecasts: **{test["matched_games"]}** of {test["baseline_games"]}. '
                      f'Polymarket Brier **{m["brier"]:.4f}** (accuracy {m["accuracy"]:.1%}) versus boosting **{g["brier"]:.4f}** '
                      f'(accuracy {g["accuracy"]:.1%}). Paired Brier difference, market minus boosting: **{fmt_interval(diff)}**; negative favors the market.', ""])
    lines.extend(["## Coverage", "", f'{coverage["scope"]}.', "",
                  "| Season | Games | Cutoff price | No market | Ambiguous | Stale / no pregame price | Fetch failed |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for season, row in coverage["by_season"].items():
        d = row["decisions"]
        lines.append(f'| {season}–{str(int(season) + 1)[-2:]} | {row["games"]} | {d.get("matched_price_at_cutoff", 0)} | '
                     f'{d.get("no_matching_market", 0)} | {d.get("ambiguous_markets", 0)} | '
                     f'{d.get("stale_price", 0) + d.get("no_price_before_cutoff", 0)} | {d.get("price_fetch_failed", 0)} |')
    volume = coverage["matched_market_lifetime_volume_usd"]
    lines.extend(["", f'Median price point is {coverage["median_minutes_before_cutoff"]:.0f} minutes before the cutoff '
                  f'(maximum {coverage["max_minutes_before_cutoff"]:.0f}; limit {coverage["config"]["max_staleness_minutes"]}). '
                  f'Matched-market lifetime volume: median ${volume["median"]:,.0f}, 10th percentile ${volume["p10"]:,.0f}.', ""])
    for name in ("test", "validation"):
        split = result["splits"][name]
        if not split["matched_games"]:
            continue
        lines.extend([f'## {name.title()} season {split["season"]}–{str(split["season"] + 1)[-2:]} '
                      f'({split["matched_games"]} games, {split["first_date"]} to {split["last_date"]})', "",
                      "| Forecast | Brier ↓ | 95% interval | Log loss ↓ | ROC AUC ↑ | Accuracy ↑ |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for model, metrics in split["metrics"].items():
            ci = split["uncertainty"]["brier_intervals"][model]
            auc = f'{metrics["roc_auc"]:.4f}' if metrics["roc_auc"] is not None else "—"
            lines.append(f'| {model} | {metrics["brier"]:.4f} | {fmt_interval(ci)} | {metrics["log_loss"]:.4f} | {auc} | {metrics["accuracy"]:.1%} |')
        u = split["uncertainty"]
        lines.extend(["", "Paired differences (negative Brier / positive accuracy favors the first forecast):", ""])
        for key, ci in u["brier_differences"].items():
            acc = u["accuracy_differences"][key]
            lines.append(f'- `{key}`: Brier {fmt_interval(ci)}; accuracy [{acc[0]:+.1%}, {acc[1]:+.1%}]')
        liquid = split["liquid_subset"]
        if liquid["metrics"]:
            lines.extend(["", f'Markets with lifetime volume ≥ ${liquid["threshold_usd"]:,} ({liquid["games"]} games): '
                          f'Polymarket Brier {liquid["metrics"]["polymarket_cutoff"]["brier"]:.4f}, '
                          f'boosting {liquid["metrics"]["gradient_boosting"]["brier"]:.4f}.'])
        lines.append("")
    stack = result["stack"]
    stack_diff = stack["uncertainty"]["brier_differences"]["market_plus_boosting_stack_minus_polymarket_cutoff"]
    c = stack["coefficients"]
    lines.extend(["## Does boosting add information beyond the market?", "",
                  f'A two-feature logistic stack, logit(market) and logit(boosting), was fitted on the 2024–25 season and scored once on 2025–26. '
                  f'Coefficients: market {c["market_logit"]:.3f}, boosting {c["boosting_logit"]:.3f}, intercept {c["intercept"]:.3f}. '
                  f'Test Brier {stack["test_metrics"]["brier"]:.4f}; paired difference versus the market alone {fmt_interval(stack_diff)}. '
                  "An interval below zero would mean the boosting features carry information the market price lacks.", ""])
    consistency = result["resolution_consistency"]
    lines.extend(["## Protocol and limits", "",
                  "- Discovery: Polymarket Gamma `/markets`, NBA tag, `moneyline` type, closed markets, paged by end-date window. "
                  "Games match on both team names and a market start within 180 minutes of the ESPN scheduled start. "
                  "Primary `nba-{away}-{home}-{date}` slugs win over low-volume `-moneyline` duplicates; selection never uses volume or prices.",
                  "- Price: last CLOB price-history point (10-minute fidelity) for the home outcome token at or before the cutoff, at most 60 minutes old. "
                  "The request ends at scheduled tipoff, so no in-game prices are fetched. `polymarket_tipoff_reference` is the last pregame point and is later than the model cutoff; it is a reference, not a fair competitor.",
                  "- Prices come from exchange trading history retrieved retrospectively through Polymarket's API; the API's sampling is not independently archived. "
                  "Lifetime volume includes post-cutoff trading and is used only to describe liquidity.",
                  f'- Label consistency: market resolution agrees with the ESPN outcome for {consistency["agree"]} games, disagrees for {len(consistency["disagree"])}, unresolved {consistency["unresolved"]}.',
                  "- The 2025–26 test season was already examined for the boosting baseline; this is a repeated evaluation on that season. "
                  "Polymarket listed no per-game NBA markets in 2023–24, so the training season has no market comparison.",
                  "- Polymarket keeps price history for closed markets, so the same collector can score a later held-out period (for example 2026–27) after games finish.", "",
                  "## Reproduce", "", "```bash",
                  ".venv/bin/python scripts/nba_dataset.py validate",
                  ".venv/bin/python scripts/evaluate_polymarket.py",
                  ".venv/bin/python scripts/validate_polymarket.py",
                  ".venv/bin/python -m unittest tests.test_polymarket -v", "```", ""])
    (REPORT_DIR / "findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
