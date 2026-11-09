"""Score market prices eight hours and one hour before tipoff against boosting on identical games.

Only games whose price is supported by recent trading at every horizon are scored, so the
horizons differ only in timing. Boosting predictions are the pipeline's DEV (2024-25) and
TEST (2025-26) forecasts, both from the TRAIN-only fit.
"""
import argparse
import json

import numpy as np

from collect_pilot import ROOT, write_json
from collect_polymarket import CONFIG_PATH, REPORT_DIR, load_games, load_horizon_records
from evaluate_polymarket import SPLITS, joined_split, paired_bootstrap
from run_gradient_boosting import score_predictions


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    hours = config["horizons"]["hours_before_tipoff"]
    games = {g["game_id"]: g for g in load_games(config)}
    prices = {}
    for row in load_horizon_records():
        by_hour = {h["hours_before_tipoff"]: h for h in row["horizons"]}
        if all(by_hour[h]["usable"] for h in hours):
            prices[row["game_id"]] = {h: by_hour[h]["history_probability"] for h in hours}
    names = {h: f"market_{h}h_before" for h in hours}
    result = {"rules": config["horizons"], "splits": {}}
    for split in SPLITS:
        rows, joined = joined_split(split, config, prices, games)
        y = np.array([int(r["home_win"]) for r, _, _ in joined])
        predictions = {names[h]: np.array([p[h] for _, p, _ in joined]) for h in hours}
        predictions["gradient_boosting"] = np.array([float(r["gradient_boosting"]) for r, _, _ in joined])
        dates = [g["game_date_et"] for _, _, g in joined]
        early, late = names[max(hours)], names[min(hours)]
        result["splits"][split] = {
            "season": joined[0][2]["season"], "baseline_games": len(rows), "scored_games": len(joined),
            "first_date": min(dates), "last_date": max(dates),
            "metrics": {k: score_predictions(y, p) for k, p in predictions.items()},
            "uncertainty": paired_bootstrap(y, predictions, dates,
                                            [(early, "gradient_boosting"), (late, "gradient_boosting"), (early, late)],
                                            config["comparison"]["bootstrap_replicates"], config["comparison"]["seed"])}
    write_json(REPORT_DIR / "horizon_metrics.json", result)
    write_report(result, hours)
    print(json.dumps({s: {k: round(v["brier"], 4) for k, v in r["metrics"].items()} for s, r in result["splits"].items()}, indent=1))


def write_report(result, hours):
    rules = result["rules"]
    lines = ["# Market price at note-publication time versus one hour before tipoff", "",
             f"Game notes are typically published about {max(hours)} hours before tipoff, so the fair market reference "
             f"for a notes-based forecast is the price {max(hours)} hours before tipoff. Each price is the last CLOB "
             f"price-history point at or before tipoff minus the horizon (at most {rules['max_history_age_minutes']} minutes old). "
             f"It counts only when the preceding {rules['trade_window_hours']} hours hold at least {rules['min_window_trades']} trades "
             f"and ${rules['min_window_cash_usd']:,} traded, the last trade is at most {rules['max_last_trade_age_minutes']} minutes old, "
             f"and the price is within {rules['max_history_vwap_gap']} of the trades' volume-weighted price. "
             "Only games usable at every horizon are scored, against the existing out-of-sample boosting predictions.", ""]
    for split, r in result["splits"].items():
        u = r["uncertainty"]
        lines += [f"## {split.title()} season {r['season']}-{str(r['season'] + 1)[-2:]}: {r['scored_games']} of {r['baseline_games']} games", "",
                  "| Forecast | Brier ↓ | 95% interval | Log loss ↓ | Accuracy |", "| --- | ---: | --- | ---: | ---: |"]
        for name, m in r["metrics"].items():
            lo, hi = u["brier_intervals"][name]
            lines.append(f"| {name} | {m['brier']:.4f} | [{lo:.4f}, {hi:.4f}] | {m['log_loss']:.4f} | {m['accuracy']:.1%} |")
        lines += ["", "Paired Brier differences (negative favors the first):", ""]
        lines += [f"- `{pair}`: [{lo:+.4f}, {hi:+.4f}]" for pair, (lo, hi) in u["brier_differences"].items()]
        lines.append("")
    (REPORT_DIR / "horizon_findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
