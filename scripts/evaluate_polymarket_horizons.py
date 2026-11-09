"""Score market prices eight hours and one hour before tipoff against boosting on identical games.

Only games whose price is supported by recent trading at every horizon are scored, so the
horizons differ only in timing. Boosting predictions are the pipeline's DEV (2024-25) and
TEST (2025-26) forecasts, both from the TRAIN-only fit.
"""
import argparse
import json

import numpy as np

from collect_pilot import ROOT, write_json
from collect_polymarket import CONFIG_PATH, DATA_DIR, read_rows
from evaluate_polymarket import SPLITS, joined_split, paired_bootstrap
from run_gradient_boosting import score_predictions


METRICS_PATH = ROOT / "results/polymarket_horizons.json"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    hours = config["horizons"]["hours_before_tipoff"]
    games = {g["game_id"]: g for g in read_rows(ROOT / config["history_path"])}
    prices = {}
    for row in read_rows(DATA_DIR / "horizon_prices.jsonl"):
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
    write_json(METRICS_PATH, result)
    print(json.dumps({s: {k: round(v["brier"], 4) for k, v in r["metrics"].items()} for s, r in result["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
