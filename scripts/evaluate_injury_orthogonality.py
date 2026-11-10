"""Is injury-report text orthogonal to, and informative beyond, the boosting forecast?

Pre-specified feature (no tuning): injury burden per team = 1 per player listed
Out with an Injury/Illness reason + 0.5 per Doubtful/Questionable. Two-way and
G League assignments are excluded. feature = away burden - home burden
(positive favours the home team). The "text forecast" shifts boosting's log-odds
by SCALE * feature; SCALE does not affect the correlations or the likelihood-ratio test.
Reports come from collect_injuries.py (report time at or before the cutoff).
"""
import argparse
import json
import re

import numpy as np

from collect_pilot import ROOT, stamp
from evaluate_orthogonality import SPLITS, REPORT_DIR, boosting_forecasts, bootstrap, day_folds, logit, sigmoid, statistics

INJURIES = ROOT / "data/processed/injury_full/injuries.jsonl"
WEIGHTS = {"Out": 1.0, "Doubtful": 0.5, "Questionable": 0.5}
SCALE = 0.1


def burden(segment):
    return sum(WEIGHTS[m.group(1)] for m in re.finditer(r"\b(Out|Doubtful|Questionable)\s+Injury/Illness", segment))


def feature(row):
    text = row["injury_report"]["text"]
    i_home, i_away = text.find(row["home_team"]), text.find(row["away_team"])
    if i_home < 0 or i_away < 0:
        return None
    if i_home < i_away:
        home, away = text[i_home:i_away], text[i_away:]
    else:
        away, home = text[i_away:i_home], text[i_home:]
    return burden(away) - burden(home)


def load(split, forecasts, reports):
    rows = []
    for r in reports:
        if r["split"] != split or r["injury_report"] is None or r["game_id"] not in forecasts:
            continue
        if stamp(r["injury_report"]["report_time_utc"]) > stamp(r["cutoff_utc"]):
            raise ValueError(f"Injury report after cutoff: {r['game_id']}")
        f = feature(r)
        if f is not None:
            p = forecasts[r["game_id"]]
            rows.append((r["game_id"], p["game_date_et"], p["home_win"], p["p_boost"], f))
    return rows, sum(f["split"] == split for f in forecasts.values())


def analyse(name, rows, total, seed=230, replicates=2000):
    y = np.array([r[2] for r in rows])
    base = np.array([r[3] for r in rows])
    feat = np.array([r[4] for r in rows], dtype=float)
    dates = np.array([r[1] for r in rows])
    other = sigmoid(logit(base) + SCALE * feat)
    point = statistics(base, other, y, day_folds(dates, 5, seed))
    point["intervals"] = bootstrap(base, other, y, dates, replicates, seed)
    point.update(name=name, games_with_report=len(rows), games_in_split=total, scale=SCALE,
                 feature_mean_abs=float(np.abs(feat).mean()), feature_corr_with_residual_raw=float(np.corrcoef(feat, y - base)[0, 1]))
    return point


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=SPLITS)
    args = parser.parse_args()
    forecasts = boosting_forecasts(args.splits)
    with open(INJURIES, encoding="utf-8") as stream:
        reports = [json.loads(line) for line in stream if line.strip()]
    results, pooled = {}, []
    for split in args.splits:
        rows, total = load(split, forecasts, reports)
        if len(rows) >= 30:
            results[split] = analyse(split, rows, total)
            pooled.extend(rows)
    if len(results) > 1:
        results["pooled"] = analyse("pooled", pooled, sum(r["games_in_split"] for r in results.values()))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "injury_text_full.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, p in results.items():
        iv = p["intervals"]
        ci = lambda k: f"[{iv[k][0]:+.3f}, {iv[k][1]:+.3f}]"
        print(f"{name}: {p['games_with_report']}/{p['games_in_split']} games with a matched report")
        print(f"  share of feature explained by boosting {p['explained_by_base']:.3f}  {ci('explained_by_base')}")
        print(f"  corr(delta, boosting residual)        {p['delta_residual']:+.3f}  {ci('delta_residual')}")
        print(f"  coef on delta (0 = nothing)           {p['delta_coef']:+.3f}  {ci('delta_coef')}  LR p = {p['lr_p']:.2g}")
        print(f"  out-of-fold Brier gain                {p['oof_brier_gain']:+.4f}")


if __name__ == "__main__":
    main()
