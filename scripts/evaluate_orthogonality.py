"""Measure how much a second forecast (market, text model) adds beyond a base forecast (boosting).

Works on any pair of probability columns for the same games, so the same code
answers "is the market redundant with statistics?" today and "is the text
correction orthogonal to statistics?" once per-game text predictions exist.

Quantities (all with ET-calendar-day cluster bootstrap 95% intervals):
  logit_corr        correlation of the two forecasts' log-odds
  delta_residual    correlation of delta = logit(other) - logit(base) with the
                    base residual (outcome - base probability)
  delta_coef        coefficient on delta in  y ~ logit(base) + delta  (0 = nothing added)
  lr_p              likelihood-ratio p-value for delta_coef (df = 1; ignores day clustering)
  explained_by_base share of delta's variance a linear fit on logit(base) explains
  oof_brier_gain    out-of-fold Brier(base) - Brier(base+delta), folds grouped by day
"""
import argparse
import csv
import json
import math

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from collect_pilot import ROOT

REPORT_DIR = ROOT / "reports/orthogonality"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(x, y):
    model = LogisticRegression(C=1e6, max_iter=2000).fit(x, y)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
    return model, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def statistics(base, other, y, folds=None):
    lb, lo = logit(base), logit(other)
    delta = lo - lb
    out = {"games": int(len(y)),
           "logit_corr": float(np.corrcoef(lb, lo)[0, 1]),
           "delta_sd": float(delta.std()),
           "delta_residual": float(np.corrcoef(delta, y - base)[0, 1]) if delta.std() > 0 else 0.0}
    slope = np.polyfit(lb, delta, 1)
    out["explained_by_base"] = float(1 - np.var(delta - np.polyval(slope, lb)) / np.var(delta)) if delta.std() > 0 else 0.0
    reduced, ll0 = fit(lb[:, None], y)
    full, ll1 = fit(np.column_stack((lb, delta)), y)
    out["delta_coef"] = float(full.coef_[0][1])
    out["lr_stat"] = max(0.0, 2 * (ll1 - ll0))
    out["lr_p"] = float(chi2.sf(out["lr_stat"], 1))
    if folds is not None:
        gain = []
        for fold in np.unique(folds):
            test, train = folds == fold, folds != fold
            if len(np.unique(y[train])) < 2:
                continue
            m0 = LogisticRegression(C=1e6, max_iter=2000).fit(lb[train, None], y[train])
            m1 = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack((lb, delta))[train], y[train])
            b0 = (m0.predict_proba(lb[test, None])[:, 1] - y[test]) ** 2
            b1 = (m1.predict_proba(np.column_stack((lb, delta))[test])[:, 1] - y[test]) ** 2
            gain.extend(b0 - b1)
        out["oof_brier_gain"] = float(np.mean(gain))
    return out


def bootstrap(base, other, y, dates, repeats, seed, keys):
    unique, inverse = np.unique(dates, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(repeats):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[idx])) < 2:
            continue
        stat = statistics(base[idx], other[idx], y[idx])
        for k in keys:
            draws[k].append(stat[k])
    return {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in draws.items()}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_pair(args):
    base_rows = {r["game_id"]: r for r in read_csv(ROOT / args.base)}
    if args.other == "polymarket":
        other = {}
        with open(ROOT / "data/processed/polymarket/market_probs.jsonl", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                other[row["game_id"]] = float(row["home_probability"])
    else:
        other = {r["game_id"]: float(r[args.other_col]) for r in read_csv(ROOT / args.other)}
    ids = sorted(set(base_rows) & set(other))
    y = np.array([int(base_rows[i]["home_win"]) for i in ids])
    base = np.array([float(base_rows[i][args.base_col]) for i in ids])
    dates = np.array([base_rows[i]["game_date_et"] for i in ids])
    return ids, y, base, np.array([other[i] for i in ids]), dates, len(base_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="data/processed/gradient_boosting/test_predictions.csv")
    parser.add_argument("--base-col", default="gradient_boosting")
    parser.add_argument("--other", default="polymarket", help="'polymarket' or a CSV with game_id and --other-col")
    parser.add_argument("--other-col", default="probability")
    parser.add_argument("--name", default="polymarket_vs_boosting")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    ids, y, base, other, dates, base_games = load_pair(args)
    if len(ids) < 30:
        raise SystemExit(f"Only {len(ids)} games overlap; too few for a meaningful estimate")
    days = np.unique(dates)
    fold_of_day = {d: i % args.folds for i, d in enumerate(np.random.default_rng(args.seed).permutation(days))}
    folds = np.array([fold_of_day[d] for d in dates])
    point = statistics(base, other, y, folds)
    keys = ["logit_corr", "delta_residual", "delta_coef", "explained_by_base"]
    point["intervals"] = bootstrap(base, other, y, dates, args.replicates, args.seed, keys)
    point.update(name=args.name, base_games_available=base_games, overlap_games=len(ids),
                 first_date=min(dates.tolist()), last_date=max(dates.tolist()), bootstrap_replicates=args.replicates)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{args.name}.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
    iv = point["intervals"]
    ci = lambda k: f"[{iv[k][0]:+.3f}, {iv[k][1]:+.3f}]"
    print(f"{args.name}: {point['overlap_games']} games ({point['first_date']} to {point['last_date']}), {len(days)} days")
    print(f"  log-odds correlation        {point['logit_corr']:+.3f}  {ci('logit_corr')}")
    print(f"  share of delta explained    {point['explained_by_base']:.3f}   {ci('explained_by_base')}")
    print(f"  corr(delta, base residual)  {point['delta_residual']:+.3f}  {ci('delta_residual')}")
    print(f"  coef on delta (0 = nothing) {point['delta_coef']:+.3f}  {ci('delta_coef')}   LR p = {point['lr_p']:.2g}")
    print(f"  out-of-fold Brier gain      {point['oof_brier_gain']:+.4f}  (positive = other helps)")


if __name__ == "__main__":
    main()
