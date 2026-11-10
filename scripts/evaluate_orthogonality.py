"""Measure how much a second forecast (market, text model) adds beyond the boosting forecast.

Works on any probability column for the same games as the boosting pipeline's
out-of-sample forecasts (shared predictions.parquet, model gradient_boosting; outcomes
from the requested labels_<split>.parquet only), so the same code answers "is the
market redundant with statistics?" today and "is the text correction orthogonal to
statistics?" once per-game text predictions exist.

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

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from collect_pilot import ROOT
from nba_dataset import SPLITS, Dataset

REPORT_DIR = ROOT / "reports/orthogonality"
KEYS = ["logit_corr", "delta_residual", "delta_coef", "explained_by_base"]


def join_forecasts(predictions, labels, splits):
    """game_id -> {split, game_date_et, home_win, p_boost}; every forecast in `splits` needs a label."""
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits {sorted(unknown)}; expected some of {SPLITS}")
    outcome = {r["game_id"]: int(r["home_win"]) for r in labels}
    forecasts = {}
    for row in predictions:
        if row["split"] in splits:
            if row["game_id"] not in outcome:
                raise ValueError(f"No outcome label for {row['game_id']}")
            forecasts[row["game_id"]] = {"split": row["split"], "game_date_et": row["game_date_et"],
                                         "home_win": outcome[row["game_id"]], "p_boost": float(row["p_boost"])}
    return forecasts


def boosting_forecasts(splits, dataset=None):
    """The frozen boosting forecasts, all from models that never saw the game (TRAIN seasons from
    earlier-season fits, DEV/TEST from the TRAIN fit), with outcomes from only the requested splits."""
    dataset = Dataset() if dataset is None else dataset
    return join_forecasts(dataset.predictions("gradient_boosting", stage="baseline"), dataset.labels(splits), splits)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def day_folds(dates, folds, seed):
    """Assign each ET date (and all its games) to one of `folds` folds."""
    order = np.random.default_rng(seed).permutation(np.unique(dates))
    fold_of = {d: i % folds for i, d in enumerate(order)}
    return np.array([fold_of[d] for d in dates])


def fit(x, y):
    model = LogisticRegression(C=1e6, max_iter=2000).fit(x, y)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
    return model, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def statistics(base, other, y, folds=None):
    lb, lo = logit(base), logit(other)
    delta = lo - lb
    varies = delta.std() > 0
    out = {"games": int(len(y)),
           "logit_corr": float(np.corrcoef(lb, lo)[0, 1]),
           "delta_sd": float(delta.std()),
           "delta_residual": float(np.corrcoef(delta, y - base)[0, 1]) if varies else 0.0}
    slope = np.polyfit(lb, delta, 1)
    out["explained_by_base"] = float(1 - np.var(delta - np.polyval(slope, lb)) / np.var(delta)) if varies else 0.0
    _, ll0 = fit(lb[:, None], y)
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


def bootstrap(base, other, y, dates, repeats, seed, keys=KEYS):
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


def other_forecasts(source, column):
    """game_id -> probability: Polymarket prices at the boosting cutoff (tipoff - 60 min), or a CSV/JSONL
    file with game_id and `column`."""
    if source == "polymarket":
        from collect_polymarket import load_price_observations
        return {r["game_id"]: float(r["home_probability"]) for r in load_price_observations()
                if r["record_kind"] == "cutoff" and r["home_probability"] is not None}
    path = ROOT / source
    with open(path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream)) if path.suffix == ".csv" else [json.loads(l) for l in stream if l.strip()]
    return {r["game_id"]: float(r[column]) for r in rows}


def joined(forecasts, other):
    ids = sorted(set(forecasts) & set(other))
    y = np.array([forecasts[i]["home_win"] for i in ids])
    base = np.array([forecasts[i]["p_boost"] for i in ids])
    dates = np.array([forecasts[i]["game_date_et"] for i in ids])
    return ids, y, base, np.array([other[i] for i in ids]), dates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=SPLITS)
    parser.add_argument("--other", default="polymarket", help="'polymarket' or a CSV/JSONL path with game_id and --other-col")
    parser.add_argument("--other-col", default="probability")
    parser.add_argument("--name", default="polymarket_vs_boosting")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    forecasts = boosting_forecasts(args.splits)
    ids, y, base, other, dates = joined(forecasts, other_forecasts(args.other, args.other_col))
    if len(ids) < 30:
        raise SystemExit(f"Only {len(ids)} games overlap; too few for a meaningful estimate")
    point = statistics(base, other, y, day_folds(dates, args.folds, args.seed))
    point["intervals"] = bootstrap(base, other, y, dates, args.replicates, args.seed)
    point.update(name=args.name, splits=args.splits, base_games_available=len(forecasts), overlap_games=len(ids),
                 by_split={s: sum(forecasts[i]["split"] == s for i in ids) for s in args.splits},
                 first_date=min(dates.tolist()), last_date=max(dates.tolist()), bootstrap_replicates=args.replicates)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{args.name}.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
    iv = point["intervals"]
    ci = lambda k: f"[{iv[k][0]:+.3f}, {iv[k][1]:+.3f}]"
    print(f"{args.name}: {point['overlap_games']} games {point['by_split']} "
          f"({point['first_date']} to {point['last_date']}), {len(np.unique(dates))} days")
    print(f"  log-odds correlation        {point['logit_corr']:+.3f}  {ci('logit_corr')}")
    print(f"  share of delta explained    {point['explained_by_base']:.3f}   {ci('explained_by_base')}")
    print(f"  corr(delta, base residual)  {point['delta_residual']:+.3f}  {ci('delta_residual')}")
    print(f"  coef on delta (0 = nothing) {point['delta_coef']:+.3f}  {ci('delta_coef')}   LR p = {point['lr_p']:.2g}")
    print(f"  out-of-fold Brier gain      {point['oof_brier_gain']:+.4f}  (positive = other helps)")


if __name__ == "__main__":
    main()
