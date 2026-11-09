"""Is game-note text orthogonal to, and informative beyond, the boosting forecast?

Text -> correction: TF-IDF of each game's approved pregame note excerpt, ridge
regression onto the boosting residual (outcome - boosting probability). The
correction for each game is out-of-fold: folds are grouped by ET date and the
ridge penalty is chosen inside the training folds only. The text forecast is
sigmoid(logit(boosting) + c * correction); c does not affect the correlations
or the likelihood-ratio test. A permutation test (shuffling residuals across
games, redoing the whole out-of-fold fit) gives an honest p-value at small n.
Only games with out-of-sample boosting probabilities are used (validation + test).
"""
import argparse
import csv
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

from collect_pilot import ROOT
from evaluate_orthogonality import REPORT_DIR, bootstrap, logit, statistics

KEYS = ["logit_corr", "delta_residual", "delta_coef", "explained_by_base"]
SCALE = 1.0


def load():
    boosting = {}
    for split in ("validation", "test"):
        with open(ROOT / f"data/processed/gradient_boosting/{split}_predictions.csv", encoding="utf-8", newline="") as s:
            for r in csv.DictReader(s):
                boosting[r["game_id"]] = (split, int(r["home_win"]), float(r["gradient_boosting"]))
    rows = []
    with open(ROOT / "data/processed/game_notes_collection/approved_excerpts.jsonl", encoding="utf-8") as s:
        for line in s:
            r = json.loads(line)
            if r["game_id"] in boosting:
                split, y, p = boosting[r["game_id"]]
                rows.append({"game_id": r["game_id"], "date": r["game_date_et"], "text": r["text"], "team": r["team"],
                             "split": split, "y": y, "base": p})
    return rows


def oof_correction(texts, target, dates, folds, seed):
    days = np.unique(dates)
    order = np.random.default_rng(seed).permutation(days)
    fold_of = {d: i % folds for i, d in enumerate(order)}
    fold = np.array([fold_of[d] for d in dates])
    out = np.zeros(len(texts))
    for k in range(folds):
        train, test = fold != k, fold == k
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
                              RidgeCV(alphas=np.logspace(-1, 3, 9)))
        model.fit([texts[i] for i in np.flatnonzero(train)], target[train])
        # Drop the training-fold mean: predicting a held-out fold with it anti-correlates
        # with that fold's mean (fixed overall total), a leave-fold-out artifact, not signal.
        out[test] = model.predict([texts[i] for i in np.flatnonzero(test)]) - target[train].mean()
    return out, fold


def analyse(rows, args):
    texts = [r["text"] for r in rows]
    y = np.array([r["y"] for r in rows])
    base = np.array([r["base"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    residual = y - base
    correction, fold = oof_correction(texts, residual, dates, args.folds, args.seed)
    other = 1 / (1 + np.exp(-(logit(base) + SCALE * correction)))
    point = statistics(base, other, y, fold)
    point["intervals"] = bootstrap(base, other, y, dates, args.replicates, args.seed, KEYS)
    observed = point["delta_residual"]
    rng = np.random.default_rng(args.seed)
    null = []
    for _ in range(args.permutations):
        shuffled = rng.permutation(residual)
        c, _ = oof_correction(texts, shuffled, dates, args.folds, args.seed)
        null.append(np.corrcoef(c, residual)[0, 1] if c.std() > 0 else 0.0)
    point["permutation_p_one_sided"] = float((1 + sum(n >= observed for n in null)) / (1 + len(null)))
    point["permutation_null_95th_percentile"] = float(np.quantile(null, 0.95))
    point.update(games=len(rows), by_split={s: sum(r["split"] == s for r in rows) for s in ("validation", "test")},
                 by_team={t: sum(r["team"] == t for r in rows) for t in sorted({r["team"] for r in rows})},
                 correction_sd=float(correction.std()), permutations=args.permutations)
    return point


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    rows = load()
    if len(rows) < 30:
        raise SystemExit(f"Only {len(rows)} games; too few")
    result = analyse(rows, args)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "game_notes_tfidf.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    iv = result["intervals"]
    ci = lambda k: f"[{iv[k][0]:+.3f}, {iv[k][1]:+.3f}]"
    print(f"game notes (TF-IDF ridge, out-of-fold): {result['games']} games {result['by_split']}")
    print(f"  share of text correction explained by boosting {result['explained_by_base']:.3f}  {ci('explained_by_base')}")
    print(f"  corr(correction, boosting residual)            {result['delta_residual']:+.3f}  {ci('delta_residual')}")
    print(f"  permutation p (one-sided)                      {result['permutation_p_one_sided']:.3f}  (null 95th pct {result['permutation_null_95th_percentile']:+.3f})")
    print(f"  coef on correction (0 = nothing)               {result['delta_coef']:+.3f}  {ci('delta_coef')}  LR p = {result['lr_p']:.2g}")
    print(f"  out-of-fold Brier gain                         {result['oof_brier_gain']:+.4f}")


if __name__ == "__main__":
    main()
