"""Is game-note text orthogonal to, and informative beyond, the boosting forecast?

Text -> correction: TF-IDF of each game's approved pregame notes (notes v2, every eligible
team document for the game joined), ridge regression onto the boosting residual
(outcome - boosting probability). The correction for each game is out-of-fold: folds are
grouped by ET date, and the TF-IDF vocabulary and ridge penalty are fitted inside the
training folds only. The text forecast is sigmoid(logit(boosting) + c * correction); c
does not affect the correlations or the likelihood-ratio test. A permutation test
(shuffling residuals across games, redoing every ridge fit) gives an honest p-value.

Documents follow the BAM snapshot's rules: HTML posts revised after the cutoff are
dropped, and every kept document must have been public at or before the cutoff.
Only games with an out-of-sample boosting forecast are used; TEST is excluded unless
requested with --splits.
"""
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV

from bam_data import availability_time
from collect_pilot import ROOT, stamp
from evaluate_orthogonality import SPLITS, boosting_forecasts, bootstrap, day_folds, logit, read_rows, sigmoid, statistics, REPORT_DIR

NOTES = ROOT / "data/processed/notes_v2/approved.jsonl"
SCALE = 1.0
ALPHAS = np.logspace(-1, 3, 9)


def eligible_documents(docs):
    """Approved notes usable as pregame input, and counts of those dropped by reason."""
    kept, dropped = [], {"html_modified_after_cutoff": 0}
    for doc in docs:
        if doc.get("modified_after_cutoff"):
            dropped["html_modified_after_cutoff"] += 1
            continue
        if availability_time(doc) > stamp(doc["cutoff_utc"]):
            raise ValueError(f"Note not shown public before its cutoff: {doc['text_path']}")
        kept.append(doc)
    return kept, dropped


def load(splits, notes_path=NOTES, root=ROOT, forecasts=None):
    """One row per game with a forecast in `splits` and at least one eligible note."""
    forecasts = boosting_forecasts(splits) if forecasts is None else forecasts
    docs, dropped = eligible_documents(read_rows(notes_path))
    by_game = {}
    for doc in sorted(docs, key=lambda d: (d["game_id"], d["team"])):
        if doc["game_id"] in forecasts:
            by_game.setdefault(doc["game_id"], []).append(doc)
    rows = []
    for game_id, game_docs in by_game.items():
        f = forecasts[game_id]
        rows.append({"game_id": game_id, "date": f["game_date_et"], "split": f["split"], "y": f["home_win"],
                     "base": f["p_boost"], "publishers": [d["team"] for d in game_docs],
                     "text": "\n\n".join((root / d["text_path"]).read_text(encoding="utf-8") for d in game_docs)})
    return sorted(rows, key=lambda r: (r["date"], r["game_id"])), dropped


def fold_matrices(texts, fold):
    """Per fold, TF-IDF fitted on the training folds only. It never sees outcomes, so permutations reuse it."""
    matrices = []
    for k in np.unique(fold):
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        x_train = vectorizer.fit_transform([texts[i] for i in train])
        matrices.append((train, test, x_train, vectorizer.transform([texts[i] for i in test])))
    return matrices


def oof_correction(matrices, target, n):
    out = np.zeros(n)
    for train, test, x_train, x_test in matrices:
        model = RidgeCV(alphas=ALPHAS).fit(x_train, target[train])
        # Drop the training-fold mean: predicting a held-out fold with it anti-correlates
        # with that fold's mean (fixed overall total), a leave-fold-out artifact, not signal.
        out[test] = model.predict(x_test) - target[train].mean()
    return out


def analyse(rows, args):
    y = np.array([r["y"] for r in rows])
    base = np.array([r["base"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    residual = y - base
    fold = day_folds(dates, args.folds, args.seed)
    matrices = fold_matrices([r["text"] for r in rows], fold)
    correction = oof_correction(matrices, residual, len(rows))
    other = sigmoid(logit(base) + SCALE * correction)
    point = statistics(base, other, y, fold)
    point["intervals"] = bootstrap(base, other, y, dates, args.replicates, args.seed)
    observed = point["delta_residual"]
    rng = np.random.default_rng(args.seed)
    null = []
    for _ in range(args.permutations):
        c = oof_correction(matrices, rng.permutation(residual), len(rows))
        null.append(np.corrcoef(c, residual)[0, 1] if c.std() > 0 else 0.0)
    point["permutation_p_one_sided"] = float((1 + sum(n >= observed for n in null)) / (1 + len(null)))
    point["permutation_null_95th_percentile"] = float(np.quantile(null, 0.95))
    publishers = [p for r in rows for p in r["publishers"]]
    point.update(splits=args.splits, by_split={s: sum(r["split"] == s for r in rows) for s in args.splits},
                 by_publisher={t: publishers.count(t) for t in sorted(set(publishers))},
                 correction_sd=float(correction.std()), permutations=args.permutations)
    return point


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"], choices=SPLITS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    rows, dropped = load(args.splits)
    if len(rows) < 30:
        raise SystemExit(f"Only {len(rows)} games; too few")
    result = analyse(rows, args)
    result["documents_dropped"] = dropped
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
