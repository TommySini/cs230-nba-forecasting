"""Is game-note text orthogonal to, and informative beyond, the boosting forecast?

Text -> correction: TF-IDF of each game's admitted pregame notes (the shared BAM cohort:
notes.parquet, every document for the game joined, page text from note_pages.parquet),
ridge regression onto the boosting residual (outcome - boosting probability). The
correction for each game is out-of-fold: folds are grouped by ET date, and the TF-IDF
vocabulary and ridge penalty are fitted inside the training folds only. The text
forecast is sigmoid(logit(boosting) + c * correction); c does not affect the
correlations or the likelihood-ratio test. A permutation test (shuffling residuals
across games, redoing every ridge fit) gives an honest p-value.

The admitted corpus already drops HTML posts revised after the cutoff; every document
is re-checked here to have been public at or before the cutoff. Only games with an
out-of-sample boosting forecast are used; TEST is excluded unless requested with --splits.
"""
import argparse
from collections import defaultdict
import json

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV

from bam_data import read_snapshot_inputs
from collect_pilot import ROOT, stamp
from evaluate_orthogonality import SPLITS, REPORT_DIR, bootstrap, day_folds, join_forecasts, logit, sigmoid, statistics
from nba_dataset import Dataset

SCALE = 1.0
ALPHAS = np.logspace(-1, 3, 9)


def page_texts(directory):
    """source sha256 -> full text, pages in order."""
    pages = defaultdict(list)
    for row in pq.read_table(directory / "note_pages.parquet", columns=["source_sha256", "page", "text"]).to_pylist():
        pages[row["source_sha256"]].append((row["page"], row["text"] or ""))
    return {sha: "\n".join(text for _, text in sorted(rows)) for sha, rows in pages.items()}


def game_rows(cohort, forecasts, texts):
    """One row per game with a forecast and at least one note; rejects any note public after the cutoff."""
    rows = []
    for game in cohort:
        if game["game_id"] not in forecasts or not game["sources"]:
            continue
        for source in game["sources"]:
            if stamp(source["archive_time"]) > stamp(game["cutoff_utc"]):
                raise ValueError(f"Note not shown public before its cutoff: {game['game_id']} {source['sha256']}")
        f = forecasts[game["game_id"]]
        rows.append({"game_id": game["game_id"], "date": f["game_date_et"], "split": f["split"], "y": f["home_win"],
                     "base": f["p_boost"], "publishers": [s["publisher"] for s in game["sources"]],
                     "text": "\n\n".join(texts[s["sha256"]] for s in game["sources"])})
    return sorted(rows, key=lambda r: (r["date"], r["game_id"]))


def load(splits, root=ROOT):
    directory = root / "data/processed"
    dataset = Dataset(root)
    cohort = read_snapshot_inputs(directory, root, splits=splits)
    forecasts = join_forecasts([g for g in cohort if g["p_boost"] is not None], dataset.labels(splits), splits)
    return game_rows(cohort, forecasts, page_texts(directory))


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
    rows = load(args.splits)
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
