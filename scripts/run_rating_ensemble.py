"""The statistics-only ensemble: XGBoost + a lean logistic regression + TabPFN on player-rating features.

Three members, all fitted on TRAIN and blended with fixed weights chosen on TRAIN walk-forward:
  xgb     XGBoost on the published 252 features, the situational families and both rating sets
  lean    logistic regression on four numbers: the injury-adjusted rating margin from game-level
          ratings, the same from stint ratings, the rest-day difference and the back-to-back difference
  tabpfn  TabPFN-2.5 on the compact home-minus-away columns (optional: needs torch and the checkpoint)

Stages (TEST is never read: only TRAIN and DEV rows are loaded and play-by-play stops at DEV):
  features  build both rating sets and the three matrices; cached in data/processed/rating_ensemble/
  tabpfn    fit on TRAIN seasons before 2023-24 and predict that season, then fit TRAIN and predict DEV
  evaluate  TRAIN walk-forward (seasons 2019-20 to 2023-24, each fitted on earlier seasons), then one DEV
            comparison with Polymarket's cutoff price; writes results/rating_ensemble.json

The market price is used only to score against, never as a model input.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from collect_pilot import ROOT
from nba_dataset import Dataset
import rating_features as rf
import run_gradient_boosting as rgb
import situational_features as sf
import stint_features

CACHE = ROOT / "data/processed/rating_ensemble"
RESULT = ROOT / "results/rating_ensemble.json"
INJURY_DIR = ROOT / "data/processed/injury_full"
WALK_FORWARD = (2019, 2020, 2021, 2022, 2023)
TABPFN_CHECK_SEASON = 2023
STINT_RIDGE = 500.0          # with carry: each weekly refit is shrunk toward the previous one
WEIGHTS = {"tabpfn": 0.50, "lean": 0.35, "xgb": 0.15}
WEIGHTS_WITHOUT_TABPFN = {"lean": 0.70, "xgb": 0.30}
XGB = dict(tree_method="hist", objective="binary:logistic", subsample=0.8, colsample_bytree=0.6, min_child_weight=20,
           reg_lambda=1.0, learning_rate=0.03, max_depth=2, n_estimators=300)
SEED, XGB_SEEDS, BOOTSTRAP = 230, 3, 2000


def build_features():
    dataset = Dataset()
    inputs, labels = dataset.inputs(("train", "dev")), dataset.labels(("train", "dev"))
    x, y = rgb.arrays(inputs, labels)
    names = list(rgb.FEATURE_NAMES)
    ids = [r["game_id"] for r in inputs]
    injuries = sorted(INJURY_DIR.glob("part_*.jsonl"))
    start = time.time()
    log = lambda text: print(f"[{time.time() - start:5.0f}s] {text}", flush=True)
    situational = sf.from_dataset(dataset, injuries)
    log("situational features built")
    game = rf.from_dataset(dataset, injuries, INJURY_DIR / "player_names.json")
    log("game-level ratings built")
    stints = stint_features.from_dataset(dataset, log=log)
    stint = rf.from_dataset(dataset, injuries, INJURY_DIR / "player_names.json", ridge=STINT_RIDGE, stints=stints, carry=True)
    log("stint ratings built")
    table = lambda features, columns: np.array([[features[g][n] for n in columns] for g in ids], float)
    stint_columns = rf.feature_names(["apm", "apm_injury"])

    def margin(features):       # injury-adjusted margin, else the full-rotation margin, else 0
        best = np.array([features[g]["apm_margin_injury"] for g in ids])
        full = np.array([features[g]["apm_margin_full"] for g in ids])
        best = np.where(np.isfinite(best), best, full)
        return np.where(np.isfinite(best), best, 0.0)

    out = {
        "ids": np.array(ids), "y": y, "split": np.array([r["split"] for r in inputs]),
        "season": np.array([r["season"] for r in inputs]), "dates": np.array([str(r["game_date_et"]) for r in inputs]),
        "xgb": np.hstack([x, table(situational, sf.feature_names(list(sf.FAMILIES))),
                          table(game, rf.feature_names(["elo", "apm", "apm_injury"])), table(stint, stint_columns)]),
        "lean": np.column_stack([margin(game), margin(stint),
                                 np.nan_to_num(x[:, names.index("difference_days_since_previous_game")]).clip(-5, 5),
                                 np.nan_to_num(x[:, names.index("difference_back_to_back")])]),
        "tabpfn": np.hstack([x[:, [i for i, n in enumerate(names) if n.startswith("difference_")]],
                             table(situational, [n for n in sf.FEATURE_NAMES if n.startswith("diff_") and n not in sf.CONSTANT]),
                             table(game, rf.FEATURE_NAMES), table(stint, stint_columns)]),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "features.pkl").open("wb") as stream:
        pickle.dump(out, stream)
    log(f"matrices: xgb {out['xgb'].shape}, lean {out['lean'].shape}, tabpfn {out['tabpfn'].shape}")


def load(name):
    with (CACHE / name).open("rb") as stream:
        return pickle.load(stream)


def run_tabpfn(device, estimators):
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    import run_tabpfn as rt
    config = json.loads(rt.CONFIG_PATH.read_text())
    config["model"].update(device=device, n_estimators=estimators)
    f = load("features.pkl")
    train, out = f["split"] == "train", {"ids": f["ids"], "device": device, "n_estimators": estimators}
    for label, fit, predict in (("check", train & (f["season"] < TABPFN_CHECK_SEASON), train & (f["season"] == TABPFN_CHECK_SEASON)),
                                ("dev", train, f["split"] == "dev")):
        start = time.time()
        model = rt.model_for(config)
        model.fit(f["tabpfn"][fit], f["y"][fit])
        p = np.full(len(f["y"]), np.nan)
        p[predict] = rt.positive_probabilities(model, f["tabpfn"][predict])
        out[label] = p
        print(f"{label}: Brier {np.mean((p[predict] - f['y'][predict]) ** 2):.4f} ({time.time() - start:.0f}s)", flush=True)
        with (CACHE / "tabpfn.pkl").open("wb") as stream:
            pickle.dump(out, stream)


def fit_xgb(f, fit, predict, seeds=XGB_SEEDS):
    import xgboost
    return np.mean([xgboost.XGBClassifier(random_state=SEED + s, n_jobs=os.cpu_count(), **XGB).fit(f["xgb"][fit], f["y"][fit])
                    .predict_proba(f["xgb"][predict])[:, 1] for s in range(seeds)], axis=0)


def fit_lean(f, fit, predict):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=2000).fit(f["lean"][fit], f["y"][fit]).predict_proba(f["lean"][predict])[:, 1]


def blend(members):
    weights = WEIGHTS if "tabpfn" in members else WEIGHTS_WITHOUT_TABPFN
    return sum(w * members[k] for k, w in weights.items())


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def evaluate():
    from collect_polymarket import load_price_observations
    from evaluate_orthogonality import day_folds, statistics
    f = load("features.pkl")
    y, season, train, dev = f["y"], f["season"], f["split"] == "train", f["split"] == "dev"
    tab = load("tabpfn.pkl") if (CACHE / "tabpfn.pkl").exists() else None
    if tab is not None and list(tab["ids"]) != list(f["ids"]):
        raise ValueError("TabPFN predictions were made for a different feature build")

    # TRAIN walk-forward: out-of-season forecasts for every validation season.
    oof = {"xgb": np.full(len(y), np.nan), "lean": np.full(len(y), np.nan)}
    for s in WALK_FORWARD:
        fit, predict = train & (season < s), train & (season == s)
        oof["xgb"][predict], oof["lean"][predict] = fit_xgb(f, fit, predict, 2), fit_lean(f, fit, predict)
    validation = train & np.isin(season, WALK_FORWARD)
    walk = {k: brier(v[validation], y[validation]) for k, v in oof.items()}
    walk["blend_without_tabpfn"] = brier(blend(oof)[validation], y[validation])
    result = {"train_walk_forward": {"seasons": list(WALK_FORWARD), "games": int(validation.sum()), "brier": walk}}
    if tab is not None:
        check = train & (season == TABPFN_CHECK_SEASON)
        members = {"xgb": oof["xgb"][check], "lean": oof["lean"][check], "tabpfn": tab["check"][check]}
        result["train_check_season"] = {"season": TABPFN_CHECK_SEASON, "games": int(check.sum()),
                                        "brier": {**{k: brier(v, y[check]) for k, v in members.items()},
                                                  "blend": brier(blend(members), y[check])}}

    # DEV: fit on all of TRAIN, compare once with the market's cutoff price.
    members = {"xgb": fit_xgb(f, train, dev), "lean": fit_lean(f, train, dev)}
    if tab is not None:
        members["tabpfn"] = tab["dev"][dev]
    members["blend"] = blend(members)
    price = {r["game_id"]: r["home_probability"] for r in load_price_observations()
             if r["record_kind"] == "cutoff" and r["home_probability"] is not None}
    market = np.array([price.get(g, np.nan) for g in f["ids"][dev]])
    priced = np.isfinite(market)
    y_dev, dates = y[dev], f["dates"][dev]
    candidates = {"polymarket": market[priced], **{k: v[priced] for k, v in members.items()}}
    interval = rgb.bootstrap_brier(y_dev[priced], candidates, dates[priced], BOOTSTRAP, SEED, "polymarket")["difference_vs_reference"]
    folds = day_folds(dates[priced], 5, SEED)
    market_adds = statistics(members["blend"][priced], market[priced], y_dev[priced], folds)
    model_adds = statistics(market[priced], members["blend"][priced], y_dev[priced], folds)
    result["dev"] = {
        "games": int(dev.sum()), "priced_games": int(priced.sum()),
        "brier_all_games": {k: brier(v, y_dev) for k, v in members.items()},
        "brier_priced_games": {k: brier(v, y_dev[priced]) for k, v in candidates.items()},
        "brier_minus_polymarket_95": {k: list(map(float, interval[k])) for k in members},
        "market_adds_to_blend": {k: float(market_adds[k]) for k in ("oof_brier_gain", "delta_coef", "lr_p")},
        "blend_adds_to_market": {k: float(model_adds[k]) for k in ("oof_brier_gain", "delta_coef", "lr_p")}}
    result["weights"] = WEIGHTS if tab is not None else WEIGHTS_WITHOUT_TABPFN
    result["tabpfn"] = None if tab is None else {"device": tab["device"], "n_estimators": tab["n_estimators"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (CACHE / "forecasts.pkl").open("wb") as stream:      # for later stages that stack on these forecasts
        pickle.dump({"ids": f["ids"], "train_walk_forward": oof, "dev": members, "dev_market": market}, stream)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("features", "tabpfn", "evaluate"))
    parser.add_argument("--device", default="cpu", help="TabPFN device: cpu, mps or cuda")
    parser.add_argument("--estimators", type=int, default=2, help="TabPFN ensemble members")
    args = parser.parse_args()
    if args.stage == "features":
        build_features()
    elif args.stage == "tabpfn":
        run_tabpfn(args.device, args.estimators)
    else:
        evaluate()


if __name__ == "__main__":
    main()
