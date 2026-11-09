"""TRAIN-only augmentation using Prior Labs' official joint-table sampler.

The continuous-regression checkpoint was unavailable at experiment setup. The
qualified fallback bins primitive variables using TRAIN-only quantiles, samples
their joint categorical distribution with TabPFNUnsupervisedModel, and maps bins
back to their TRAIN medians. Exact engineered identities are reconstructed.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from box_score_features import PLAYER_STAT_METRICS, TEAM_METRICS, WINDOWS
from collect_pilot import ROOT


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def primitive_names(names):
    return [name for name in names if not (
        name.startswith("difference_") or name.endswith("_estimated_net_rating")
        or name.endswith("_boxscore_missing") or name.endswith("_back_to_back"))]


def encode_columns(x, names, max_bins=8):
    """Contiguous integer categories, plus TRAIN-only inverse-bin medians.

    Low-cardinality values are encoded exactly. A separate missing category
    preserves feature availability without imputing outcomes or consulting DEV.
    Constants are omitted from model calls and restored on output.
    """
    columns, specs = [], []
    for index, name in enumerate(names):
        values = x[:, index]
        finite = np.isfinite(values)
        observed = values[finite]
        if not len(observed):
            specs.append({"name": name, "constant": None})
            continue
        unique = np.unique(observed)
        if len(unique) == 1 and finite.all():
            specs.append({"name": name, "constant": float(unique[0])})
            continue
        if len(unique) <= max_bins:
            raw_codes = np.searchsorted(unique, observed)
            representatives = unique.astype(float).tolist()
            boundaries = None
        else:
            boundaries = np.unique(np.quantile(observed, np.linspace(0, 1, max_bins + 1)[1:-1]))
            raw = np.searchsorted(boundaries, observed, side="right")
            present = np.unique(raw)
            raw_codes = np.searchsorted(present, raw)
            representatives = [float(np.median(observed[raw == c])) for c in present]
            boundaries = boundaries.tolist()
        codes = np.empty(len(values), dtype=np.float32)
        codes[finite] = raw_codes
        if not finite.all():
            codes[~finite] = len(representatives)
            representatives.append(None)
        columns.append(codes)
        specs.append({"name": name, "column": len(columns),
                      "representatives": representatives, "quantile_boundaries": boundaries})
    encoded = (np.column_stack(columns).astype(np.float32) if columns else
               np.empty((len(x), 0), dtype=np.float32))
    return encoded, specs


def decode_columns(encoded, specs):
    result = {}
    for spec in specs:
        if "constant" in spec:
            result[spec["name"]] = np.full(len(encoded), np.nan if spec["constant"] is None else spec["constant"])
            continue
        codes = encoded[:, spec["column"]]
        if not np.isfinite(codes).all() or not np.equal(codes, np.rint(codes)).all():
            raise ValueError("Official sampler returned non-integral categories")
        codes = codes.astype(int)
        values = np.asarray([np.nan if v is None else v for v in spec["representatives"]])
        if np.any(codes < 0) or np.any(codes >= len(values)):
            raise ValueError("Official sampler returned an unknown category")
        result[spec["name"]] = values[codes]
    return result


def reconstruct_features(values, names, fill_values=None):
    """Enforce algebraic identities and missing-stat semantics of the baseline."""
    result = {name: value.copy() for name, value in values.items()}
    for side in ("home", "away"):
        for field in ("previous_games", "days_since_previous_game"):
            name = f"{side}_{field}"
            result[name] = np.rint(result[name])
        games = result[f"{side}_previous_games"]
        name = f"{side}_season_win_rate"
        result[name] = np.rint(result[name] * games) / games
        name = f"{side}_last_5_win_rate"
        result[name] = np.rint(result[name] * 5) / 5
        result[f"{side}_back_to_back"] = (result[f"{side}_days_since_previous_game"] == 1).astype(float)
        for window, limit in WINDOWS:
            count = games if limit is None else np.minimum(games, limit)
            prefix = f"{side}_{window}_"
            for kind in ("team", "player"):
                coverage = prefix + kind + "_boxscore_coverage"
                result[coverage] = np.rint(result[coverage] * count) / count
                if kind == "player":
                    result[coverage] = np.minimum(result[coverage], result[prefix + "team_boxscore_coverage"])
                result[prefix + kind + "_boxscore_missing"] = (result[coverage] < 1).astype(float)
            # A synthetic missing category sampled independently of coverage
            # needs an observed fallback before the coverage mask is applied.
            for metric in (*TEAM_METRICS, *PLAYER_STAT_METRICS):
                name = prefix + metric
                if name.endswith("estimated_net_rating"):
                    continue
                if name not in result:
                    raise ValueError(f"Missing primitive feature {name}")
                if np.isnan(result[name]).any():
                    observed = result[name][np.isfinite(result[name])]
                    fill = (fill_values[name] if fill_values is not None else
                            float(np.median(observed)) if len(observed) else 0.0)
                    result[name] = np.where(np.isfinite(result[name]), result[name], fill)
            result[prefix + "estimated_net_rating"] = (
                result[prefix + "estimated_offensive_rating"] - result[prefix + "estimated_defensive_rating"])
            for metric in PLAYER_STAT_METRICS:
                result[prefix + metric][result[prefix + "player_boxscore_coverage"] < 1] = np.nan
            for metric in TEAM_METRICS:
                result[prefix + metric][result[prefix + "team_boxscore_coverage"] == 0] = np.nan
        for window, limit in WINDOWS:
            if limit is None:
                continue
            same_window = games <= limit
            for metric in (*TEAM_METRICS, *PLAYER_STAT_METRICS,
                           "team_boxscore_coverage", "player_boxscore_coverage",
                           "team_boxscore_missing", "player_boxscore_missing"):
                result[f"{side}_{window}_{metric}"][same_window] = result[f"{side}_season_{metric}"][same_window]
        same_window = games <= 5
        result[f"{side}_last_5_win_rate"][same_window] = result[f"{side}_season_win_rate"][same_window]
    for name in names:
        if name.startswith("difference_"):
            tail = name.removeprefix("difference_")
            result[name] = result["home_" + tail] - result["away_" + tail]
    matrix = np.column_stack([result[name] for name in names]).astype(np.float64)
    if np.isinf(matrix).any():
        raise ValueError("Synthetic reconstruction produced infinite features")
    return matrix


def generate(config, X_early, y_early, names, source_ids, output_dir, smoke=False):
    """Generate once and return a provenance manifest; never consume query data."""
    from run_gradient_boosting import FEATURE_NAMES

    names, source_ids = list(names), list(source_ids)
    x, y = np.asarray(X_early, dtype=float), np.asarray(y_early)
    settings = dict(config["generator"])
    output = Path(output_dir)
    if list(FEATURE_NAMES) != names or y.shape != (len(x),) or x.shape != (len(y), len(names)):
        raise ValueError("Expected the frozen 252-feature matrix and aligned outcomes")
    if len(source_ids) != len(y) or len(set(source_ids)) != len(y):
        raise ValueError("Expected unique, aligned real earlier-TRAIN game IDs")
    if set(np.unique(y)) != {0, 1} or np.isinf(x).any():
        raise ValueError("Expected binary TRAIN outcomes and no infinite features")
    if settings.get("method") != "categorical_quantile_joint" or not 2 <= settings["max_bins"] <= 9:
        raise ValueError("Expected the qualified categorical generator with at most nine bins")
    if (output / "manifest.json").exists():
        raise ValueError("Synthetic data already exist; validate and reuse them")
    for filename in ("features.npy", "labels.npy"):
        if (output / filename).exists():
            raise ValueError("Unbound synthetic artifact exists; inspect before replacement")
    seed = int(config["seed"])
    context_count = min(len(x), 32 if smoke else int(settings["context_rows"]))
    n_samples = 16 if smoke else int(settings["n_samples"])
    if context_count < 16 or n_samples < 1:
        raise ValueError("Generation requires at least 16 real TRAIN rows")
    chosen = np.sort(np.random.default_rng(seed).choice(len(x), context_count, replace=False))
    x, y = x[chosen], y[chosen]
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The deterministic generator context must contain both outcomes")
    primitive = primitive_names(names)
    encoded, specs = encode_columns(x[:, [names.index(n) for n in primitive]], primitive,
                                   max_bins=int(settings["max_bins"]))
    joint = np.column_stack([y, encoded]).astype(np.float32)
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if os.environ.get("FAST_TEST_MODE") == "1":
        raise ValueError("FAST_TEST_MODE changes generation sample count")
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    if version("tabpfn-extensions") != settings["extension_version"]:
        raise ValueError("Installed generator differs from the pinned extension")
    baseline = json.loads((ROOT / config["baseline_config"]).read_text())
    model_config = baseline["model"]
    checkpoint = ROOT / model_config["checkpoint_path"]
    if digest(checkpoint) != model_config["sha256"]:
        raise ValueError("Classifier checkpoint hash changed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    classifier = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, model_path=checkpoint,
        n_estimators=settings["n_estimators"], device=config["device"],
        random_state=seed, fit_mode="fit_preprocessors", n_preprocessing_jobs=1,
        memory_saving_mode="auto", inference_config={"SUBSAMPLE_SAMPLES": None})
    sampler = TabPFNUnsupervisedModel(tabpfn_clf=classifier, tabpfn_reg=None)
    sampler.set_categorical_features(list(range(joint.shape[1])))
    sampler.fit(joint)
    started = time.monotonic()
    print(f"Official joint sampler: {context_count} TRAIN rows, {joint.shape[1]} categorical columns -> {n_samples} synthetic rows", flush=True)
    generated = sampler.generate_synthetic_data(
        n_samples=n_samples, t=settings["temperature"],
        n_permutations=settings["n_permutations"]).detach().cpu().numpy()
    if generated.shape != (n_samples, joint.shape[1]) or not np.isfinite(generated).all():
        raise ValueError("Unexpected output from official joint sampler")
    labels = generated[:, 0]
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Joint sampler returned invalid binary outcomes")
    fill_values = {}
    for name in primitive:
        column = x[:, names.index(name)]
        observed = column[np.isfinite(column)]
        fill_values[name] = float(np.median(observed)) if len(observed) else 0.0
    features = reconstruct_features(decode_columns(generated, specs), names, fill_values)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "features.npy", features, allow_pickle=False)
    np.save(output / "labels.npy", labels.astype(np.int64), allow_pickle=False)
    manifest = {
        "schema_version": 1, "status": "complete", "smoke": bool(smoke),
        "method": "official TabPFNUnsupervisedModel.generate_synthetic_data, TRAIN-only categorical quantile bins",
        "input_partition_game_ids": source_ids,
        "source_game_ids": [source_ids[i] for i in chosen],
        "feature_names": names, "context_rows": context_count, "n_samples": n_samples,
        "seed": seed, "settings": settings, "model": model_config,
        "input_partition_sha256": hashlib.sha256(np.ascontiguousarray(X_early,dtype=np.float64).tobytes() + np.ascontiguousarray(y_early,dtype=np.int64).tobytes()).hexdigest(),
        "runtime_seconds": time.monotonic() - started,
        "versions": {name: version(name) for name in ("tabpfn", "tabpfn-extensions", "torch", "numpy", "scikit-learn")},
        "bin_encoding": specs, "missing_reconstruction_fill_values": fill_values,
        "artifact_sha256": {name: digest(output / name) for name in ("features.npy", "labels.npy")},
        "implementation_sha256": digest(__file__),
        "limitation": "Continuous-regressor checkpoint unavailable (HuggingFace TLS timeouts; official fallback HTTP404). Quantile-bin median decoding is coarse; synthetic rows are model-generated augmentation, not additional observed NBA evidence.",
        "constraints": "Reconstruct home-away differences, net ratings, rest flags, win-rate/count arithmetic, coverage/missingness; all bin fitting uses selected earlier TRAIN only.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
