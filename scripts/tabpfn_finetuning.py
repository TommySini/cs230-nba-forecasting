"""Bounded official TabPFN fine-tuning with explicit TRAIN-only validation.

The caller owns chronological cohort selection. This module never reads outcome
files and never chooses settings using DEV. Both real-only and augmented runs
use the same helper and official FinetunedTabPFNClassifier training algorithm.
"""
from __future__ import annotations

import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from collect_pilot import ROOT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _validated_xy(x, y):
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("Expected matching feature rows and binary outcome vector")
    # Frozen box-score features include missing values. Preserve those for the
    # official TabPFN preprocessing; infinities and invalid labels are rejected.
    if np.isinf(x).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite-or-missing features and binary labels")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both classes are required in the fixed TRAIN partition")
    return x, y.astype(np.int64)


def _prepare_environment(config):
    settings = config["model"]
    if version("tabpfn") != settings["package_version"] or settings["version"] != "v2.5":
        raise ValueError("Fine-tuning must use the pinned baseline TabPFN package and v2.5")
    checkpoint = ROOT / settings["checkpoint_path"]
    if file_sha256(checkpoint) != settings["sha256"]:
        raise ValueError("Original pretrained checkpoint changed")
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return checkpoint


class _History:
    def __init__(self, output):
        self.output = output
        self.steps = []
        self.epochs = []

    def setup(self, config):
        pass

    def log_step(self, metrics, step):
        if not all(math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError("Fine-tuning produced a non-finite training metric")
        self.steps.append({"step": step, **metrics})

    def log_epoch(self, metrics, step):
        self.epochs.append({"step": step, **metrics})
        self.finish()

    def finish(self):
        _write_json(self.output / "training_history.json", {"steps": self.steps, "epochs": self.epochs})


def train(config, X_early, y_early, X_val, y_val, output_dir):
    """Fine-tune on supplied earlier TRAIN rows; select using later TRAIN only.

    ``X_early`` may contain explicitly audited synthetic rows generated only from
    earlier TRAIN. The caller records real/synthetic membership separately.
    The returned checkpoint can be fitted with all TRAIN context for DEV inference.
    """
    pretrained = _prepare_environment(config)
    x, y = _validated_xy(X_early, y_early)
    x_val, y_val = _validated_xy(X_val, y_val)
    if x.shape[1] != x_val.shape[1]:
        raise ValueError("Inner TRAIN and validation feature columns must match")
    output = Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    if not output.resolve().is_relative_to((ROOT / "data/processed/tabpfn/adaptation").resolve()):
        raise ValueError("Fine-tuning artifacts must stay in the adaptation output directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fine-tuning output is not empty; completed runs are immutable")
    output.mkdir(parents=True, exist_ok=True)
    settings = {"epochs": 5, "max_runtime_seconds": 3600, "learning_rate": 1e-5,
                "meta_batch_size": 256, "n_estimators_finetune": 1,
                "n_estimators_validation": 1, "n_estimators_final": 4, "early_stopping_patience": 2,
                **config.get("finetuning", {})}
    if settings["epochs"] < 1 or settings["max_runtime_seconds"] <= 0 or settings["meta_batch_size"] < 4:
        raise ValueError("Fine-tuning epochs, time limit and meta-batch size must be positive")

    import torch
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier
    from tabpfn.model_loading import save_tabpfn_model

    torch.manual_seed(config["seed"])
    evaluations = []

    class AuditedFineTuner(FinetunedTabPFNClassifier):
        def _evaluate_model(self, *args, **kwargs):
            weights_hash = state_sha256(self.finetuned_estimator_.model_)
            result = super()._evaluate_model(*args, **kwargs)
            if not math.isfinite(result.primary):
                raise ValueError("Official fine-tuning validation returned a non-finite score")
            evaluations.append({"evaluation": len(evaluations), "weights_sha256": weights_hash,
                                "primary_negative_log_loss": float(result.primary),
                                **{k: float(v) for k, v in result.secondary.items()}})
            _write_json(output / "validation_history.json", evaluations)
            return result

    history = _History(output)
    model = AuditedFineTuner(
        device=config["model"]["device"], model_version=ModelVersion.V2_5,
        epochs=settings["epochs"], time_limit=settings["max_runtime_seconds"],
        learning_rate=settings["learning_rate"], validation_split_ratio=None,
        n_finetune_ctx_plus_query_samples=settings["meta_batch_size"],
        finetune_ctx_query_split_ratio=0.2, n_inference_subsample_samples=None,
        random_state=config["seed"], early_stopping=True,
        early_stopping_patience=settings["early_stopping_patience"], min_delta=0.0,
        use_lr_scheduler=False, n_estimators_finetune=settings["n_estimators_finetune"],
        n_estimators_validation=settings["n_estimators_validation"],
        n_estimators_final_inference=settings["n_estimators_final"],
        use_activation_checkpointing=True, save_checkpoint_interval=None,
        experiment_logger=history, eval_metric="log_loss",
        extra_classifier_kwargs={"model_path": pretrained, "n_preprocessing_jobs": 1,
                                 "memory_saving_mode": "auto",
                                 "inference_config": {"SUBSAMPLE_SAMPLES": None}},
    )
    started = time.monotonic()
    model.fit(x, y, X_val=x_val, y_val=y_val, output_dir=output)
    selected_hash = state_sha256(model.finetuned_estimator_.model_)
    if not history.steps or len(evaluations) < 2:
        raise ValueError("Fine-tuning did not execute and evaluate a gradient-update epoch")
    changed = any(row["weights_sha256"] != evaluations[0]["weights_sha256"] for row in evaluations[1:])
    if not changed:
        raise ValueError("Fine-tuning ran without changing any model weights")
    checkpoint = output / "selected.ckpt"
    save_tabpfn_model(model.finetuned_estimator_, checkpoint)
    result = {"status": "complete", "tool": "tabpfn.finetuning.FinetunedTabPFNClassifier",
              "runtime_seconds": time.monotonic() - started, "settings": settings,
              "selection_metric": "log_loss", "validation_scope": "later TRAIN only",
              "gradient_rows": len(x), "validation_rows": len(x_val), "features": x.shape[1],
              "gradient_features_sha256": array_sha256(x), "gradient_labels_sha256": array_sha256(y),
              "validation_features_sha256": array_sha256(x_val), "validation_labels_sha256": array_sha256(y_val),
              "optimizer_steps": len(history.steps), "epochs_completed": len(history.epochs),
              "weights_changed_during_training": changed,
              "weights_before_sha256": evaluations[0]["weights_sha256"],
              "weights_after_sha256": selected_hash,
              "selected_pretrained_weights": selected_hash == evaluations[0]["weights_sha256"],
              "checkpoint_path": str(Path("data") / checkpoint.resolve().relative_to((ROOT / "data").resolve())),
              "checkpoint_sha256": file_sha256(checkpoint),
              "initial_inner_log_loss": evaluations[0]["log_loss"],
              "best_inner_log_loss": min(e["log_loss"] for e in evaluations),
              "evaluations": evaluations}
    _write_json(output / "finetuning.json", result)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def classifier_from_checkpoint(config, checkpoint_path):
    """Load selected adapted weights with the baseline inference configuration."""
    _prepare_environment(config)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, model_path=Path(checkpoint_path),
        n_estimators=config["model"]["n_estimators"], device=config["model"]["device"],
        random_state=config["seed"], fit_mode="fit_preprocessors", n_preprocessing_jobs=1,
        memory_saving_mode="auto", inference_config={"SUBSAMPLE_SAMPLES": None})
