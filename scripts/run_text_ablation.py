"""Matched compact-LM statistics / statistics-plus-injury-text experiment."""
import argparse
from collections import Counter
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random

import numpy as np

from collect_pilot import parse_injury, stamp, write_json, write_jsonl
from run_gradient_boosting import (FEATURE_NAMES, ROOT, load_verified_history,
                                   prepare_examples, read_rows, score_predictions,
                                   sha256, write_csv)

CONFIG_PATH = ROOT / "config/text_ablation.json"
DATA_DIR = ROOT / "data/processed/text_ablation"
REPORT_DIR = ROOT / "reports/text_ablation"
ARMS = ("stats_only", "stats_plus_text")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def assign_splits(records, counts):
    ordered = sorted(records, key=lambda r: (r["game_date_et"], r["scheduled_start_utc"], r["game_id"]))
    if sum(counts.values()) != len(ordered) or set(counts) != {"train", "dev", "test"}:
        raise ValueError("Split counts must assign every game once")
    result, offset = [], 0
    for split in ("train", "dev", "test"):
        end = offset + counts[split]
        if counts[split] <= 0 or (end < len(ordered) and ordered[end - 1]["game_date_et"] == ordered[end]["game_date_et"]):
            raise ValueError("A chronological split cannot divide a calendar date")
        result.extend(dict(row, split=split) for row in ordered[offset:end])
        offset = end
    return result


def verify_injury(record, numeric, parsed_cache):
    injury = record["injury_report"]
    if not injury or not injury["text"].strip() or injury["not_yet_submitted"]:
        raise ValueError("Expected a verified, submitted injury excerpt")
    if record["narrative_text"] is not None:
        raise ValueError("Unapproved narratives cannot enter this experiment")
    for key in ("game_date_et", "scheduled_start_utc", "cutoff_utc", "home_abbr", "away_abbr"):
        if record[key] != numeric[key]:
            raise ValueError(f"Audit / numeric mismatch: {key}")
    if stamp(injury["report_time_utc"]) > stamp(record["cutoff_utc"]):
        raise ValueError("Post-cutoff report")
    if injury["matchup"] != record["away_abbr"] + "@" + record["home_abbr"] or injury["game_date_et"] != record["game_date_et"]:
        raise ValueError("Wrong injury matchup/date")
    source = injury["source"]
    path = ROOT / source["raw_path"]
    if source["status"] != 200 or sha256(path) != source["sha256"]:
        raise ValueError("Injury source checksum mismatch")
    if source["sha256"] not in parsed_cache:
        parsed_cache[source["sha256"]] = parse_injury(path.read_bytes())
    timestamp, blocks = parsed_cache[source["sha256"]]
    matches = [b for b in blocks if b["game_date_et"] == injury["game_date_et"] and b["matchup"] == injury["matchup"]]
    if timestamp != injury["report_time_utc"] or len(matches) != 1 or matches[0]["text"] != injury["text"]:
        raise ValueError("Cached excerpt differs from source PDF")
    text = injury["text"]
    aliases = {record["home_team"]: "HOME", record["away_team"]: "AWAY"}
    for side in ("home", "away"):
        if record[side + "_abbr"] == "LAC":
            aliases["Los Angeles Clippers"] = side.upper()
    for name in sorted(aliases, key=len, reverse=True):
        text = text.replace(name, aliases[name])
    return text


def make_cohort(config):
    base = json.loads((ROOT / config["baseline_config"]).read_text())
    numeric, labels, _ = prepare_examples(load_verified_history(base), base)
    numeric = {r["game_id"]: r for r in numeric}
    labels = {r["game_id"]: r for r in labels}
    audit = [r for name in config["audit_inputs"] for r in read_rows(ROOT / name)]
    if len({r["game_id"] for r in audit}) != len(audit):
        raise ValueError("Duplicate audited games")
    records, parsed_cache = [], {}
    for row in audit:
        if row["game_id"] not in numeric:
            continue
        source = numeric[row["game_id"]]
        text = verify_injury(row, source, parsed_cache)
        records.append({**source, "pregame_text": text, "injury_report": row["injury_report"]})
    if len(records) != config["expected_games"]:
        raise ValueError("Unexpected paired cohort size")
    records = assign_splits(records, config["split_counts"])
    return records, [labels[r["game_id"]] for r in records]


def prompt_for(record, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    # Strict allowlist: no game IDs, dates, outcomes, provenance or raw feeds.
    stats = "\n".join(f"{name}: {record['features'][name]:.6g}" for name in FEATURE_NAMES)
    text = record["pregame_text"] if arm == "stats_plus_text" else "[withheld]"
    return ("Predict whether HOME wins this NBA game using only pregame information.\n"
            "Statistics (difference means HOME minus AWAY):\n" + stats +
            "\nPregame injury report (team headings HOME/AWAY; missing rows do not imply healthy):\n" + text +
            "\nEnd of pregame information. Forecast HOME win probability:")


def source_hashes(config, records):
    base = json.loads((ROOT / config["baseline_config"]).read_text())
    names = ["config/text_ablation.json", "scripts/run_text_ablation.py", "requirements-text.txt",
             config["baseline_config"], base["history_path"], base["history_config"],
             "scripts/run_gradient_boosting.py", "scripts/collect_pilot.py", *config["audit_inputs"]]
    names += [r["injury_report"]["source"]["raw_path"] for r in records]
    return {name: sha256(ROOT / name) for name in sorted(set(names))}


def hashes_in(directory):
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and ".cache" not in p.parts}


def prepare(config):
    records, labels = make_cohort(config)
    write_jsonl(DATA_DIR / "inputs.jsonl", records)
    write_jsonl(DATA_DIR / "labels.jsonl", labels)
    write_csv(DATA_DIR / "split_manifest.csv", [{k: r[k] for k in ("game_id", "game_date_et", "split")} for r in records])
    summaries = {}
    for split in ("train", "dev", "test"):
        subset = [r for r in records if r["split"] == split]
        summaries[split] = {"games": len(subset), "first_date": subset[0]["game_date_et"], "last_date": subset[-1]["game_date_et"]}
    if config["model_revision_date"] >= summaries["dev"]["first_date"]:
        raise ValueError("Backbone snapshot must predate dev/test")
    original = hashes_in(ROOT / "reports/gradient_boosting") | hashes_in(ROOT / "data/processed/gradient_boosting")
    protocol = {"config": config, "source_sha256": source_hashes(config, records),
                "checkpoint_sha256": hashes_in(ROOT / config["model_path"]),
                "original_baseline_sha256": original, "splits": summaries,
                "inputs_sha256": sha256(DATA_DIR / "inputs.jsonl"), "labels_sha256": sha256(DATA_DIR / "labels.jsonl"),
                "test_status": "Exploratory: these games were already inspected in prior baseline evaluations; new split is not fresh data."}
    if not any(p.endswith(".safetensors") for p in protocol["checkpoint_sha256"]):
        raise ValueError("Download the pinned model before preparing")
    write_json(REPORT_DIR / "protocol.json", protocol)
    print(json.dumps(summaries, indent=2), flush=True)


def load_prepared(config):
    protocol = json.loads((REPORT_DIR / "protocol.json").read_text())
    if protocol["config"] != config:
        raise ValueError("Configuration changed after preparation")
    for group in ("source_sha256", "checkpoint_sha256", "original_baseline_sha256"):
        for name, expected in protocol[group].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Changed frozen source/artifact: {name}")
    for name in ("inputs", "labels"):
        if sha256(DATA_DIR / f"{name}.jsonl") != protocol[name + "_sha256"]:
            raise ValueError(f"Changed {name}")
    records, labels = read_rows(DATA_DIR / "inputs.jsonl"), read_rows(DATA_DIR / "labels.jsonl")
    if [r["game_id"] for r in records] != [r["game_id"] for r in labels]:
        raise ValueError("Misaligned labels")
    return protocol, records, labels


def get_runtime(config, device_name):
    import torch
    from transformers import AutoTokenizer
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device_name = ("mps" if torch.backends.mps.is_available() else "cpu") if device_name == "auto" else device_name
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model_path"], local_files_only=True)
    tokenizer.padding_side = "right"
    return torch.device(device_name), tokenizer


def make_model(config, tokenizer, adapter_path=None):
    import torch
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    torch.manual_seed(config["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(
        ROOT / config["model_path"], num_labels=1, dtype=torch.float32,
        local_files_only=True, attn_implementation="sdpa")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    with torch.no_grad():
        model.score.weight.zero_()
    if adapter_path is not None:
        return PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                      lora_dropout=config["lora_dropout"], target_modules=config["lora_targets"], modules_to_save=["score"])
    return get_peft_model(model, lora)


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def encode(records, arm, tokenizer, config):
    result = tokenizer([prompt_for(r, arm) for r in records], truncation=False, padding=False)
    if max(map(len, result["input_ids"])) > config["max_tokens"]:
        raise ValueError("Prompt exceeds token budget; do not silently truncate evidence")
    return [{key: value[i] for key, value in result.items()} for i in range(len(records))]


def batch_inputs(encoded, indices, tokenizer, device):
    return tokenizer.pad([encoded[i] for i in indices], padding=True, return_tensors="pt").to(device)


def predict(model, encoded, tokenizer, device, batch_size):
    import torch
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = batch_inputs(encoded, list(range(start, min(start + batch_size, len(encoded)))), tokenizer, device)
            probabilities.extend(torch.sigmoid(model(**batch).logits[:, 0]).cpu().tolist())
    return np.asarray(probabilities)


def select_shared_epoch(history):
    epochs = sorted({r["epoch"] for r in history})
    scores = []
    for epoch in epochs:
        rows = [r for r in history if r["epoch"] == epoch]
        if sorted(r["arm"] for r in rows) != sorted(ARMS):
            raise ValueError("Every epoch needs both matched arms")
        scores.append({"epoch": epoch, "mean_dev_brier": float(np.mean([r["dev_brier"] for r in rows]))})
    return min(scores, key=lambda r: (r["mean_dev_brier"], r["epoch"])), scores


def train(config, device_name):
    import torch
    _, records, labels = load_prepared(config)
    device, tokenizer = get_runtime(config, device_name)
    subsets = {s: [i for i, r in enumerate(records) if r["split"] == s] for s in ("train", "dev")}
    train_records, dev_records = ([records[i] for i in subsets[s]] for s in ("train", "dev"))
    y_train, y_dev = (np.asarray([labels[i]["home_win"] for i in subsets[s]]) for s in ("train", "dev"))
    orders = [np.random.default_rng(config["seed"] + epoch).permutation(len(train_records)).tolist() for epoch in range(1, config["epochs"] + 1)]
    history, initial, details, dev_predictions = [], {}, {}, []
    for arm in ARMS:
        model = make_model(config, tokenizer)
        initial[arm] = trainable_digest(model)
        details[arm] = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "total_parameters": sum(p.numel() for p in model.parameters())}
        if len(set(initial.values())) != 1:
            raise ValueError("The two arms must start from identical trainable parameters")
        model.to(device)
        encoded_train, encoded_dev = encode(train_records, arm, tokenizer, config), encode(dev_records, arm, tokenizer, config)
        groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and "score" not in n], "lr": config["lora_learning_rate"]},
                  {"params": [p for n, p in model.named_parameters() if p.requires_grad and "score" in n], "lr": config["head_learning_rate"]}]
        optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
        train_y = torch.tensor(y_train, dtype=torch.float32, device=device)
        print(f"Training {arm}: {len(train_records)} games, {device}, {details[arm]['trainable_parameters']} trainable parameters", flush=True)
        for epoch, order in enumerate(orders, 1):
            model.train()
            batches = [order[i:i + config["batch_size"]] for i in range(0, len(order), config["batch_size"])]
            total_loss = 0.0
            for start in range(0, len(batches), config["gradient_accumulation_steps"]):
                group = batches[start:start + config["gradient_accumulation_steps"]]
                count = sum(map(len, group))
                optimizer.zero_grad(set_to_none=True)
                for indices in group:
                    inputs = batch_inputs(encoded_train, indices, tokenizer, device)
                    logits = model(**inputs).logits[:, 0]
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y[indices], reduction="sum")
                    total_loss += loss.item()
                    (loss / count).backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["max_gradient_norm"])
                optimizer.step()
            probabilities = predict(model, encoded_dev, tokenizer, device, config["batch_size"])
            score = score_predictions(y_dev, probabilities)
            history.append({"arm": arm, "epoch": epoch, "train_log_loss": total_loss / len(y_train), "dev_brier": score["brier"], "dev_log_loss": score["log_loss"]})
            dev_predictions.extend({"arm": arm, "epoch": epoch, "game_id": row["game_id"], "probability": float(p)} for row, p in zip(dev_records, probabilities))
            model.save_pretrained(DATA_DIR / arm / f"epoch_{epoch}")
            write_csv(REPORT_DIR / "learning_curves.csv", history)
            print(json.dumps(history[-1]), flush=True)
        if trainable_digest(model) == initial[arm]:
            raise ValueError("Training did not change any trainable weights")
        del model, optimizer, train_y, groups
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    choice, epoch_scores = select_shared_epoch(history)
    write_csv(DATA_DIR / "dev_predictions.csv", dev_predictions)
    selection = {"protocol_sha256": sha256(REPORT_DIR / "protocol.json"), "selected_epoch": choice["epoch"],
                 "selection_scores": epoch_scores, "initial_trainable_sha256": initial, "parameters": details,
                 "train_game_ids": [r["game_id"] for r in train_records], "dev_game_ids": [r["game_id"] for r in dev_records],
                 "training_order_game_ids": [[train_records[i]["game_id"] for i in order] for order in orders],
                 "device": str(device), "package_versions": {name: version(name) for name in ("torch", "transformers", "peft", "numpy")},
                 "checkpoint_sha256": {arm: hashes_in(DATA_DIR / arm / f"epoch_{choice['epoch']}") for arm in ARMS},
                 "learning_curves_sha256": sha256(REPORT_DIR / "learning_curves.csv"),
                 "dev_predictions_sha256": sha256(DATA_DIR / "dev_predictions.csv")}
    write_json(REPORT_DIR / "selection.json", selection)
    print(f"Locked common epoch {choice['epoch']} using dev only; no test evaluation yet.", flush=True)


def paired_brier_interval(y, stats, text, dates, config):
    _, inverse = np.unique(dates, return_inverse=True)
    count = np.bincount(inverse)
    delta = (text - y) ** 2 - (stats - y) ** 2
    sums = np.bincount(inverse, weights=delta)
    sampled = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(count), size=(config["bootstrap_replicates"], len(count)))
    distribution = sums[sampled].sum(axis=1) / count[sampled].sum(axis=1)
    return {"text_minus_stats": float(delta.mean()), "paired_date_bootstrap_ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
            "test_dates": len(count), "replicates": config["bootstrap_replicates"], "conditional_on": "fitted models and sampled dates; no training/selection uncertainty"}


def verify_selection(selection):
    if selection["protocol_sha256"] != sha256(REPORT_DIR / "protocol.json"):
        raise ValueError("Protocol changed after training")
    for group in selection["checkpoint_sha256"].values():
        for path, expected in group.items():
            if sha256(ROOT / path) != expected:
                raise ValueError("Selected weights changed")
    for filename, key in ((REPORT_DIR / "learning_curves.csv", "learning_curves_sha256"),
                          (DATA_DIR / "dev_predictions.csv", "dev_predictions_sha256")):
        if sha256(filename) != selection[key]:
            raise ValueError("Dev selection artifact changed")


def evaluate(config, device_name):
    import joblib
    import torch
    protocol, records, labels = load_prepared(config)
    selection = json.loads((REPORT_DIR / "selection.json").read_text())
    verify_selection(selection)
    device, tokenizer = get_runtime(config, device_name)
    indices = [i for i, r in enumerate(records) if r["split"] == "test"]
    test = [records[i] for i in indices]
    y = np.asarray([labels[i]["home_win"] for i in indices])
    predictions = {}
    for arm in ARMS:
        model = make_model(config, tokenizer, DATA_DIR / arm / f"epoch_{selection['selected_epoch']}").to(device)
        predictions[arm] = predict(model, encode(test, arm, tokenizer, config), tokenizer, device, config["batch_size"])
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    original = joblib.load(ROOT / "data/processed/gradient_boosting/model.joblib")
    assert not set(original["fit_game_ids"]) & {r["game_id"] for r in test}
    predictions["original_boosting_reference"] = original["model"].predict_proba(np.array([[r["features"][n] for n in FEATURE_NAMES] for r in test]))[:, 1]
    rows = [{"game_id": row["game_id"], "game_date_et": row["game_date_et"], "home_win": int(y[i]),
             **{arm: float(p[i]) for arm, p in predictions.items()}} for i, row in enumerate(test)]
    write_csv(DATA_DIR / "test_predictions.csv", rows)
    result = {"protocol_sha256": sha256(REPORT_DIR / "protocol.json"), "selection_sha256": sha256(REPORT_DIR / "selection.json"),
              "selected_epoch": selection["selected_epoch"], "test": {arm: score_predictions(y, p) for arm, p in predictions.items()},
              "paired_comparison": paired_brier_interval(y, predictions[ARMS[0]], predictions[ARMS[1]], [r["game_date_et"] for r in test], config),
              "test_predictions_sha256": sha256(DATA_DIR / "test_predictions.csv")}
    write_json(REPORT_DIR / "metrics.json", result)
    write_report(config, protocol, selection, result)
    print(json.dumps(result["test"], indent=2), flush=True)
    print(json.dumps(result["paired_comparison"], indent=2), flush=True)


def write_report(config, protocol, selection, result):
    lines = ["# Matched pregame-text pilot", "", "Does verified pregame injury text improve predictions beyond prior-game statistics?", "",
             "## Controlled comparison", "", "Both arms use the same pinned Qwen2.5-0.5B-Instruct backbone, a binary probability head, and LoRA on query/value projections. The head starts at zero, and both arms have identical initial trainable weights, game ordering, optimizer settings and training budgets. Statistics are serialized identically; only the injury-text block changes. Team headings are replaced by HOME/AWAY. Player names remain in the text arm. No game dates, IDs, target outcomes, raw ESPN content or unapproved narratives enter either prompt.", "",
             "This is a compact language-model adaptation of the forecasting idea, not a BAM replication. BAM uses research agents, a fine-tuned GPT-4o forecaster, hindsight-guided report supervision and an ensemble. Here we directly train a binary probability head on game outcomes, with no research agents, generated reasoning targets, market smoothing or ensemble.", "",
             "## Split", "", "The user approved whole-date chronological 108/14/18 instead of an exact 105/17/18 split that would cut through a game date.", "",
             "| Split | Games | First date | Last date |", "| --- | ---: | --- | --- |"]
    for split, info in protocol["splits"].items():
        lines.append(f"| {split} | {info['games']} | {info['first_date']} | {info['last_date']} |")
    lines += ["", f"Checkpoint revision `{config['model_revision']}` is dated {config['model_revision_date']}, before dev/test. The original 40/48/52 counts described the old season split only; this experiment trains new models on the new split. Historical games outside the cohort supply prior-game statistics but are not supervised training examples.", "",
              "## Training and selection", "", f"Five epochs per arm; LoRA rank {config['lora_rank']}, alpha {config['lora_alpha']}, dropout {config['lora_dropout']}; adapter learning rate {config['lora_learning_rate']}, head rate {config['head_learning_rate']}; AdamW weight decay {config['weight_decay']}; batch {config['batch_size']} with accumulation {config['gradient_accumulation_steps']}; seed {config['seed']}. Binary cross-entropy trains both arms. No calibration or threshold search; threshold 0.5. Full prompts must fit {config['max_tokens']} tokens; truncation is rejected.", "",
              f"One common epoch (**{selection['selected_epoch']}**) is selected by the mean dev Brier across the two arms, so both selected models receive the same number of training passes. No refit on dev. `selection.json` locks the chosen weights and dev results before test evaluation.", "",
              "## Results on the identical 18 test games", "", "| Model | Brier ↓ | Log loss ↓ | Accuracy at 0.5 |", "| --- | ---: | ---: | ---: |"]
    for arm, score in result["test"].items():
        lines.append(f"| {arm} | {score['brier']:.6f} | {score['log_loss']:.6f} | {score['accuracy']:.1%} |")
    comparison = result["paired_comparison"]
    lines += ["", f"Paired Brier difference (text minus stats): **{comparison['text_minus_stats']:.6f}**. A negative value favors text. Paired date-bootstrap 95% interval: {comparison['paired_date_bootstrap_ci95']}. This conditions on fitted models and sampled dates; it excludes training and selection uncertainty.", "",
              "The original boosting model is a reference only: it was trained on a different, much larger cohort. Only the two language-model arms form the matched experiment.", "",
              "## Limits", "", "Only 18 test games and 14 dev games; both estimates and epoch selection are unstable. This is one fixed training seed and one small language model. The text consists of injury excerpts, not rich narratives, and missing rows do not prove a healthy roster. Text includes player identity as well as injury information. These test games were inspected in previous baseline evaluations; repartitioning does not make them an untouched final benchmark. A positive or negative result cannot establish whether all pregame text helps in general.", "",
              "## Reproduce", "", "```bash", "uv pip install --python .venv/bin/python -r requirements-text.txt", ".venv/bin/python scripts/run_text_ablation.py --stage prepare", ".venv/bin/python scripts/run_text_ablation.py --stage train --device mps", ".venv/bin/python scripts/run_text_ablation.py --stage evaluate --device mps", ".venv/bin/python scripts/validate_text_ablation.py --device mps", "```", "",
              "The pinned public checkpoint must first be downloaded to the model path in the config. Thereafter all stages are offline. Inputs, separate labels, adapters, split manifest and predictions are in `data/processed/text_ablation/`. The original baseline and raw sources are preserved.", "",
              "Sources: [BAM architecture and training](https://arxiv.org/html/2607.09921v1#S4), [Qwen2.5 release](https://qwenlm.github.io/blog/qwen2.5/), [LoRA/PEFT](https://huggingface.co/docs/peft/en/quicktour).", ""]
    (REPORT_DIR / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate"), required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "train":
        train(config, args.device)
    else:
        evaluate(config, args.device)


if __name__ == "__main__":
    main()
