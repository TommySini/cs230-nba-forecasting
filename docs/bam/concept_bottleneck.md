# Concept-bottleneck forecaster head

Status 2026-11-09: implemented and tested on synthetic tensors (CPU, tiny model). Not yet
run on the real model or on NBA data. This replaces the scalar linear head for the TRAIN
pilot when `config/bam_train_only_concept_pilot.json` is used. The original linear head is
unchanged and still selected by `config/bam_train_only_pilot.json`.

## Model

```
h            = normalized last pregame hidden state of Qwen2.5-14B + LoRA (5,120 numbers)
c            = tanh(W_c h)          20 named concept scores in [-1, 1]  (net home advantage)
e            = W_e h                20 logits: does pregame evidence address the concept?
delta        = MLP([logit p_boost, c, sigmoid(e)])     41 -> 64 -> 32 -> 1, GELU, dropout 0.1
p_hat        = sigmoid(logit p_boost + delta)
```

The MLP's last layer starts at exactly zero, so the untrained model reproduces boosting;
the worker refuses to start otherwise. Gradients from the outcome reach the LoRA adapters
through the MLP and the concept layer once that last layer moves (from step 2); the concept
loss reaches them from step 1. Code: `scripts/bam_concepts.py` (head, labels, taxonomy),
`scripts/bam_train_only.py` (training), `scripts/bam_train_contract.py` (bundle checks).

## Concepts

`config/bam_concepts.json` defines 24 candidates (availability, rotation, form, notes
sentiment, matchup edges, rest, travel, schedule, motivation, coaching and so on) and the 20
selected for the head. Each score is the net pregame advantage for the home team on that
aspect. The pilot uses a default selection fixed before any teacher output.

The teacher labels all 24 candidates for every TRAIN game, including a hindsight `decisive`
flag. Before any DEV work, re-select the 20 concepts most often decisive *and* evidenced in
TRAIN labels, with no further teacher calls:

```bash
.venv/bin/python scripts/bam_concepts.py select --labels <bundle>/concepts_train.jsonl \
    --k 20 --output config/bam_concepts_v2.json
.venv/bin/python scripts/bam_concepts.py hash --taxonomy config/bam_concepts_v2.json
```

Pin the printed `taxonomy_sha256` in the forecaster config. The worker rejects a taxonomy
file that differs from its pin.

## Concept labels (TRAIN only)

Labels are a fourth teacher task, derived from each game's accepted gold teacher request
(`bam_concepts.build_concept_request`): the same frozen pregame evidence, researcher
modules and TRAIN postmortem, plus the concept definitions. For every concept the teacher
returns a score, `evidence_present`, `decisive`, citations and a short rationale.
`ingest_concept_response` accepts a response only if every candidate appears once, scores
lie in [-1, 1], evidenced scores cite real pregame source lines, and unevidenced ones are 0
with no citations. Hindsight may set salience and `decisive`; it may not set a score that
pregame evidence does not support. Labels never enter the forecaster's input.

The API boundary (`bam_api_teacher.validate_training_request`) admits the `concept_labels`
stage only for manifest-verified TRAIN games with an authorized request.

## Bundle schema 2

A concept bundle adds `concepts_train.jsonl` and `concept_pool_sha256` to the manifest
(`"schema_version": 2`). Each record holds the request and the teacher candidate. The
contract rebuilds the request from the accepted gold and re-ingests the raw response, so a
changed label, a different concept pool, labels predating their gold, or a pilot game
without labels all fail before the model loads.

## Objective and learning criterion

```
loss = mean_N(Brier) + lambda * mean_M(gold CE) + mu * mean_C(masked concept MSE + evidence BCE)
```

The concept MSE counts only concepts with pregame evidence. The pilot pins `mu = 1.0` and
requires, on the TRAIN pilot games: Brier down at least 50%, gold CE down 10%, concept loss
down 30%, both head and adapters updated, and a nonzero Brier-only adapter gradient after
step 1 (`brier_probe_adapter_gradient_norm`, measured on one game per step). This is a
proof that the system can learn, not a generalization claim.

## DEV phase (after the TRAIN pilot proves learning)

`bam_training.py --stage forecast-train` (fit on TRAIN, choose the epoch by DEV Brier) and
`--stage evaluate` (score TEST once) use the same head. Set `forecast.head` to the concept
spec above and `forecast.concept_loss_weight` (> 0) in that config, and place
`forecast_concepts_train.jsonl` in the run directory beside
`forecast_explanations_train.jsonl`. Produce it from contract-validated bundle records with
`bam_concepts.forecast_label_rows`; the run lock pins its hash. Concept labels are TRAIN only.
DEV and TEST games are scored on Brier with no concept targets, and evaluation rebuilds the
configured head before loading `head.pt`. With the default linear head, both stages behave
exactly as before.

## Running it on AWS

Prerequisites outside the code: EC2 GPU quota (planned `g6e.2xlarge`, one L40S 48 GB),
Bedrock access for an independent teacher and reviewer, a funded budget ledger and
preflight. Use the Linux `training` dependency group.

1. **Synthetic GPU smoke (first, cheap).** Loads the real pinned model and runs the concept
   head for two steps on synthetic prompts at 80% of `max_seq_length`. No bundle, no NBA data.

   ```bash
   .venv/bin/python scripts/bam_train_only.py --smoke \
       --config config/bam_train_only_concept_pilot.json --output runs/concept-smoke
   ```

   Read `runs/concept-smoke/smoke/smoke_result.json`: confirm peak memory fits, then use
   `projected_pilot_seconds_per_step_32_games` to set `max_runtime_seconds` and the preflight
   cost bound before admitting the pilot.
2. **Select 16–32 TRAIN games** with approved notes and boosting forecasts, spread across
   publishers, teams, packet lengths and probabilities (`docs/bam_aws_architecture.md`).
3. **Frozen research** on the GPU: selection, module and review requests from
   `bam_research`, generated with `bam_inference` (vLLM, `research` group).
4. **Teacher tasks** for each game: postmortem, gold, independent gold review, then concept
   labels from the accepted gold request.
5. **Assemble the schema-2 bundle** (`manifest.json`, `inputs_train.jsonl`,
   `outcomes_train.jsonl`, `gold_train.jsonl`, `concepts_train.jsonl`) and record the
   manifest SHA256 in the approved preparation record.
6. **Preflight and admission** (`bam_aws_preflight.py`), then train:

   ```bash
   .venv/bin/python scripts/bam_train_only.py --config config/bam_train_only_concept_pilot.json \
       --bundle <bundle> --manifest-sha256 <sha> --output runs/concept-pilot \
       --admission <receipt.json> --preflight-report <preflight.json> --job-id <id>
   ```

7. **Read** `result_train.json` (`learning_criterion_met`, the three reductions) and
   `learning_curve_train.json`. `head.pt`, `head_config.json` and the adapter are saved
   with hashes in `model_files.json`.

## Not built yet

These gaps predate the concept head and apply to the linear head too:

- **Bedrock sender:** request builders and the API boundary exist, but no code submits
  requests to Bedrock or records responses and costs (steps 3–4 teacher side).
- **Pilot assembler:** nothing yet selects games and writes pregame rows with their full
  provenance (booster, features, research receipts) into a bundle (steps 2 and 5).
- **EC2 launcher:** no code launches the GPU instance, issues the admission receipt or arms
  external termination; the worker only validates a receipt.
- **Teacher choice:** no teacher or reviewer model has been qualified.
