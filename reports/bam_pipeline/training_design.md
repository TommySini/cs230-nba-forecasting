# BAM training worker contract

Implemented in `scripts/bam_training.py`. This is an executable training and evaluation engine, not a claim that GPU training or a new experiment has completed. Keep artifacts under a new run directory. Do not overwrite the earlier researched-forecaster pilot.

## Runtime and checkpoint

The worker imports only Python standard-library modules until a numerical or GPU function is called. Local boundary tests use the project `.venv`. CUDA workers use a dedicated Python 3.12 environment and `requirements-bam-training.txt`, separately from the existing MLX/vLLM environments. Never install these dependencies into system Python or the local MLX environment.

Load the official Hugging Face model using `model_id` and an immutable 40-character `model_revision`; the original MLX four-bit checkpoint is not silently reused or relabeled. The configured `minimum_eval_date` applies to every heldout research and forecast input. Pinning a historical checkpoint and date is a protocol safeguard, not proof of absence of memorization. Research and forecast adapters start independently from this same base. Research weights are never loaded into the frozen reader or forecasting adapter.

Training follows the documented [PEFT quantization procedure](https://huggingface.co/docs/peft/developer_guides/quantization): bitsandbytes NF4 with double quantization, BF16 compute, preparation for k-bit training, and separate LoRA parameters. The fixed remote stack uses Transformers 4.51.3 and PEFT 0.15.2. It needs a real CUDA smoke run before a full launch; the CPU tests do not certify CUDA compatibility or memory capacity. A single H100 80GB is the intended starting device for 14B. Maximum sequence length and generated-token allowance are enforced without silent truncation.

## Files supplied by the orchestrator

All paths below are relative to `run_dir`:

| File | Required content |
| --- | --- |
| `research_examples.jsonl` | Accepted train-only targets: `game_id`, ISO `game_date_et`, `split`, `module`, optional `unit_id`, `messages` exactly system/user/assistant, nonempty `source_hashes`, `teacher_hash`, `review_status="accepted"` |
| `research_inputs.jsonl` | Outcome-blind train/dev/test requests, same identity/source metadata, messages exactly system/user; no target, recap, or outcome fields |
| `prediction_inputs.jsonl` | One forecast-eligible row per game: identity/date/split, original `p_boost`, system/user messages containing assembled student research and booster information, `booster_provenance`, `research_provenance` |
| `labels_train.jsonl` | Exactly the forecast train games, `{game_id, home_win}` with actual binary outcomes |
| `labels_dev.jsonl` | Exactly the forecast dev games, same schema |
| `labels_test.jsonl` | Exactly the forecast test games; opened only after test predictions have been saved |
| `forecast_explanations_train.jsonl` | Optional, required only for positive explanation loss: `{game_id, explanation, review_status:"accepted"}` for exactly the forecast train games |

`booster_provenance.training_max_date` must precede the game date. The original booster must actually have been fit on prior-only data; the data builder owns that audit and exact baseline-game matching. `research_provenance` is one object or a list of component objects copied from generated module rows. Each has `mode`, `training_game_ids`, `training_max_date`, `adapter_sha256` and base-model identity. Non-cold provenance requires a nonempty training pool, adapter hash, and strictly earlier latest training date. A target game may never appear in its own researcher's training pool. Research-only examples can train the research adapter; they cannot enter the outcome forecaster.

The validator enforces machine-checkable boundaries, not semantic truth of source text or reports. The source/teacher/reviewer pipeline must validate timestamps, citations, hindsight removal, table meaning, and substantive research quality before marking a target accepted. Generated reports need the independent source-review gate before the orchestrator assembles forecasting inputs. A hash cannot certify correctness.

## Worker stages

```text
python scripts/bam_training.py --stage STAGE --run-dir RUN --config CONFIG
```

The equivalent functions accept `(run_dir, config)`:

1. `generate_research` (`research-generate`) reads research inputs and gold, divides training dates into a fixed number of contiguous blocks, and fits each researcher only on accepted gold with strictly earlier dates. All same-game modules remain together. The first block uses the base prompted model when no earlier gold is available. Earlier research-only games can supervise a fold even without a booster forecast. A final independent adapter is fit on all allowed training gold only for dev/test research. Generated modules and adapter lineage are saved to `generated_modules.jsonl`. Fixed-epoch research SFT uses no dev/test outcomes and no outcome-based checkpoint selection.
2. `train_research` (`research-train`) is an optional standalone final training call, returning the same final artifact used above. The normal expanding-fold stage does not require this to run first.
3. The orchestrator independently reviews student modules and assembles `prediction_inputs.jsonl`. It must not substitute hindsight gold for student output. Generated module content is not itself a validated forecast input merely because generation completed.
4. `encode_reports` (`encode`) uses the frozen base reader, the final pregame-prefix token, and unit-normalized features. It saves hashed features in `reader/`; outcome labels are not read.
5. `train_forecaster` (`forecast-train`) fits the frozen control with binary log loss plus L2. Development Brier selects regularization, with an exact zero-correction candidate first. It separately trains forecasting LoRA and a zero-initialized linear residual head using actual-outcome Brier. Optional reviewed explanation completion CE updates the forecasting LoRA; the probability head sees a separate pregame-prefix forward pass before any teacher-forced explanation. Epoch zero competes with every training epoch, and dev Brier alone selects a checkpoint. No refit on dev. Train and dev labels are physically separate; test labels are never opened in this stage. `forecast/selection.json` freezes chosen parameters and hashes.
6. `evaluate` (`evaluate`) verifies selection and artifact hashes, locks evaluation, saves predictions, then opens test outcomes. It compares original boosting and both selected correction recipes on exactly the same games. Repeat calls return the saved result, not another test evaluation. An interrupted evaluation requires inspection rather than automatic rerunning. Once evaluation starts, fitting and reselection are rejected.

Set `research.mode="prompted"` in a **separate run directory** to create a matched prompted-research control under the same frozen-reader/head protocol. The trained and prompted arms must share game cohorts, sources, rubric, decoder settings and forecast selection budgets. The default mode is `trained`. A separate final comparison plan should specify which arm is the primary hypothesis; there is no automatic choice among arms using test outcomes.

## Configuration

Shared fields: `model_id`, `model_revision`, `minimum_eval_date`, optional `cache_dir`, and `seed` (default 230).

Research fields: `epochs`, `folds`, `learning_rate`, `lora_rank`, `lora_alpha`, `lora_dropout`, `gradient_accumulation_steps`, `max_seq_length`, `max_new_tokens`, `generation_batch_size` (default 2), and optional `mode`. Training uses one complete example per microbatch with accumulated gradients. Generation uses left-padded batches with attention masks. Use batch size 1 for an initial maximum-context smoke test; measure throughput and memory before enlarging it. There is no unmeasured claim about full-run duration.

Forecast fields: `epochs`, `learning_rate` (adapter), `head_learning_rate`, `lora_rank`, `lora_alpha`, `lora_dropout`, `gradient_accumulation_steps`, `max_seq_length`, `explanation_loss_weight`, `brier_loss_weight`, `train_lora` (default true), and `l2_candidates` (default `[0.001, 0.01, 0.1]`). The zero-correction candidate is always included; no setting disables it. `refit_on_dev` is always false. Do not enlarge the search after observing results from the very small dev cohort.

Evaluation fields: `status`, `bootstrap_replicates`, and `bootstrap_seed`. The report includes unweighted Brier, binary log loss, accuracy, mean predicted probability, observed home-win rate, and paired date-bootstrap confidence intervals for Brier differences. The bootstrap samples whole dates jointly and conditions on fitted models/dev selection; it does not quantify training uncertainty. Existing previously inspected test games remain exploratory.

## Resume and verification

Research fit requests lock config, input-row hashes and worker hash. Saved adapters are rehashed before reuse; generated folds also have content manifests. Changed config/data/code requires a new run directory. Partial research training can be restarted from the pinned base, but does not claim optimizer-state continuation. The downstream selection and test lock prevent post-test reselection.

CPU verification: `.venv/bin/python -m unittest discover -s tests -p test_bam_training.py -v` checks date/game grouping, earlier-only research folds, research-only supervision, cold starts, target acceptance, completion-only token masking, absence of explanation tokens from the prefix representation, zero-correction selection, exact outcome joins, shared-date bootstrap units, and a complete frozen-head fit/evaluation where the test outcome file does not exist until fitting is finished.

## Resumable driver and student repairs

`run_bam_pipeline.py` connects the immutable snapshot, normalized training-only postgame records, research builders, deployed Modal workers and training engine. Its stages are `prepare-research`, `selection`, `modules`, `gold`, `export`, `research-train`, `review-student`, `forecast` and `evaluate`; `all` executes them in order. It writes only a new `data/processed/bam_runs/<run-id>` and `reports/bam_pipeline/<run-id>`. The protocol freezes the configuration, game cohort, source manifests, worker hashes, model-specific tokenizer files and predeclared primary forecast arm. Keep command-line batch settings and explicit reference-game subsets identical when resuming the same run.

Each cloud call is spawned once, its durable call ID is immediately saved, and the driver waits for completion. Restarting recovers the same call or its checksummed result. `--retry-failed` starts a bounded new invocation only for inspected cloud-execution failures, preserving previous attempts. Completed source/page and module requests are never silently skipped. Model-specific context limits include any explicit reasoning-token reserve; local and remote prompt token hashes must match.

`--game-ids` is an explicit training-reference subset and cannot run a heldout forecast evaluation. A full experiment must use its own complete immutable cohort. Before research SFT, the run needs a manually authored `quality_gate.json` with `ready_for_training: true`, `reviewer`, `reference_game_ids` (all training games), and substantive `notes`. This gate records factual source/teacher review, not acceptance based on heldout outcomes. Automated agreement is not factual certification.

Rejected actual student reports receive at most two source-only repair attempts. The `research-repair` worker accepts an execution-only `research.repair_batch_id` and the corresponding `student_repair_requests/<batch-id>.jsonl`. It verifies the original source prompt, preserved previous output and full chronological adapter lineage, then loads the exact original fold/final adapter by hash. A cold initial researcher stays the same cold base model. It never substitutes a teacher target or a differently trained researcher.

Every repair records its parent-output hash, source-review feedback hash, actual repair-prompt/token hashes and attempt number. Initial `generated_modules.jsonl` remains unchanged. Separate repair outputs are independently reviewed again, and the selected student generation is recorded per game. Exhausting the repair budget leaves a failed run with all evidence retained. Gold reports and normalized postmortems remain absent from repair prompts and forecasting inputs.

The downstream language-model prompt contains an explicit allowlist of game metadata, original booster probability, prior-game statistical features, and compact reviewed module reports with citation IDs. Research training-game IDs, fold identifiers, adapter hashes, duplicate quotations and other audit lineage stay outside the model messages. Test labels are first transferred in the evaluation stage, after selected parameters and the primary comparison are locked.
