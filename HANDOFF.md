# CS230 NBA forecasting — handoff

Short current state for the next session. Chronological agent history through
2026-11-01 is archived in [`docs/archive/handoff-through-2026-11-01.md`](docs/archive/handoff-through-2026-11-01.md).
Experiment map: [`docs/experiments.md`](docs/experiments.md). Public summary: [`README.md`](README.md).

## Current status (2026-11-08 UTC)

Latest blocker: AWS MCP search/retrieve now return `Unknown tool`, and CLI
profile `cs230` reports an expired session. Renew sign-in before further account
work; no current CPU quota was obtained. The user's new email limits EC2 launches
to three regions. Earlier 17-region opt-in/quota checks did not establish launch
permission in all 17. Confirm the actual allowed three. The latest pasted denial
has no case ID. CPU-only TRAIN data/booster preparation is a useful next stage;
the GPU appeal now asks whether 4 G-family vCPUs for one g6e.xlarge could be
approved. That smaller machine is unqualified; the production gate remains 8.
The user delegated region/instance choice and wants the fastest useful start.
Ready-to-send messages in `docs/aws_next_support_messages.md` request N. Virginia,
Ohio and Oregon, a reduced 4-vCPU G allocation in existing N. Virginia case
179411421300610, and separate Bedrock account authorization. They are not sent.

**AWS-only TRAIN-first work is active; no real pilot fit has run.** Follow
[`docs/bam_training_first.md`](docs/bam_training_first.md) and
[`docs/aws_toolkit_setup.md`](docs/aws_toolkit_setup.md), which supersede the old
Modal/default execution commands below. Keep DEV deferred and TEST excluded.

- Integrated AWS tools work through the existing Codex-only `cs230` binding.
  Root-backed access remains local setup access; no credentials went to workers
  or parallel coding agents.
- User permits any suitable AWS region and prioritizes cost per accepted gold
  report. Preflight/worker now require an explicit matching region; 32 focused
  synthetic tests passed. gpt-oss-120b and DeepSeek V3.2 are the first candidate
  teacher/reviewer pair to qualify, with no new model IAM grants. One minimal
  gpt-oss access probe was subsequently rejected; see below.
- Confirmed $100 earned account-plan credit. Five unfinished $20 activities
  explain another $100 of potential credit. The user supplied applicable products
  including EC2, Bedrock foundation models, S3 and CloudWatch, and reports expiry
  in late 2027 (exact date not supplied). No browser check is needed for those
  fields. No additional earned balance or cash shortfall is authorized.
  Existing $200 ledger/history is preserved; one new one-cent probe reservation
  is retained uncertain. Totals: $6.77 completed estimates, $1.83 outstanding,
  $5 buffer, $186.40 admission availability.
- IAM-only stack `cs230-train-pilot-20261108` is `CREATE_COMPLETE`: separate
  controller/researcher/forecaster EC2 roles and profiles restricted to one TRAIN
  prefix. Runtime attachment and a scoped launch controller are not yet tested.
- Direct checks found On-Demand G/VT and G/VT Spot quotas 0 in all 17 enabled
  regions. New us-east-1 request for 8 vCPUs is `CASE_OPENED`, case
  `179411421300610`, request `8405145fff6c4426a52d9e373c4fac98BKDvHupa`.
  SageMaker g6e.2xlarge training checks found zero or unavailable quota records;
  the fully paginated us-east-1 GPU training/Spot quota scan was also all zero.
- Oregon G/VT quota request for 8 vCPUs remains `CASE_OPENED`, case
  `179411216100407`; applied quota remains 0. The user subsequently supplied a
  denial email inviting appeal; the API status does not overturn it. An unsent
  detailed draft is in `docs/aws_gpu_quota_appeal.md`. Support API case reading
  requires paid support and was not available. Do not buy a support plan.
- Missing Anthropic first-use form was confirmed, submitted truthfully with the
  training use case, and Sonnet/Opus agreements created. The user changed payment
  method; retrying the same agreements cleared INVALID_PAYMENT_INSTRUMENT.
  Both now show agreement AVAILABLE but authorization NOT_AUTHORIZED. AWS-sold
  candidates also show NOT_AUTHORIZED. Recheck after activation processing;
  unsent inquiry `docs/aws_bedrock_access_followup.md` is ready if it persists.
  A single gpt-oss Converse probe was rejected with ValidationException:
  Operation not allowed; no usage returned. Do not ask for another card change
  based on old errors. No repeated inference attempt was made.
- New `bam_train_contract.py`, `bam_train_only.py`, and `bam_aws_preflight.py`
  implement strict TRAIN/gold boundaries, joint adapter+head training and funding
  checks. Validation so far is synthetic. The preflight is not a reservation,
  and receipt fields do not independently attest AWS state.

Next: track the new us-east-1 quota request and Bedrock authorization, then
finish AWS-only source preparation, prior-only booster forecasts, full-note
research, qualified teacher/reviewer transport and gold, and the single-writer
launcher with bounded costs and external termination. Freeze 16–32 TRAIN games
before the first learning diagnostic. Preserve locked rich-stat results and all
existing datasets. See `reports/aws_setup/20261108/` for account evidence.

## Historical status (2026-11-01)

**Main full-note experiment is complete.** No research/training/GPU job should be
running. Pipeline `20261101T201235Z` finished research, source validation, MLX
embeddings, head training, evaluation, and review index.

| Item | Location |
| --- | --- |
| Findings / metrics | `reports/researched_forecaster/` |
| Research quality | `reports/game_note_research/quality_audit.md` |
| Evidence browser | `.venv/bin/python scripts/browse_evidence.py` → http://127.0.0.1:8765/ |
| Modal app (idle, reusable) | `cs230-nba-research-vllm` in workspace `tsini` |

**Headline result.** Same 14 held-out games: original boosting Brier **0.186** vs
full-note hybrid **0.193** (hybrid worse by ~0.007; paired 95% CI includes zero).
Not evidence that text helps. Limits: tiny / previously inspected test set, 11/14
Cleveland packets. Frozen Qwen2.5-14B; only the 5,121-parameter head was trained.

**Earlier pilots (keep as history).** Stats vs injury text (`text_ablation`) and
boosting vs first-page notes (`boosting_notes`) also failed to show a text gain.
Details in `docs/experiments.md` and their `reports/*/findings.md`.

## What to work on next

1. More verified pregame text beyond the current team/date sample.
2. Fresh final holdout (e.g. 2026–27 games) — do not treat the 14-game pilot as
   the definitive test.
3. Joint probability + explanation training and market-smoothed targets remain
   unimplemented (see README approach).

## Workspace

- **This worktree (preferred for new work):**  
  `/Users/tommasosini/.superset/worktrees/cs230-nba-forecasting/read-then-continue-dataset-09962af6`
- **Canonical clone:** `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting`
- **GitHub:** https://github.com/TommySini/cs230-nba-forecasting  
  Worktree may be ahead of `origin/main`; push when you want remote to match.
- `data/` is local and git-ignored. Use `.venv/bin/python`.

## Historical commands (do not launch for the current TRAIN-first phase)

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/run_gradient_boosting.py          # stats baseline
.venv/bin/python scripts/evaluate_polymarket.py            # market vs boosting
.venv/bin/python scripts/run_research_experiment.py --stage all   # full pipeline (heavy)
.venv/bin/modal deploy --env main scripts/modal_research_vllm.py  # only if cloud code changed
.venv/bin/python scripts/modal_research_vllm.py --check-deployment
```

Do **not** start `scripts/archive/modal_research_batch.py` or `modal run` ephemeral
apps for the main path. Active cloud path is deploy-once + Python CLI on
`modal_research_vllm.py`. Shared Modal helpers still live in `scripts/modal_research.py`
(legacy MLX benchmark app + volumes/`unpack`); leave that file alone unless
refactoring deliberately.

## Rules that never change

- Pregame inputs only; enforce cutoff timestamps. No raw ESPN → model.
- Outcomes separate from inputs. Unverified narratives stay quarantined.
- Do not claim league-wide note coverage from the current selective sample.
- After collector changes: leakage tests + pilot validator (`collect_pilot.py --offline`).
