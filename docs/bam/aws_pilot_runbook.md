# AWS runbook: TRAIN concept pilot

Everything needed to go from the shared dataset to a trained concept-bottleneck pilot on
AWS. The code is implemented and tested on synthetic data (CPU). It has never run against
AWS: as of 2026-11-09 the account has no GPU quota and Bedrock access is not authorized.

| Piece | Code | Config |
| --- | --- | --- |
| Orchestrator (select, research rounds, rows, teacher, bundle) | `scripts/bam_pilot.py` | `config/bam_pilot.json` |
| Bedrock caller (spend cap, cached answers) | `scripts/bam_bedrock.py` | teacher/reviewer in `config/bam_pilot.json` |
| EC2 launcher (termination armed, receipts) | `scripts/bam_aws_launch.py` | `config/bam_aws.json` |
| Research generation (vLLM) | `scripts/bam_inference.py` via `bam_pilot.py infer` | `inference` in `config/bam_pilot.json` |
| Training worker | `scripts/bam_train_only.py` | `config/bam_train_only_concept_pilot.json` |
| Funding preflight | `scripts/bam_aws_preflight.py` | evidence, plan, ledger |

## Decisions and account work still required

`bam_pilot.py check` and `bam_aws_launch.py check` list these until they are resolved.

1. **GPU quota** in `us-west-2`: at least 8 vCPUs of On-Demand G/VT (one `g6e.2xlarge`).
2. **Bedrock access** in `us-west-2` for the teacher and reviewer. The deployed controller
   IAM already permits Claude Sonnet 4.6 (teacher) and Opus 4.6 (reviewer) through the
   `us.` inference profiles, so the config is pre-filled with those. Using gpt-oss or
   DeepSeek instead needs an IAM change.
3. **Current Bedrock prices** for both models (`price_per_million_*_usd`). The spend cap
   and preflight cost bound depend on them.
4. **Research critic:** a pinned open model, different from Qwen2.5-14B, trained before
   DEV (October 2024), to review the researcher's modules blind to outcomes (served by vLLM).
5. **Artifact bucket:** create the stack's `ArtifactBucketName` bucket in Oregon and set
   `bucket` in `config/bam_aws.json`.
6. **AWS connector or credentials:** the claude.ai AWS connector needs re-authorization
   before Claude can read the account; launches use your own operator credentials.

## Local preparation (CPU, no AWS)

Run from the repository root on Linux/WSL with the `local` group, after
`scripts/download_data.py --include-raw` (postgame records need the raw bundle).

```bash
python scripts/bam_pilot.py check
python scripts/bam_pilot.py select            # 40 candidate 2023-24 TRAIN games, PDF notes only
python scripts/bam_pilot.py research          # writes runs/concept_pilot/research/rounds/000-researcher
python scripts/bam_pilot.py postgame          # normalized postgame records for the teacher
```

The first research round on the default 40 games is 1,528 page-chunk requests (18-90 per
game, prompts up to about 28k characters).

## On AWS

`RUN=pilot-YYYYMMDD` throughout; `P=s3://<bucket>/train/pilot-20261108`.

1. **Code:** commit, then `python scripts/bam_aws_launch.py upload-code` (prints `CODE_KEY`).
2. **GPU smoke (cheap, first):**
   `python scripts/bam_aws_launch.py launch --job smoke --run-id $RUN --code-key $CODE_KEY`
   (add `--dry-run` first to test permissions). Read `$P/metrics/$RUN/smoke/smoke/smoke_result.json`:
   peak memory must fit, and `projected_pilot_seconds_per_step_32_games` sets
   `max_runtime_seconds` in the forecaster config and the preflight cost bound.
3. **Research rounds** (repeat until `research` reports no pending requests):
   - `aws s3 cp runs/concept_pilot/research/rounds/<ROUND>/request.json $P/pregame/$RUN/rounds/<ROUND>/request.json`
   - `python scripts/bam_aws_launch.py launch --job research --run-id $RUN --round <ROUND> --code-key $CODE_KEY`
   - `aws s3 cp --recursive $P/research/$RUN/rounds/<ROUND>/output runs/concept_pilot/research/rounds/<ROUND>/output`
   - `python scripts/bam_pilot.py ingest --round runs/concept_pilot/research/rounds/<ROUND>`
   - `python scripts/bam_pilot.py research` (next round: modules, then critic reviews)
4. **Rows:** `python scripts/bam_pilot.py rows` (games whose three modules passed review).
5. **Teacher** (needs Bedrock permission: the controller role on a small CPU instance, or
   your own credentials): `python scripts/bam_pilot.py teacher --dry-run` prints the
   worst-case cost; then `python scripts/bam_pilot.py teacher`. Answers are cached per
   request, so reruns resume without paying twice; the cap is `teacher_spend_cap_usd`.
6. **Bundle:** `python scripts/bam_pilot.py assemble` writes `runs/concept_pilot/bundle`,
   validates it with the training contract and prints `MANIFEST_SHA256`. Upload it:
   `aws s3 cp --recursive runs/concept_pilot/bundle $P/targets/$RUN/bundle`.
7. **Preflight and admission:** collect evidence and run `bam_aws_preflight.py` for job
   `$RUN`, the bundle hash and the forecaster config. Reserve the bounded cost in the budget
   ledger, then
   `python scripts/bam_aws_launch.py admit --preflight preflight.json --run-id $RUN --manifest-sha256 $MANIFEST_SHA256 --reservation-id <id> --reserved-cents <n> --output admission.json`
   and upload both files to `$P/targets/$RUN/`.
8. **Train:** `python scripts/bam_aws_launch.py launch --job train --run-id $RUN --code-key $CODE_KEY --manifest-sha256 $MANIFEST_SHA256`.
   Results: `$P/metrics/$RUN/result_train.json` (`learning_criterion_met` and the Brier,
   gold CE and concept reductions), checkpoints and the trained adapter/head under
   `$P/checkpoints/$RUN/`.

Every instance runs `shutdown -h +<max_minutes>` before any other command and is launched
with `InstanceInitiatedShutdownBehavior=terminate`. Limits are in `config/bam_aws.json`
(smoke 60, research 240, train 60 minutes).

## Known limits

- Rejected or failed research or teacher answers drop that game instead of retrying;
  the pilot selects 40 candidates for 24 slots. Repair loops are not wired into the pilot.
- `bam_pilot.py infer` hashes every checkpoint file before generation (minutes for a 14B model).
- Spend-cap reservations assume about three characters per token; Bedrock's reported
  usage sets the recorded cost.
- Not exercised against real AWS services: EC2 launch, S3 permissions, Bedrock Converse
  responses for these models, vLLM on L40S.
