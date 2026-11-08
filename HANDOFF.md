# CS230 NBA forecasting — handoff

Current state for the next session. Public summary: [`README.md`](README.md).
Everything removed in the November 2026 cleanup is at git tag `pre-cleanup`.

## Status (2026-11-08)

**Plan: BAM teacher-explanation training on AWS, TRAIN-first.** Prove learning on 16–32
TRAIN games before touching DEV; TEST stays excluded. Design: [`docs/bam_aws_architecture.md`](docs/bam_aws_architecture.md),
[`docs/bam_training_first.md`](docs/bam_training_first.md). Account details and evidence:
[`docs/aws_toolkit_setup.md`](docs/aws_toolkit_setup.md), `reports/aws_setup/20261108/`.
No real pilot fit has run.

**Blocked on AWS compute and model access.**

- GPU quota is 0 for G/VT On-Demand and Spot in all 17 enabled regions. Open quota cases:
  us-east-1 8 vCPUs, case `179411421300610`; Oregon 8 vCPUs, case `179411216100407`
  (denied by email; appeal sent 2026-11-08). The user reports EC2
  launches are limited to three regions; confirm which.
- Bedrock: Anthropic first-use form submitted, Sonnet/Opus agreements AVAILABLE but
  authorization NOT_AUTHORIZED; AWS-sold gpt-oss/DeepSeek also NOT_AUTHORIZED. One gpt-oss
  probe returned `ValidationException: Operation not allowed`. The user sent the
  Bedrock follow-up and the support messages (quota for N. Virginia, Ohio, Oregon) on 2026-11-08.
- IAM-only stack `cs230-train-pilot-20261108` is `CREATE_COMPLETE` (`infra/aws/train-pilot-iam.json`).
- CLI profile `cs230` is bound to Codex only; its session had expired at last check.
- Budget cap $200 (`config/bam_budget_200_20261107.json`; ledger in
  `data/processed/bam_budget/`): $6.77 completed estimates, $1.83 outstanding, $5 buffer,
  $186.40 available. $100 earned AWS credit confirmed.

**Data is ready for the pilot.**

- BAM snapshot `data/processed/bam_pipeline/bam_notes_v2_20261108`: 663 games with notes,
  2023–26, built from notes_v2 by `bam_data.py`. 80 HTML posts revised after their cutoff are
  excluded. Boosting forecasts in it still come from the basic 18-feature model with
  earlier-month folds.
- Rich boosting (234 features) is the baseline to use: 0.206 vs 0.210 Brier on 2025–26.
  It was fitted on 2023–24 (+2024–25 for the final model) and its locked results must not be
  overwritten; a new run needs new output paths.

## Next

1. Rebuild the baseline with the rich features, fitted chronologically (prior-only folds) on
   all available training seasons, and wire it into `bam_data.py` in place of the basic booster.
   Box scores exist only for 2023–26; older seasons need `collect_rich_stats.py` runs first.
2. When quota and Bedrock access arrive: qualify the teacher/reviewer on a few TRAIN games,
   generate research and reviewed gold for 16–32 TRAIN games, build the TRAIN bundle
   (`bam_train_contract.py`) and run `bam_train_only.py` with bounded spend.
3. DEV evaluation only after TRAIN learning is shown. The 2025–26 TEST season was already
   inspected in earlier pilots, so treat it as exploratory; a fresh 2026–27 holdout is cleaner.

## Workspace

- Canonical checkout: `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting` (all data lives
  in its `data/`; other worktrees link `data` there).
- Use `.venv/bin/python`. The Codex worktree's `.venv` has the full ML stack (torch, transformers).
- GitHub: https://github.com/TommySini/cs230-nba-forecasting. Local `main` may be ahead.

## Rules that never change

- Pregame inputs only; enforce cutoff timestamps. No raw ESPN responses to the model.
- Outcomes separate from inputs.
- Rewrite instead of accumulating; delete superseded code, configs and reports.
