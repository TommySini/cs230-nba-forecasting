# CS230 NBA forecasting — handoff

Current state for the next session. Public summary: [`README.md`](README.md).
Retired experiment code is at git tag `pre-cleanup`. Reports consolidated during
the folder cleanup remain in Git at `67596558a188014803724bd60a158ce6f205ec3e`.
Read [`reports/README.md`](reports/README.md) for the three maintained reports;
current run records stay where the dataset release pins them (`reports/gradient_boosting/`,
`reports/polymarket/`, `reports/tabpfn/`); archived pilot metrics live in `results/`.

## Status (2026-11-09)

**Dataset distribution.** `download_data.py` installs the checksum-pinned processed
release by default; `--include-raw` adds the source archive parts. `package_data.py`
builds deterministic assets from the validated inventory without modifying source
files. The checked-in `config/dataset_release.json` pins release
`nba-data-4e5ba4eb0554`. Downloaded data uses the grouped raw layout directly and
has no legacy migration folders. Packaging and downloader tests are included;
the full suite passes 420 tests with one optional skip. An isolated checkout
passes processed-only checks, all source-level validators, and checksums for all
74,799 raw files. The [public prerelease](https://github.com/TommySini/cs230-nba-forecasting/releases/tag/nba-data-4e5ba4eb0554)
is published: all 17 hosted asset digests and 16 anonymous archive URLs passed.
A fresh public clone downloads and validates with baseline-only dependencies.
Evidence is in `reports/data_reconstruction/validation.json`; temporary checkouts
and their caches were removed. [PR #5](https://github.com/TommySini/cs230-nba-forecasting/pull/5)
contains this work and overlaps the separate TabPFN, dependency and reports PRs.

**Shared Parquet migration.** `data-reconstruction` branches from the completed
TabPFN baseline at `7681631`; its worktree is
`/Users/tommasosini/.superset/worktrees/cs230-nba-forecasting/data-reconstruction`.
The canonical shared tables are published and this branch reads them in place of
model-specific input copies. See
[`docs/data_reconstruction.md`](docs/data_reconstruction.md) for the table map,
feature dictionary, TRAIN-game walkthrough and migration commands. Features,
labels, forecasts, raw provenance and citation lines preserve the frozen values.
The full suite passes 383 tests (one optional tokenizer test skipped); boosting,
both TabPFN cohorts, notes, Polymarket and BAM validation pass on the installed data.
Evidence: `reports/data_reconstruction/validation.json`. The staging copy was removed.
No training or new forecasting experiment was launched for this migration;
validation replays the saved boosting models to check existing predictions.

**Compatibility window remains open.** A live TabPFN adaptation process was observed
during migration; its separate worktree references the old baseline files. Publication uses
`--keep-raw-layout`: original raw and legacy processed paths remain intact.
Do not move or delete them until their consumers have switched. Unique adaptation
weights and records are outside the cleanup scope. After coordination, run
`nba_dataset.py relocate-raw`, all matching validators, then `cleanup --validated`.
Cleanup verifies replacement coverage and surviving raw bytes before deletion.

**TabPFN experiments complete (2026-11-09).** Work is on `tabpfn-implementation`,
based on the fixed datasets at `a75d4cc`. The user authorized local compute and
three experiments: synthetic augmentation, fine-tuning, and both combined.
Original boosting/BAM outputs and the original TabPFN run remain frozen.

`scripts/run_tabpfn.py` uses the pinned synthetic-only TabPFN-2.5 checkpoint on
Apple MPS, four estimators, all 252 frozen features, and all 13,457 TRAIN rows
without subsampling. The 32-game TRAIN diagnostic passed (14 context / 18 later
queries, split at an ET-date boundary). The full 1,123-game DEV comparison completed
in 236.6 seconds: TabPFN Brier **0.207831**, boosting **0.210332**. Difference
**-0.002500**, paired day-bootstrap 95% interval **[-0.005122, +0.000148]**: a
modest point-estimate improvement, not a conclusive advantage. Accuracy is 66.8%
versus 66.9%. No TEST inference or scoring was added.
`scripts/validate_tabpfn.py` passes for both stages; all 15 frozen baseline hashes
remain unchanged. Results are in `reports/tabpfn/findings.md` and `metrics.json`;
per-game forecasts are in shared `data/processed/predictions.parquet`; cohort/run
records are in `reports/tabpfn/{train-smoke,dev}/`. Original boosting/BAM forecast
values remain frozen. DEV comparison is exploratory because boosting settings used
DEV. That baseline run finished; separate adaptation work is now active. No AWS
resources were launched by the data migration.

The original TabPFN-2.5 baseline uses all 13,457 TRAIN games and 252 features,
four estimators on Apple MPS, and scores 1,123 DEV games: Brier **0.207831**
(boosting **0.210332**). New experiments found **no Brier improvement**:

- Synthetic-only: 1,024 generated rows added to context; Brier **0.208017**,
  difference from original TabPFN **+0.000186**, paired 95% interval
  **[-0.000243, +0.000606]**. Accuracy 67.05% versus 66.79% (three more correct games).
- Fine-tuning-only: 96 actual updates across two epochs; later-TRAIN validation
  log loss worsened (0.604215 -> 0.606617 -> 0.607731), so early stopping restored
  pretrained weights. DEV probabilities exactly equal the original TabPFN run.
- Combined: the same synthetic rows plus 104 updates across two epochs;
  validation worsened (0.605741 -> 0.612362 -> 0.617062). Pretrained weights were
  restored; DEV probabilities exactly equal synthetic-only.

Protocol: generation and weight updates use 12,342 games from 2012–22;
1,115 games from the 2023–24 TRAIN season select checkpoints. Final context
includes every original TRAIN game. DEV is exploratory; TEST remains excluded.
The official generator used a deterministic 1,024-game context and eight-bin
categorical representations because its continuous regressor checkpoint could
not be downloaded. This is a bounded binned-generation study. Fine-tuning used
LR 1e-5, 256-row meta-batches, five-epoch cap/patience two, one estimator for
training/validation and four for final inference. The official package warns
that these different estimator counts with fixed preprocessing can affect
performance; one seed and these settings do not exhaust either method.

Code: `run_tabpfn.py`, generation/fine-tuning helpers, `validate_tabpfn.py`;
settings: `config/tabpfn_adaptation.json`. All three saved-result validators pass;
371 tests ran successfully with one optional tokenizer test skipped. In the consolidated branch the
adaptation reads the frozen baseline through the shared dataset (`reports/tabpfn/dev/`,
`predictions.parquet`) instead of the removed `data/processed/tabpfn/dev/` copies.
Report: `reports/tabpfn/findings.md`; complete per-game outputs, separate synthetic
features/labels, checkpoints and provenance: `data/processed/tabpfn/adaptation/`.
No experiment job remains running. These methods have not replaced boosting in BAM.

**Plan: BAM teacher-explanation training on AWS, TRAIN-first.** Prove learning on 16–32
TRAIN games before touching DEV; TEST stays excluded. Design: [`docs/bam_aws_architecture.md`](docs/bam_aws_architecture.md),
[`docs/bam_training_first.md`](docs/bam_training_first.md). Account details and evidence:
[`docs/aws_toolkit_setup.md`](docs/aws_toolkit_setup.md), the historical evidence references in that document.
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

**Data and baseline (values fixed 2026-11-08; storage reorganized).**

- Fixed splits: TRAIN 2012–13 to 2023–24, DEV 2024–25, TEST 2025–26 (`config/gradient_boosting.json`).
  `bam_train_contract.py` reads its TRAIN date range from the same config.
- Box scores: shared `box_score_games`, `team_box_scores`, `player_box_scores` and
  `sources` Parquet tables, 16,810 of 16,821 games 2012–26 (2,104 repaired from
  NBA.com; 11 missing with reasons in `box_score_audit.json`). `collect_rich_stats.py` normalizes pre-2021
  ESPN team rebounds, sends scrambled 2017–18 team stats and empty boxes to NBA.com repair, and
  rebuilds offline from the raw cache in about 90 s. Known, left as is: 7 ESPN 2020–21 team rows
  have offensive or defensive rebounds 1 below their player sum.
- `run_gradient_boosting.py` is the single baseline: fit on TRAIN, settings chosen on DEV (depth 2,
  200 trees; DEV Brier 0.2103), TEST scored once: **0.2058**. TRAIN games 2013–24 get forecasts from
  models fitted on earlier seasons only (`produced_by: prior_seasons`); 2012–13 has none. Output:
  `data/processed/predictions.parquet`, filtered to `model_id=gradient_boosting`.
- Market at 8 h beats boosting by about 0.010 Brier on DEV and TEST ([baseline report](reports/baselines.md); records in `reports/polymarket/`).
- BAM cohort in shared `notes.parquet`, `note_pages.parquet`, `bam_manifest.json`
  (`bam_data.py validate data/processed` passes):
  1,922 games with notes 2012–26, 2,064 documents (80 HTML posts revised after their cutoff
  excluded). 1,768 have a boosting probability (TRAIN 1,456, DEV 182, TEST 130); the rest are
  research-only (138 too early in the season, 16 in 2012–13). Summary:
  `summary` in `data/processed/bam_pipeline/bam_notes_v2/manifest.json`.

## Next

Finish data cleanup only after legacy consumers switch, as described above.

1. When quota and Bedrock access arrive: qualify the teacher/reviewer on a few TRAIN games,
   generate research and reviewed gold for 16–32 TRAIN games, build the TRAIN bundle
   (`bam_train_contract.py`) and run `bam_train_only.py`.
2. DEV evaluation only after TRAIN learning is shown. TEST (2025–26) was already inspected in
   earlier pilots, so treat it as exploratory; a fresh 2026–27 holdout is cleaner.

## Workspace

- Canonical checkout: `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting` (all data lives
  in its `data/`; other worktrees link `data` there).
- Use `.venv/bin/python`. The Codex worktree's `.venv` has the full ML stack (torch, transformers).
- Dependency pins live in `pyproject.toml`: `local`, `training`, `research`, and `validation`
  groups. Local setup: `uv pip install --python .venv/bin/python --group local`.
- GitHub: https://github.com/TommySini/cs230-nba-forecasting. Local `main` may be ahead.

## Rules that never change

- Pregame inputs only; enforce cutoff timestamps. No raw ESPN responses to the model.
- Outcomes separate from inputs.
- Rewrite instead of accumulating; delete superseded code, configs and reports.
