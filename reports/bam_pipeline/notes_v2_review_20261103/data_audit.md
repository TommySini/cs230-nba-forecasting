# Claude expansion: data and split audit

Inspected canonical local commit `723c8b0743ddc7acd42df60b4023ed8c969dc6cb` on 2026-11-03 UTC. GitHub `main` and `continue-dataset-work-09962af6`, checked with `git ls-remote`, still pointed at `101b96049c83f44a7658015a1b23f8e9d27ddb58`; the new commit was local, not pushed to those branches at inspection. No branch was merged, reset, committed or pushed. The worktree's existing README modification was preserved.

The new collector adds 13 tracked files: collection/configuration/validation code, tests, archive research reports and a preliminary memorization probe. Actual notes and normalized history remain local and git-ignored. Canonical data is under `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting/data/processed/notes_v2/`; it is not already imported into the frozen BAM snapshot in this worktree.

## Counts from the actual manifests

The new approved manifest contains **2,144 team-game documents for 1,997 games**, with no duplicate `(game_id, publisher)` rows. There are 1,634 archive-proven documents and 510 publisher-timestamped documents. At game level, 1,527 have at least one archive-proven source and 470 rely entirely on publisher timestamps. Publisher timestamps remain a separate accepted evidence tier; this review does not require every source to have an archive capture.

The old snapshot contains 424 games. It overlaps the new collection on 376; **48 old games would be lost by replacing the snapshot blindly**. The raw union is 2,045 games, including 1,621 games absent from the old snapshot.

The provenance review identifies 80 HTML documents explicitly modified after prediction cutoff. Removing those document versions leaves 1,922 games in the new collection; 75 games have no remaining new source. The old validated snapshot supplies a separate source for 10 of those 75. Preserving old sources and combining the remainder produces **1,980 provisional candidate games**. This is a metadata gate, not a statement that every document has passed source reconstruction or semantic review. See [provenance review](provenance_review.md).

## Recommended chronological partition

| Role | Seasons | Candidate research games | At least five prior games per team | Also at least 200 earlier-month numeric games |
| --- | --- | ---: | ---: | ---: |
| Train | 2012–13 through 2023–24 | 1,632 | 1,512 | 1,506 |
| Dev | 2024–25 | 201 | 190 | 190 |
| Test | 2025–26 | 147 | 140 | 140 |

Actual candidate date ranges are 2012-11-12–2024-04-10, 2024-10-23–2025-04-10, and 2025-10-22–2026-04-10. These are the collected portions of those seasons, not complete regular-season coverage. Splits use dates and season membership only, never outcomes. Keep both teams' documents for a game together.

The last column is **potential forecasting eligibility**, not existing generated features or fitted probabilities. It mirrors the current five-previous-ET-date-games rule and the BAM earlier-calendar-month minimum of 200 numeric examples. Six initial 2012 training games meet the per-team warmup but lack enough earlier-month numeric history. Final eligibility also depends on schedule-cache verification, source reconstruction, record/header review and existing game-type exceptions. Games without a valid booster forecast can remain researcher-training examples.

For comparison, Claude's new-only raw counts are 1,592 / 223 / 182. The combined, version-gated counts above preserve more training data while excluding known late revisions. An archive-only-new-plus-old sensitivity cohort would contain 1,468 / 113 / 77 research games (1,353 / 113 / 77 potential forecasting games); this is a sensitivity analysis, not a new mandatory source policy.

A later-DEV alternative would train through 2024–25, use October–December 2025 as DEV and January–April 2026 as TEST: **1,833 / 60 / 87 candidate research games**. It allows more models with verified 2025 releases but materially shrinks DEV/TEST. The whole-season proposal is preferable when using a model with a documented cutoff before October 2024.

## Numeric history and boosting

The canonical checkout contains 13,208 older historical rows plus 3,621 seasonal rows, totaling **16,829 unique game IDs**. All old/new note game IDs match this combined normalized history. This audit used dates, seasons, team identities and prior-game counts; it did not use winning teams, scores or Brier results to choose partitions. Source file hashes and detailed counts are recorded in [data_audit.json](data_audit.json).

The current frozen baseline configuration only accepts the 2023, 2024 and 2025 seasons. Its original final model was fitted on 2023–24 plus 2024–25. Consequently its in-sample probabilities cannot be reused as the proposed 2024–25 DEV inputs. Extend verified numeric-history loading, retain the original selected boosting hyperparameters, and produce prior-only forecasts for historical TRAIN and DEV games. Predeclare the final TEST fit/refit policy and give both forecasting arms identical booster probabilities. Preserve all original baseline models, membership and predictions as historical artifacts.

No older-game booster forecasts were fit in this review. The metadata count is not a substitute for the collector's raw-cache checksum validation or the numerical feature audit.

## Source distribution and previous exposure

The new TRAIN source mix is 1,515 archived PDFs, 125 live dated PDFs and 76 HTML posts. DEV has 75 / 71 / 97 respectively; TEST has 44 / 62 / 79. Thus older training data is much more archive/PDF-heavy, while recent evaluation data depends more on publisher timestamps and web articles. The raw recent TEST documents include 79 NYK and 63 CLE packets out of 185. Evaluate source/publisher coverage explicitly; this remains a selected corpus, not representative league-wide narrative availability.

The new historical TEST overlaps previously evaluated text pilots: all 14 games from both the first-page and full-note hybrid tests are present, and one game from the injury-text test is present after the source gate. The old 424-game snapshot also contains 10 old-TRAIN, 21 old-DEV and 37 old-TEST games within this new season-long TEST. The unfinished revised BAM fit has not used these for a new NBA training run, but prior labels/reports and the statistical/market analysis have already been inspected. Therefore this historical comparison is **exploratory, not a fresh untouched final benchmark**. A future prospectively captured test remains the cleanest final confirmation.

## Work required before freezing the new dataset

1. Preserve the union of old/new sources; recover pre-cutoff versions of late HTML when possible, otherwise keep them out of prediction inputs.
2. Reconstruct selected article/PDF text from exact source bytes and bind source identity, extraction settings, page/section lines and hashes.
3. Check explicit numeric years, game headers, unreadable pages and undecoded tables; preserve unknowns rather than infer missing content.
4. Verify the extended history and produce earlier-only boosting forecasts with fixed original hyperparameters.
5. Freeze a new snapshot and date partition, including source tiers and prior-exposure qualifications; leave the old 424-game snapshot intact.

No collection, GPU job, paid API call or training/evaluation was launched. The three independent reviews and the root's manifest/metadata calculations are read-only apart from these new reports.
