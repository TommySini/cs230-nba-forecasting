# Rich-statistics boosting comparison

Protocol recorded 2026-11-03 before the expanded feature models are trained or
their held-out predictions scored. This is an exploratory backtest on the
previously examined 2025–26 season.

- Preserve the original 18-feature baseline and its artifacts unchanged.
- Use original season membership: 2023–24 TRAIN, 2024–25 DEV, 2025–26 TEST,
  through April 10, with at least five prior games for each team.
- Compare original features, original plus richer team features, and original
  plus team and past-player-workload features. Current-game participants or
  actual starting lineups are never used to construct a pregame feature.
- Team features cover shooting, possession control, estimated efficiency and
  pace from historical numeric box scores. Use season-to-date and last-five /
  last-ten histories. Ratios use pooled counts. All source games must be on
  earlier Eastern calendar dates in the same season.
- Require complete eligible prior TEAM box-score histories. Every matched comparison uses the
  same TRAIN, DEV and TEST membership. Record exclusions; do not silently compare
  a filtered rich cohort against the published full-cohort baseline.
- Before any model fit, player-window missingness was explicitly accommodated:
  incomplete historical player boxes yield missing player statistics and
  coverage indicators. Preserve the actual last-five/last-ten game membership.
  Fit median imputation only on TRAIN during DEV selection, and TRAIN+DEV in the
  final model. Never infer zero minutes from ambiguous source placeholders.
- First compare all feature arms at the original fixed model settings: 100
  trees, depth 1, learning rate .05, minimum leaf 20, log loss, full subsampling,
  seed 230, no internal early stopping, no post-training calibration.
- Also apply the same original four-candidate grid to every feature arm: depth
  1 or 2 and 100 or 200 trees. Select by DEV Brier only, with deterministic ties.
  Pick the primary richer arm between team and team-plus-player using DEV only.
- Refit each frozen configuration on TRAIN plus DEV. Save TEST probabilities
  and model/data/config/code hashes before computing held-out metrics.
- Primary result: paired Brier difference between the DEV-selected richer model
  and the fixed original baseline. Include log loss, AUC, accuracy at threshold
  .5, and paired ET-date cluster-bootstrap 95% intervals (2,000 draws).
- Report the original-feature model under the same tuning grid as a capacity
  control. Report both richer arms even if they lose; no TEST-driven tuning.
- Preserve raw responses, URLs, retrieval timestamps, checksums and normalized
  numeric-only source tables. Raw ESPN packages and recaps never enter a model.

Historical NBA Stats access timed out in a bounded probe. Equivalent
box-score counters are being collected from ESPN with explicit source labeling. Possession
estimates and derived ratings will be labeled estimates, not official NBA
ratings. Historical correction/publication times remain an availability limit.

The existing injury-report sample covers only 140 of 3,375 original eligible
games and does not justify a league-wide current-availability feature. Missing
injury reports will not be treated as healthy teams. Player-workload summaries
refer to earlier participation; actual current-game availability is not inferred.

This experiment uses local CPU training and public historical data collection.
It does not start paid LLM/API or Modal work. The BAM experiment is preserved.
