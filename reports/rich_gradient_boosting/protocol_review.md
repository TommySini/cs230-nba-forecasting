# Richer gradient boosting: independent protocol review

2026-11-03. Read-only review of the existing baseline and injury-input metadata;
no fitted comparison or newly collected source is certified by this document.

## Comparison fixed before test scoring

Use three nested feature arms: original 18, original + team box-score rates,
and original + team + historical player workload/rotation summaries. Preserve
the old baseline artifacts; write new outputs separately.

For each arm report two controls:

1. Original selected architecture: depth 1, 100 trees.
2. The same original four-candidate grid: depth 1 or 2, 100 or 200 trees, selected
   by validation Brier only. Use identical learning rate 0.05, minimum leaf 20,
   log loss, subsample 1.0, seed 230, no internal early stopping or post-training
   calibration. Tie-break by the declared candidate order. No threshold search.

The fixed-architecture rows isolate feature changes more directly. The common
search gives all feature arms equal tuning opportunity. The original grid's
baseline winner is depth 1 / 100 trees; preserve that exact reference as well.

TRAIN is 2023–24, DEV is 2024–25, TEST is 2025–26 through April 10. Original
eligible counts are **1,115 / 1,123 / 1,137**, with at least five earlier games
for both teams. Select on TRAIN→DEV; fix choices, refit on TRAIN+DEV, lock TEST
predictions, then score. Earlier completed TEST games may update subsequent
features; their outcomes must not refit model weights. This differs from the
proposed expanded text experiment's split and should not be confused with it.

Headline: paired TEST Brier and differences versus the original-feature arm,
with 2,000 paired ET-calendar-date bootstrap resamples. Also report log loss,
ROC AUC, fixed-0.5 accuracy, sample counts, missing-source exclusions and DEV
selection results. The date bootstrap is conditional on the fitted model and
does not address all serial dependence or training uncertainty. This TEST
season was previously examined: call it an exploratory historical comparison.
Do not select a feature arm, retune or rerun based on its TEST outcome.

## Coverage and temporal safeguards

- Before any new fit or comparison scoring, the 367-game cached-source audit
  found 19 games with a player row whose minutes are `--` even though
  `didNotPlay` is false. Other counters being zero does not establish minutes
  were zero. The parent adopted this explicit amendment: **team-source history
  stays complete; affected player-window summaries are missing, with coverage
  and missingness indicators**. Fit median imputation on TRAIN only for DEV;
  after selection fit it on TRAIN+DEV only for final TEST. Retain the actual
  game window and use the same team-eligible cohort for every arm. This is a
  response to source missingness, not to model outcomes. It does not invent
  player minutes or claim a fully observed player history.
- Give compared arms exactly the same TRAIN/DEV/TEST games. If full source
  coverage succeeds, reproduce all original eligible games. If it fails,
  report the complete original reference separately and retrain the original
  feature arm on the same common training cohort as the richer arms before
  attributing a difference to features.
- A last-five/last-ten window means those actual preceding games, not the last
  five/ten games with available data. With five to nine prior games, the last-ten
  window includes all those games, preserving the original warmup. A season
  aggregate must not silently ignore a missing historical box score. Missing
  source coverage is not zero shooting/usage or confirmation that a player was
  absent.
- Calculate percentages from summed numerators/denominators. Clearly label
  possession-derived efficiency/pace as estimates if using box-score formulas
  instead of NBA-provided possession counts. Normalize overtime consistently.
- Use strictly earlier ET dates in the same season, matching the original
  baseline's conservative rule. Store game IDs and source hashes used by each
  feature row. Test perturbations to target/future scores, player rows, starters,
  roster identities and current news must leave that target's features unchanged.
- Historical player workload can use prior participation/minutes/shooting volume.
  Never use the target game's actual participant/starter list to decide whose
  historical statistics to aggregate. A known prior rotation is not a confirmed
  current lineup or availability list.
- Retrieval now does not independently prove historical byte availability or the
  time of later statistical corrections. Retain provenance and disclose this
  retrospective limitation, as the original baseline already does. Raw ESPN
  summaries, recaps, current news and injury fields must not enter the numeric
  feature matrix; normalize an explicit historical numeric allowlist.

## Existing injury availability is not league-wide

I inspected `data/processed/inputs.jsonl` and
`data/processed/seasonal/inputs.jsonl` without reading outcome-label files.
These contain 160 unique sampled games: 100 in November, 30 in January and 30
in April. Of these, **140** overlap the original 3,375 eligible baseline games
(**4.15%**): 40 TRAIN, 48 DEV and 52 TEST. Twenty are baseline warmup exclusions.

All 140 have an internal report timestamp at or before cutoff and no
`not_yet_submitted` flag. However, **48** stored HTTP Last-Modified timestamps
are **3–13 seconds after the exact cutoff**, despite the minute-rounded report
header being at the cutoff. Counts are 9 TRAIN, 16 DEV, 23 TEST. A strict
byte-availability policy needs an earlier report/version for these cases; do
not invent a grace period after inspecting outcomes.

Only 92 existing selected versions pass that metadata comparison (31/32/29).
One TRAIN game, `espn:401584093`, records rows for only Oklahoma City, not both
teams. This does not prove Golden State had no injuries. These counts are
metadata checks, not certification of extraction accuracy or a structured
player-status parser.

**Recommendation:** implement broad historical workload/rotation features now;
defer broad current-availability features until properly dated reports cover the
cohort. A sparse optional injury arm would answer a different sampled-cohort
question and should not be presented as complete league-wide availability.
Do not zero-fill the other 95.85% as healthy or infer status from who eventually
played. Preserve the original injury artifacts; this review does not rewrite
their prior experiment results.

## Implementation review

I independently reviewed `collect_rich_stats.py`, `rich_stats_features.py`,
`run_rich_gradient_boosting.py`, `validate_rich_gradient_boosting.py` and their
new tests while collection was in progress. Identified integration issues were
sent to their owners: duration placement/consistency, null player IDs, team
minutes depending on successfully parsed player rows, source-manifest admission
checks, explicit missing-player handling and configuration/window agreement.
The feature/runner fixes are in place. The collector's duration repair was
independently reviewed and its 13 tests pass: both sides' ordered quarter scores
must sum to their verified final scores and have matching counts; elapsed game
minutes are 48 plus five per overtime. An explicit final period must agree when
also present. Missing player identities/minutes remain unknown, even when the
team's elapsed duration is known. In a real Chicago case, 240 team-minutes are
supported by four reconciled quarters while the 229 observed player-minutes and
incomplete-player flag are preserved.

The final franchise census identified eight All-Star exhibitions in the canonical
history that ESPN labels as regular-season events: one in February 2024, three
in February 2025 and four in February 2026. These use non-franchise team names;
the 2025–26 editions also have nonstandard duration. An
explicit 30-franchise admission rule must classify them as out of scope, without
rewriting canonical history or pretending they lasted 48 minutes. They are not
part of the original eligible forecast cohort. Final collection and feature
admission must confirm all 3,375 original eligible forecasts remain paired.

The feature builder's target,
future, same-day and season tests cover meaningful leakage boundaries; its
ratio and missing-window tests reflect the intended definitions.

The runner's imputer fits TRAIN for DEV and TRAIN+DEV for the predeclared final
refit. All arms share the same cohort; feature-arm and model selection receive
DEV only; model hashes and TEST predictions are locked before scoring. The
validator checks feature/source reconstruction, final fit membership, imputer
medians, saved predictions and numerical metrics. No remaining fit/selection
blocker was found in this code review.

Final validation remains necessary after collection: rebuild normalized data
offline with the final collector code, replay every raw source through its
provider-specific normalizer (including the official-NBA repair path), check
original-baseline reproduction, and run the leakage suite and cached pilot
validator. This review is not a claim that the final dataset/model validation
has already run or passed.

Parent completion note: the required final checks subsequently passed. All
3,613 retained game box scores reconstruct exactly from preserved raw sources;
all 3,375 original eligible examples remain paired (1,115/1,123/1,137); the saved
models and imputation reproduce their predictions and metrics; original baseline
predictions and artifact hashes are unchanged. The 53 targeted unit tests and
both cached pilot validators passed. See `validation.json` and
`source_validation.json` for the completed run's record.
