# CS230 NBA forecasting — archived handoff (through 2026-11-01)

**Superseded.** Current short handoff: [`HANDOFF.md`](../../HANDOFF.md).
Experiment map: [`docs/experiments.md`](../experiments.md). This file is the
chronological agent log kept for provenance; do not treat mid-file “active run”
instructions as current.

## Latest completed full-note experiment — November 1, 2026

**The requested experiment is complete. No research/training/GPU job is running.** Pipeline
`20261101T201235Z` completed research, source validation, all107 uniform MLX CUDA
vectors, head training, dev selection, test evaluation, replay and review index.
See `reports/researched_forecaster/findings.md`, `metrics.json`, `validation.json`
and `evidence_index.md`; global status is `complete`.

All107 PDFs/3783pages/3858research portions were processed. All10074 retained
evidence items (8092 research items +1982 extra code-decoded table facts) were
encoded in352 chunks/1749153tokens. Split80train/13dev/14test. Dev selected
L2=.001: Brier .347535 booster -> .332851 hybrid. Test Brier .1862960396 booster
vs .1932064626 hybrid; difference +.006910423 (3.71% worse), paired date-bootstrap
95% CI[-.011765789,.029195764]. Both accuracy78.6%, AUC.836735. This does not
establish added predictive value;14repeated historical test games,11CLE packets,
and uncertain note interpretation limit inference. No test-driven tuning.

All85tests, pilot validator, full-source validator and final forecaster replay
pass. Head is5121parameters; Qwen2.5-14B frozen; original booster unchanged.
Modal execute/encode_reports/convert_model all have0running inputs/containers.
The named deployment remains reusable; do not stop it or unrelated apps.
Older sections below are chronological history, including superseded active-run
instructions. The user explicitly wants work carried through actual completion,
not a final response merely announcing a background launch.

## Current request and location

The user clarified the intended main comparison: **original boosting** versus a
**BAM-inspired forecaster that consumes the boosting probability plus rich
pregame game notes**. Exactly these two main systems, evaluated on identical
held-out games. The previous raw-statistics LM versus injury-text LM pilot did
not test this intended architecture; preserve it only as exploratory history.
The corrected hybrid is implemented with an expanded verified game-note cohort.
The frozen experiment uses 80 train / 13 dev / 14 test games, all after the
September 2024 backbone checkpoint. The earlier first-page experiment and the
full-note research revision above are both complete; both hybrids performed
worse on their shared small test cohort. The original baseline remains intact, and its discarded expanded
tuning experiment stays removed. No purchases, scheduling or GitHub push.

Canonical path: `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting`.
**The new work is in this Superset worktree**, on `continue-dataset-work-09962af6`:
`/Users/tommasosini/.superset/worktrees/cs230-nba-forecasting/read-then-continue-dataset-09962af6`.
Its `data/` began as a copy of the canonical dataset; it has a separate `.venv`
with the pinned requirements. New data and code have not been copied back or
merged into the canonical checkout. Continue here to retain the expanded cache.

## Research objective

Forecast NBA home-team win probabilities from pregame text and prior-game stats.
The proposal adapts BAM's *Global Merger-Arbitrage Forecasting with Language Models*
(https://arxiv.org/abs/2607.09921). The user's proposal is in `docs/proposal.pdf`.

Course feedback: demonstrate a simple baseline and a substantive improvement over
ordinary fine-tuning. Zhen specifically endorsed investigating a probability head
and joint training objective. Other feedback emphasized data feasibility, compute
constraints, and keeping postgame evidence out of pregame inputs and evaluation.

Language-model work: a compact Qwen2.5-0.5B-Instruct model with LoRA and an
outcome-trained probability head now has a matched stats/text pilot. Ordinary
generative SFT and probability-plus-explanation joint training are still future
comparisons. Brier is primary. The pilot uses a September 2024 checkpoint before
its dev/test dates; a larger study's backbone, GPU budget and fresh final test
remain undecided. Hindsight explanation supervision and market smoothing have
not been implemented. The original boosting result remains the stats reference.

## Completed dataset pilot

- 100 deterministic sampled games: 34 from November 1–10, 2023, and 33 each from
  November 1–10, 2024 and 2025; all 30 NBA teams represented.
- 435 historical results retained to construct prior-game features.
- 100/100 matched official NBA injury excerpts, each internally timestamped before
  the cutoff of scheduled tipoff minus 60 minutes. No matched excerpt contains
  `NOT YET SUBMITTED`. This does not prove every player status is complete.
- 100/100 ESPN historical preview-route responses rejected as non-preview
  articles (the endpoints returned recaps). Current news also appears in raw feeds.
- All 7 attempted dated Cavaliers game-note PDFs downloaded and extracted. Only
  the Cavaliers adapter was attempted; other teams' archive coverage is unknown.
- Narrative files are **candidates**, excluded from inputs pending publication
  timing and content review. Header spot checks match the intended games; PDF
  creation dates alone do not establish that they were public before the cutoff.
- Raw HTTP responses, URLs, retrieval times and SHA-256 checksums saved locally.
- Inputs and outcome labels separated. No language-model training, teacher generation,
  player-level box-score ingestion, odds purchase or future-game scheduler yet.

## Completed follow-up (2026-10-30)

- `config/seasonal.json`: 60 additional deterministic games, 10 from January
  15–24 and 10 from April 1–10 in each of 2024, 2025 and 2026. All 30 teams
  represented. Sampling ignores outcomes and source availability.
- 3,621 historical regular-season results from complete daily schedule responses
  through April 10 in each season. Transient network gaps were retried; the
  collector refuses to derive features from incomplete schedule downloads.
- **60/60** pre-cutoff, source-verified injury excerpts; zero `NOT YET SUBMITTED`
  flags. **60/60** ESPN preview-route articles were recaps and were rejected.
- Five additional Cavaliers PDFs downloaded as unapproved candidates. Visual
  checks of January/April injury pages and one note header are recorded in
  `reports/seasonal/manual_spot_checks.md`.
- Separate seasonal outputs in `data/processed/seasonal/` and
  `reports/seasonal/`. There are **160 distinct sampled games** across both
  audits; all five original pilot JSONL files remain byte-identical to canonical.
- `scripts/audit_narrative_sources.py` checks four explicit Knicks/Trail Blazers
  articles plus Orlando's notes index. Three Knicks versions have postgame
  modification timestamps and are rejected; one also has the wrong matchup in
  its opening paragraph. The Portland example passes automated metadata/header
  checks but remains quarantined pending content and independent archive review.
- Orlando's inspected index links to two rolling PDFs; this is not historical
  archive evidence. See `reports/narrative_sources/findings.md` and `coverage.json`.
- The article examples are purposive, not a coverage estimate. No narrative has
  been approved as model input; neither audit establishes full-season coverage.

## Code and verification

- `scripts/collect_pilot.py`: cached schedules, sample selection, prior features,
  injury timestamp parsing/matchup extraction, preview rejection, note candidates,
  machine-readable coverage and Markdown report.
- `scripts/validate_pilot.py`: checks all 100 joins, prior-only features, raw injury
  checksums, parsed game boundaries, source timestamps and label separation.
- `tests/test_leakage.py` and `tests/test_narrative_sources.py`: **19 passing
  tests**, including shared times, page continuations, DST, prior-only features,
  cross-year windows, schedule-failure gates, postgame revisions and exclusion
  of current/related news from extracted article text.
- Both validators pass: 100 original games and 60 seasonal games. The original
  pilot rebuilt offline; the seasonal offline rebuild reproduced all eight core
  JSONL/report files byte-for-byte. Article hashes/extraction/decisions verified.
- PDF dependencies are pinned in `requirements.txt`; fonttools supports the team
  notes' embedded fonts. The venv is recreated after relocation.

## Completed gradient-boosting baseline (2026-10-30)

- `scripts/run_gradient_boosting.py` and `config/gradient_boosting.json` use all
  3,621 cached historical results, not only the 160-game text audit. Both teams
  need five earlier same-season games: 246 warmup games excluded, 3,375 eligible.
- Eighteen numeric features: home/away prior-game count, season win rate,
  last-five win rate and mean point differential, rest days and back-to-back,
  plus home-minus-away differences. Only earlier ET dates; no raw ESPN content,
  target outcomes, injury/narrative text, odds, player data or team IDs in X.
- Train on 2023–24 (1,115), choose by 2024–25 validation Brier (1,123), then
  refit on those two seasons and test on 2025–26 (1,137). Coverage ends April 10.
- Four predeclared candidates; selected 100 depth-one trees, learning rate 0.05,
  minimum leaf size 20, log loss, subsample 1.0, seed 230. No random split, early
  stopping or probability calibration. Classification threshold stays 0.5.
- Test **Brier 0.209868**, **log loss 0.608507**, **ROC AUC 0.731198**,
  **accuracy 67.2823%**. Historical home-rate reference: Brier 0.247544,
  accuracy 54.9692%. Relative Brier reduction: 15.2%.
- 2,000 paired day-bootstrap resamples: Brier CI [0.200778, 0.219016]; GB minus
  reference CI [-0.046523, -0.028587]. Conditional on this fitted model/season,
  excluding training uncertainty and serial dependence across days.
- Reports, metrics/provenance, calibration CSV and plot in
  `reports/gradient_boosting/`. Saved model, separate inputs/labels, exclusions,
  split manifest and predictions in `data/processed/gradient_boosting/`.
- **25 tests pass**. `scripts/validate_gradient_boosting.py` verifies complete
  raw schedule lineage, prior-only features, disjoint splits, no test IDs in
  fitting/selection, and reproduces all held-out predictions/metrics from the
  saved model. Collection code was unchanged during this baseline task.
- `requirements-baseline.txt` pins the modeling environment; requires Python
  3.12+ (current worktree `.venv` is 3.14.6). Only the project venv was modified.
- The test season has been inspected more than once and is not an untouched final
  benchmark. Earlier test-game outcomes update subsequent rolling features, but
  the model is not refitted.
  Future text comparisons must use the same game cohort, and a final LLM test
  period must account for the chosen backbone's training cutoff.

```bash
uv pip install --python .venv/bin/python -r requirements-baseline.txt
.venv/bin/python scripts/run_gradient_boosting.py
.venv/bin/python scripts/validate_gradient_boosting.py
```

## Completed matched text experiment (2026-10-30)

- `config/text_ablation.json` and `scripts/run_text_ablation.py` build the exact
  140-game intersection of the audited injury-text sample and eligible baseline
  games. Every injury excerpt is re-extracted from its checksum-verified source
  PDF and checked against matchup/date and pregame cutoff. No narrative approved.
- User approved whole ET dates: train 108 (2023-11-03 through 2025-11-05), dev 14
  (2025-11-07 through 2026-01-16), test 18 (2026-01-17 through 2026-04-10).
  Exact 105/17/18 would divide a date. Previous 40/48/52 figures described the old
  season split, not a constraint on this separately trained experiment.
- Both arms use `Qwen/Qwen2.5-0.5B-Instruct`, pinned to September 25, 2024 revision
  `7ae557604adf67be50417f59c2c2f167def9a775`. The checkpoint is under ignored
  `data/models/qwen2.5-0.5b-instruct-20240925/`; downloads are about 1 GB.
- Both use identical initial weights, LoRA on q/v projections (rank 4, alpha 8,
  dropout 0), and a zero-initialized binary probability head: 271,232 trainable
  parameters each. Stats prompts serialize the same 18 features; the text arm
  adds the verified injury excerpt with team headings replaced by HOME/AWAY.
  Player names remain. No IDs, dates, outcomes or raw feeds enter prompts.
  All prompts fit without truncation: stats 222–253 tokens, text 284–571.
- Five epochs per arm, identical shuffled game orders and batch 4 / accumulation
  4; AdamW adapter rate .0001, head rate .001, weight decay .01, clipping 1,
  binary cross-entropy, seed 230. A common epoch minimizes mean dev Brier across
  both arms, giving equal passes through training data. Epoch **1** selected.
  No refit on dev, calibration or threshold tuning; threshold stays .5.
- `protocol.json` freezes sources, raw PDF hashes, initial model checkpoint and
  original baseline artifacts. `selection.json` locks the common epoch, selected
  adapters, dev curves/predictions and training membership before evaluation.
- On identical 18 test games: stats-only Brier **.236997**, log loss .666035,
  accuracy 55.6%; stats+injury-text Brier **.238987**, log loss .671024,
  accuracy 66.7%. Text minus stats Brier **+.001990**, paired date-bootstrap 95%
  interval **[-.035258, .037594]** (2,000 resamples across 12 dates).
  No clear evidence of better probability forecasts; accuracy rose by two games.
- Original boosting reference on the same 18 games: Brier .174847, accuracy
  77.8%, but its much larger training cohort makes this a reference rather than
  the matched ablation. Its original 1,137-game result/model are unchanged.
- **31 tests pass**. `scripts/validate_text_ablation.py --device mps` passes:
  rechecks raw sources/chronology, identical initialization and game order,
  common dev-selected epoch, and reproduces selected dev/test predictions and
  Brier differences from saved adapters. The baseline artifacts remain unchanged.
- Training and validation ran locally on the Mac GPU (48 GiB unified RAM).
  Sandbox access reports MPS unavailable; approved escalated commands could use
  it. `requirements-text.txt` pins the venv-only additions. A fixed seed does not
  ensure bitwise-identical retraining: MPS reports a nondeterministic gradient
  operation. Saved-model prediction replay passes within 1e-6.
- This is a compact learned-probability-head adaptation, not a full BAM replica:
  no research agents, hindsight report teacher, market smoothing or ensemble.
  Only injury text, one backbone/seed, tiny dev/test cohorts, and previously
  inspected test games. Do not infer that all pregame text lacks predictive value
  or tune further to the observed test. No additional model search was performed.
- Reports: `reports/text_ablation/`; inputs, separate labels, adapters, split
  manifest and predictions: `data/processed/text_ablation/`. The new runner/config
  are hashed, so changes require preparing and training again. Download and
  offline reproduction commands are in README. The report has full metrics.

```bash
.venv/bin/python scripts/run_text_ablation.py --stage prepare
.venv/bin/python scripts/run_text_ablation.py --stage train --device mps
.venv/bin/python scripts/run_text_ablation.py --stage evaluate --device mps
.venv/bin/python scripts/validate_text_ablation.py --device mps
```

Quick commands (from this worktree, which has the new code and data):

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_pilot.py
.venv/bin/python scripts/collect_pilot.py --offline
.venv/bin/python scripts/collect_pilot.py --config config/seasonal.json --offline
.venv/bin/python scripts/validate_pilot.py --config config/seasonal.json
.venv/bin/python scripts/audit_narrative_sources.py --offline
```

`--limit` is a debugging option that replaces processed audit outputs with a
subset. Run the full collector afterward to restore the 100-game outputs.

## Corrected hybrid design and rich-note discovery (2026-10-30)

- User explicitly means the original booster's probability as an input to the
  text forecaster, not serialized raw statistics in two separate LMs. Main
  comparison: original boosting versus boosting probability plus game notes.
  Keep both final outputs on the same held-out games. The earlier 140-game
  injury cohort does not establish game-note coverage or fix the new cohort size.
- Proposed architecture: reviewed note evidence + p_boost -> LM correction,
  with `p_hybrid = sigmoid(logit(p_boost) + correction)`. A zero-initialized
  correction preserves the baseline initially. This is documented, not implemented
  or trained. See `reports/game_notes_sources/hybrid_protocol.md`.
- Hybrid training needs chronological out-of-sample boosting probabilities with
  the original fixed boosting settings. Never feed it optimistic probabilities
  fitted on that target outcome or future games. Test IDs must be absent from
  both fitted systems. Freeze cohort/splits before fitting; retain approximate
  requested proportions with whole dates and sufficient history.
- `scripts/audit_game_notes.py`, `config/game_notes_sources.json` and
  `config/game_notes_reviews.json` preserve new sources, archived bytes and manual
  review decisions. `--offline` replays the audit. Only approved excerpts enter
  `data/processed/game_notes_sources/approved_excerpts.jsonl`; prior model inputs
  and collectors' narrative fields are unchanged.
- Dated CLE packet for 2025-02-21 NYK@CLE: 44 pages. Its exact PDF bytes match an
  Internet Archive capture at 21:37:43 UTC, before the 23:00 UTC cutoff. First
  page visually reviewed; date/matchup and 45-10 / 37-18 prior team records match
  cached history. **One first-page excerpt approved** for cohort construction;
  the remaining 43 pages need section review. Game `espn:401705363` belongs to
  the old boosting training data, so it is not eligible as a frozen-booster test
  example; its hybrid-training boosting feature must be out of sample.
- Rolling Knicks PDF exists (13 pages when fetched), but is not historical
  coverage proof. Jan 19, 2026 Knicks article was modified Jan 23 and remains
  rejected. Jan 28 article has a 19:02:45 UTC pre-cutoff archive and identical
  article text, but its opening has the wrong venue for NYK@TOR; it also has a
  stale heading. Keep that version quarantined despite valid archive timing.
- Archive verification checks actual redirect URL, Memento timestamp and cutoff,
  not just an availability API claim. Gzipped archived HTML is decoded for
  extraction while raw bytes/checksums remain preserved. The article parser now
  optionally accepts archived `Article` schema; default `NewsArticle` behavior
  is unchanged, and article-only extraction still excludes related/current news.
- **35 tests pass** plus both pilot/seasonal validators. Original boosting and
  previous LM protocol/source/artifact hashes remain valid. No new performance
  result is claimed. Reports and proposed protocol: `reports/game_notes_sources/`.

## Implemented game-note hybrid (2026-10-31)

- `scripts/collect_game_notes.py` checks the dated CLE pattern and rolling URLs
  for all 30 teams. The frozen audit contains 804 candidate versions, 482 verified
  archived PDFs and 404 distinct games with archived bytes. Actual replay URL,
  Memento time, cutoff and SHA-256 must agree; failed lookups stay explicit.
- `scripts/review_game_notes.py` and `config/game_notes_content_review.json`
  combine 19 visually reviewed publisher templates with per-document header,
  season, prior-record/game-number and time checks. This is not manual review of
  every sentence. 256 unique games are approved; 149 are before the backbone
  checkpoint and excluded from the hybrid. A Chicago packet with the wrong venue
  is explicitly excluded. Full PDFs remain preserved; model evidence is bounded
  front-page text with schedule rows removed, not the entire packets.
- `config/boosting_notes.json` / `scripts/run_boosting_notes.py`: original boosting
  probability plus notes -> zero-initialized LM correction to log odds. Exactly
  original boosting and the hybrid are the main comparison. Original model,
  reports and prior injury-pilot locks remain unchanged.
- 107 post-checkpoint games: train 80 (2024-11-04..2025-12-01), dev 13
  (2025-12-04..2026-01-31), test 14 (2026-02-01..2026-04-10). Whole ET dates;
  original-booster fitting IDs excluded from dev/test. Seven boosting lineages:
  six expanding-month fits with original fixed parameters plus the original
  frozen model after its last fitting date. All probabilities are target-excluded.
- Same pinned Qwen0.5B and LoRA rank 4, float32, BCE, five epochs, seed230,
  adapter/head LR .0001/.001, effective batch16, weight decay .01, clip1. Epoch0
  is the unchanged forecast and is eligible for dev-Brier selection; no dev refit,
  calibration or threshold tuning. Max prompt3072 / evidence2700 tokens.
- Preparation was fixed to lock only prepared inputs/labels/folds, not stale
  training checkpoints. Source hashes require reprepare/retrain after code changes.
  A long-context run used 31GiB and was interrupted before test evaluation.
  Activation checkpointing plus fixed padding per split reduced a two-step GPU
  benchmark to 4.6GiB driver allocation and ~2.9s for its second step. All input,
  label and split hashes stayed unchanged; epoch 1 scores reproduced within 2e-7.
- New validator replays raw PDFs, excerpt selection, numeric features, chronology,
  all OOF booster fits, checkpoint selection and saved dev/test predictions.
  Data validation passed107 games / 7 folds. 40 tests and both 100/60-game source
  validators pass. Selected-model replay passes within absolute tolerance 1e-6.
- Limits: 11/14 test packets are CLE; source coverage is selective. The historical
  test season was previously inspected. This compares a complete correction
  system, including additional player statistics and recalibration, not a clean
  isolation of narrative effects. No test-driven choices or third main model.

## Completed hybrid result and next work

- Five epochs completed; epoch 2 selected using dev Brier (.337811 versus the
  starting booster's .347535). No refit, test tuning or calibration.
- On identical 14 tests: original boosting Brier **.186296**, log loss .558920,
  accuracy 78.6%; hybrid Brier **.263225**, log loss .775461, accuracy 57.1%.
  Hybrid minus boosting **+.076929**, paired date-bootstrap 95% interval
  **[-.036185,.186632]** (2,000 resamples, 14 dates). The hybrid is worse by the
  point estimate; the small test sample does not establish a general text
  effect. Test mean predicted home probability fell to .2497 versus .5 observed.
- `scripts/validate_boosting_notes.py --device mps` passes: raw sources, numeric
  history, all 7 OOF fits, split boundaries, dev-selected epoch and saved dev/test
  predictions/metrics replay within 1e-6. All 40 tests and both source validators
  pass. Original boosting/prior injury-LM artifact locks remain intact.
- The requested implementation and experiment are complete; no active training
  job remains. New files and documentation are uncommitted in this worktree.
  Ignored data/models have not been transferred to canonical. No purchases,
  scheduling, remote training, commit or push.

The useful next study needs broader, more balanced game-note coverage and a fresh
held-out period. Do not tune this model to the observed 14 test results. The
2-system experiment does not isolate text from recalibration/additional numeric
facts. The earlier 140-game injury LM comparison remains historical only.

Expanded source report: `reports/game_notes_collection/findings.md`.
Experiment result/protocol/selection/validation: `reports/boosting_notes/`.
Inputs, separate labels, 7 booster folds, adapters and paired predictions:
`data/processed/boosting_notes/`. Final run log:
`/private/tmp/cs230-hybrid-efficient-run.log`.

## Full packets and evidence research (2026-10-31)

The user rejected the first-page restriction and wants emphasis on researching
the complete game notes. The old trained hybrid is unchanged and its result is
still a first-page pilot. The next design is all pages -> cited evidence report
-> forecasting with the original booster probability.

`scripts/prepare_full_game_notes.py` now extracts every page of all 107 selected
packets: 3,783 pages. It verifies raw archive timing/hashes, retains plain/layout
text, chunks complete pages without truncation, and prepares research prompts
that exclude target outcomes and booster probabilities. The final forecast would
receive the booster later. `--validate` checks sources, full page membership,
character coverage, requests and visual-transcription provenance. The
`--from-cache` option reselects text from verified extraction caches.

Some layout parses omitted content; the plain-text fallback recovers that text.
One garbled Cavaliers page was visually transcribed, pinned by source/page and
render hash in `config/full_game_notes_extraction_overrides.json`. Original
extractions remain. Images and remaining font-control flags are not all visually
audited. The full corpus is prepared, not semantically researched by a batch LM.

The assistant read all 15 extracted pages of the shortest training packet,
2024-12-02 MIA@BOS, and visually checked pages 3, 10, 11, 15. The worked example
contains 7 evidence claims, 18 quote-verified citations, page-by-page dispositions,
and distinctions between baseline-known rest, additional player workload, prior
injuries and current availability. It is an illustrative review, not a forecast.

See `reports/full_game_notes/{index,worked_example,research_protocol}.md`,
`coverage.json` and `validation.json`; all data under
`data/processed/full_game_notes/`. The earlier first-page inspection index remains
in `data/review/boosting_notes/` (now explicitly git-ignored). New tests cover
late/blank page inclusion, text gaps/truncation, layout fallback, citation source
identity and target/booster exclusion. No new forecaster training or test tuning.

Final preparation: 7,343 research chunks; 855 pages recovered via plain-text
fallback; 11 pages retain null-glyph review flags. All 47 tests, both pilot
validators, full-corpus validation and the worked-example citation checks pass.
The frozen baseline/hybrid source and artifact locks are intact. Logs:
`/private/tmp/cs230-full-notes.log` and `/private/tmp/cs230-full-notes-final.log`.

Next: choose and run a temporally appropriate research backend over all chunks,
merge/check cited reports and resolve visual quality issues before forecasting.
Keep the research stage observable; do not call token embeddings a completed
research report. Use a fresh final holdout for the subsequent model comparison.

## Automated research / revised forecaster implementation (in progress)

The user authorized the next research and training run. New scripts are
`run_game_note_research.py`, `check_research_quality.py`,
`run_researched_forecaster.py`, `validate_researched_forecaster.py`, and
`run_research_experiment.py`. The last script runs per-game research and support
checks, validates complete coverage, encodes all retained facts without truncation,
fits a regularized residual probability head, selects on dev Brier, and evaluates
the unchanged booster and revised hybrid on the same cohort. A zero correction
is eligible. The LM is frozen; only the head is trained. None of the revised
forecaster stages has run yet.

Local MLX dependencies are pinned in `requirements-research.txt`. September 18,
2024 Qwen2.5 7B/14B/32B four-bit checkpoints were downloaded and pinned, all before
the 107 games. Training-note trials exposed factual/table errors; see
`reports/game_note_research/quality_audit.md`. Rejected trials are preserved in
`data/probes/`. The latest checker reads the complete source page portion, not
only cited excerpts. A targeted training-note audit exposed remaining errors;
the source-quotation forecast design below is now selected. Current quality log:
`/private/tmp/cs230-research-quality-resume.log`. Current research inputs cover
3,783 pages in 3,858 page portions (11,570,203 prompt tokens).

57 tests and both pilot validators pass; original baseline/pilot artifact locks
remain intact. Cached sources contain no later unused approved cohort, so the
revised 80/13/14 comparison will be explicitly exploratory on the previously
inspected test games. A fresh final benchmark remains separate future work.

## Full research resumed (2026-11-01)

- Selected local four-bit Qwen2.5-14B checkpoint revision
  `dad510143ae5cdb1466778bde3161460f9e9f604` (September 18, 2024). Frozen model;
  only the eventual 5,121-parameter correction head is trained. Fixed L2 grid
  `[.001, .01, .1, 1, 10]` plus zero correction; dev Brier selection.
- Raw semantic checker passed **10/12** targeted training-note challenges and
  remains explicitly failed on two. Generated claims/scopes/caveats never enter
  forecast inputs. Instead use actual cited source lines, nearby context and
  headings, plus code-decoded table facts. The separate input-readiness guards
  pass. See `reports/game_note_research/{quality_audit.md,quality_gate.json}`.
- Miami adapter yields 51 facts from 30 tables/two packets. New independent
  `cavaliers_note_tables.py` yields 1,982 facts from 833 tables/49 packets, with
  missing participation details preserved as unknown. Cleveland facts are added
  during forecast preparation, after research, without changing its prompts.
- First run stopped after two completed games on malformed claim citations and
  metadata in game three. The validator now rejects individual malformed claims
  without dropping valid page evidence, and normalizes bare decision arrays.
  Original response text and generation-protocol hashes remain; compatible old
  responses are revalidated. History: `data/probes/research_protocol_history/`.
- All **68 tests pass**. Forecast-input preflight on the first three completed
  training packets fits every retained item and additional numeric fact in
  three chunks per game. Source/source-line checks and old model locks pass.
- Resumed full pipeline run ID `20261101T080419Z`; process output
  `/private/tmp/cs230-full-research-experiment-resume.log`. Stage logs/status:
  `data/processed/research_experiment/`. Completed games:
  `reports/game_note_research/progress.json`. Inspect these before restarting;
  the run may still be active. Revised embedding/training/evaluation have not
  run yet as of this entry. GPU research is the long stage.
- `validate_researched_forecaster.py` now also reconstructs actual evidence
  prompts and replays metrics/paired intervals. Keep these checks before calling
  the run complete. On completion update README, this file and canonical memory
  with actual selection/results; never present prior pilot scores as new scores.

## Cloud compute preparation (2026-11-01, local run still active)

The user asked to keep going and whether cloud compute would help. They confirmed
university/cloud credits and then selected **Modal**. The account authenticates
and bounded GPU work is authorized; do not ask again for provider permission.
The original local process continues; inspect live progress before restarting.
The newer Modal section below supersedes this section's initial untested status.

The measured initial uncached packets took 371/429/409 seconds, suggesting about
11 hours of local research remaining at that point. MLX 0.32.4 supports Linux
CUDA and quantized kernels, so first benchmark the same pinned model/backend
API on an available NVIDIA GPU. Do not assume exact cross-hardware outputs or
replace the checkpoint without recording a new experiment. See
`reports/game_note_research/compute_options.md` for primary sources and options.

`scripts/benchmark_research_device.py` prepares/checks/runs an isolated 15-page
training-note benchmark plus all twelve factual challenges and an encoder kernel
check. `requirements-research-cuda.txt` is separate from the active Mac environment.
Prepared bundle `data/probes/research_compute_bundle.tar.gz`: ten files, 49,894
bytes, no target labels/boosting probabilities/credentials. The public checkpoint
is downloaded separately (7.75 GiB). Hash/request/metadata checks pass; CUDA
execution is untested. Its result stays separate from main experiment outputs.

`scripts/export_research_compute.py` prepares a research-only snapshot with all
3,858 units and saved responses, preserving raw response provenance. Snapshot
`data/probes/cloud_research_transfer/20261101T083913Z.tar.gz`: 920 files, 7,589,866
bytes; all archive/source/protocol hashes and 300 cached extractions verified.
No raw ESPN, target labels, boosting features, credentials or model weights.
Refresh this snapshot at any actual cutover to retain further local progress.
Remote command would be `--stage research` only; returned outputs must pass the
local original-source validator before forecasting. Keep one verified backend
for all encoder vectors. Preserve local computation until remote access and the
benchmark work; never run two writers into the same results directory.

## Modal implementation and benchmark (2026-11-01)

- Existing Modal profile/workspace `tsini` works; no secrets printed or copied.
  `modal==1.6.3` added to the project venv with all 59 existing packages unchanged.
  Private v2 volumes `cs230-nba-research-models` and `cs230-nba-research-runs`.
  No credit purchase/account upgrade; actual credit balance unknown.
- `scripts/modal_research.py` successfully ran the exact pinned MLX/CUDA 14B
  model on H100 80 GB. Results: `data/probes/modal_benchmark/20261101T090405Z/`.
  MIA training packet: extraction 144.1 s, checking 30.0 s, versus local
  125.0/79.4 s. 9/15 exact extraction matches, finite 5120-dimensional encoder
  output; the same two factual challenges fail. About 1.17x total speedup only.
- Earlier probe failures were a symlinked venv check and mixed CUDA libraries;
  both fixed. `run_logged` resolves all pip NVIDIA libraries before base CUDA.
  Completed apps release GPUs automatically. Do not restart the slow MLX full
  batch (`modal_research_batch.py`) merely because the probe worked.
- New isolated `scripts/modal_research_vllm.py` / `research_vllm_worker.py` use
  vLLM 0.32.0, Transformers 5.19.0, NumPy 2.4.6 in a cloud-only venv. Original
  pinned quantized weights are expanded directly to FP16 on CPU, never replaced by a
  newer checkpoint. Conversion hashes and actual versions/settings accompany
  every new response. Prepared input token hashes must match local exactly.
  First image build had a NumPy/Numba constraint conflict, corrected locally.
- MLX conversion stalled with no file/IO growth. Task-owned attempts were stopped.
  `convert_research_weights.py` now unpacks affine 4-bit values with FP32 multiply
  and add, then rounds to FP16; this matches MLX GPU dequantization exactly on
  the first row of **all 338 matrices**. Evidence and reference hashes:
  `data/probes/dequantization_gpu_check.json`. CPU MLX uses different intermediate
  rounding and is not the conversion reference. Full conversion checks all 338
  reference rows again remotely, saves staged shards, then copies to the volume.
- Current vLLM benchmark build/run: tool session 26578, stdout
  `/private/tmp/cs230-modal-vllm-benchmark-v7.log`. Inspect before duplicating.
  No vLLM full-research job yet. Local pipeline PID 96599 / child 96600 remains
  active while the alternative is tested. Refresh transfer snapshot at cutover.
- `import_modal_research.py` verifies source/return hashes, conversion/runtime
  approval, preserves completed local responses and requires a non-benchmark
  completed run. Source validator checks imported response identities; forecaster
  protocol also locks cloud plan/import registry. **Plan not written yet**:
  create it from actual passing training-only benchmark identity, snapshot hash,
  worker hash, conversion, settings and versions before full cloud execution.
- 68 existing tests plus three cloud-provenance and one packed-weight test pass. No new head or
  revised test results yet. Finish research -> import -> source validator ->
  forecaster preparation/encoding/training/dev selection/evaluation/replay.

## Modal cutover — active full experiment (2026-11-01 09:47 UTC)

- vLLM benchmark `20261101T093252Z` succeeded: 22.72 s extraction +18.10 s
  checking versus local 204.4 s, about 5x faster. Same two known factual failures;
  all schema/source checks pass. Benchmark returned files under
  `data/probes/modal_vllm/20261101T093252Z/returned/`. Model weights are direct
  FP16 expansion of the frozen four-bit checkpoint, with all 338 matrix sample
  rows matching MLX GPU, original scaling precision preserved. Converted model
  path on Modal: `/models/qwen2.5-14b-instruct-4bit-20240918-dequantized-fp16-direct`.
- Local PID 96599/96600 stopped deliberately at **18/107 complete**. Do not
  restart that writer while cloud research runs. Snapshot
  `data/probes/cloud_research_transfer/20261101T094350Z.tar.gz` (1,879 files,
  8,163,544 bytes, SHA ad08c3248a92f12730c9650098124c3551255e82d603103f8929465d9c5df420).
  An auto-review initially rejected export as potentially sensitive. I audited
  every member/hash, proved the 107 public NBA sources and absence of private
  docs/labels/credentials, then retried the SAME action with that evidence; it
  was approved. `reports/game_note_research/modal_transfer_audit.json` records it.
  No approval question is pending and no further permission is needed.
- Active full run **20261101T094719Z**, tool session **72110**, stdout
  `/private/tmp/cs230-modal-full-experiment.log`, app
  https://modal.com/apps/tsini/main/ap-xk7u4mRqgr7A12pZ2Q3Dfy .
  Driver: `modal run scripts/modal_research_vllm.py --snapshot
  data/probes/cloud_research_transfer/20261101T094350Z.tar.gz --finish-local`.
  Status/results: `data/probes/modal_vllm/20261101T094719Z/`. GPU bound 6900 s;
  preemption resumes written responses. Inspect status/log before any restart.
- `--finish-local` automatically imports the complete return (strictly preserving
  completed local responses), then runs `run_research_experiment.py --stage all`:
  cached research replay -> full source validation -> prepare -> local MLX
  embeddings for all games -> head fit/dev selection -> evaluation/replay/index.
  Driver must remain running for local follow-up. Partial/failed remote runs are
  preserved and require inspection; they cannot be promoted as complete.
- `cloud_execution_plan.json` freezes the passing benchmark, worker/converter,
  actual vLLM versions/settings, source checkpoint, snapshot hash and choice.
  Scientific config, original booster, prompts, cohort and dev rule unchanged.
  Research arithmetic differs across the retained local cache and cloud runtime;
  response execution identities record this. All encoder vectors stay on MLX.
- All **72 tests pass**, `git diff --check` passes. No revised head or Brier
  result yet. Setup usage at 09:41 UTC was $1.31 metered, credited, $0 billed then;
  remaining balance unknown. No account upgrade, credit purchase, commit or push.

## Resume after incomplete verification response (2026-11-01 18:30 UTC)

The user asked whether the run finished. It had stopped at 45/107 complete at
10:02 UTC, before any head training. Request `espn-401705181-u040-v000` returned
seven verdicts for eight claims three times. No output truncation: ID7 was
omitted; the old generic retry error did not identify the missing ID. The app
stopped and returned its partial cache, so it was not consuming GPU overnight.

`research_vllm_worker.py` now gives retries the exact expected count/IDs and
missing IDs; it never synthesizes a verdict. Attempt filenames include execution
identity to retain older failures. Initial prompts, model, source protocol,
inference settings and experiment are unchanged. Four regression tests include
the complete 7-to-8 retry and preservation of an earlier attempt; **76 tests pass**.
The old execution plan is archived in the failed run directory; the current plan
accepts the previous worker hash for its unchanged cached generations and records
the narrow repair. Importer accepts both recorded worker implementations.

**Active resume:** `20261101T183024Z`, tool session **15226**, app
https://modal.com/apps/tsini/main/ap-5rdJuJFYrSD9wXS4zYGvhi , stdout
`/private/tmp/cs230-modal-full-experiment-resume.log`. It reuses the SAME audited
snapshot and remote directory `/runs/vllm/20261101T094719Z/nba_research` via
`--resume-run 20261101T094719Z --finish-local`. No duplicate writer; 45 completed
games retained. Output/status go to `data/probes/modal_vllm/20261101T183024Z/`.
Automatic import/validation/encoding/training/dev selection/evaluation remain
enabled. Monitor past the failed request and through final validation. No revised
result yet; keep the user's status answer honest and do not repeat old scores.

Live confirmation: the missing-verdict request completed in 16.1 seconds on
resume, and game46 completed. New uncached game47 is processing. The fix has
therefore passed its real failure case, not only the unit regression.

## Reusable Modal deployment (2026-11-01, side conversation)

The user requested one deployment instead of new temporary apps. I deployed
`cs230-nba-research-vllm` in environment `main`, app
`ap-GOc0WPE2sdK0dNlV20s7Vq`:
https://modal.com/apps/tsini/main/deployed/cs230-nba-research-vllm .
Its functions have zero idle containers, preserve the original CPU/H100 limits
and shared model/results volumes, and were verified through metadata lookups
with no conversion or GPU invocation. All **80 tests** pass.

For every subsequent job, run `.venv/bin/python scripts/modal_research_vllm.py`
with the existing `--snapshot`, `--resume-run` and `--finish-local` options.
**Do not use `modal run` for new jobs.** The Python CLI uses named deployed
function handles and fails if the deployment is missing. Use `--check-deployment`
for a read-only registration check. If cloud worker/converter code changes,
redeploy in place with `.venv/bin/modal deploy --env main
scripts/modal_research_vllm.py`; source hashes are checked before GPU submission.

The active resume `20261101T183024Z` in temporary app
`ap-5rdJuJFYrSD9wXS4zYGvhi` was left running untouched. It remains separate until
completion; no research was restarted by this side conversation. Old stopped
app records are retained. The unrelated `blindfold-segmented-b200` deployment
was also left untouched. README and compute options document the new commands.

## Persistence and current recovery (2026-11-01 19:23 UTC)

The user explicitly requested staying through every game, training, evaluation
and validation, rather than ending after launching a background run. Continue
monitoring and repairing failures until the actual verified result exists.

The 18:30 resume lost its client heartbeat at 18:55 (83 games logged; later
cache replay recovered 87). Named deployment run 20261101T190811Z reached 100/107
and was cancelled during an update. The encoder timing probe then completed:
48.47s local vs 45.89s CUDA, cosine .99998739. No meaningful speedup; keep ALL
experiment embeddings on the original local MLX backend. Probe outputs are
isolated under data/probes/encoder_device_probe; they never enter the head.

CURRENT run 20261101T192323Z, local tool session 6180, log
/private/tmp/cs230-modal-final-resume.log, call fc-01M6C92QVHNNX0DR222RPA78T4.
It uses the same named cs230-nba-research-vllm deployment, snapshot and original
remote root, with inspected --invocation-budget 4 and --finish-local. No more
redeployments during this call. The CLI now spawns and saves a retrievable call
ID before waiting. Automatic import/source validation/local MLX encoding/head
fit/dev selection/test evaluation/replay/index are enabled. If the local driver
fails, retrieve that SAME call before considering another invocation.
All84 tests and the pilot validator pass. Research worker and scientific config
unchanged; only bounded recovery/submission and the isolated encoder probe were
added. No revised forecast result yet. No purchases, account changes or Git push.

## Cloud encoding after laptop sleep (2026-11-01 20:14 UTC)

Research is COMPLETE for all107 games, 3783pages/3858units. Final source return
20261101T192323Z imported5030 cloud responses; source/response/line coverage
validation passed. Retained8092 evidence items plus1982 Cleveland table facts
produce10074 items in352 complete chunks/1749153tokens.

Local pipeline20261101T193059Z was stopped deliberately during encoding after
repeated maintenance sleep cycles on battery were confirmed in pmset logs. All
51 completed local vectors, inputs and protocol are archived at
`data/probes/local_encoder_interrupted/`. No head was fitted or test evaluated.
Do NOT mix these local vectors into the new run.

ACTIVE pipeline20261101T201235Z, tool session27304, stdout
/private/tmp/cs230-full-cloud-encoder-pipeline.log. Source validation and prepare
passed again; new input JSONL is byte-identical to the archived local input.
The frozen encoder_execution_plan.json now selects uniform MLX CUDA for ALL107
games with two bounded H100 workers in the SAME named Modal deployment. Local
and remote checkpoint/versions/tokens/chunks/pooling and output vectors are
checked; protocol locks encoder helper/worker/plan. Training/dev-selection
settings and original booster are unchanged. Arithmetic differs from Metal
(cosine.99998739 in the earlier training-only probe), so all vectors are rebuilt.

Cloud calls in data/probes/full_encoder_cuda_20261101/batch_*/call.json:
batch0 fc-01M6CBYM70W9MVT84SKX0EEKZ9, remote encodings/20261101T201356045901Z;
batch1 fc-01M6CBYM2DG5Y3EGCD5J1QPGRZ, remote encodings/20261101T201356044918Z.
Each call runs at most6900s and has zero idle containers. Calls are spawned and
reattachable; do not redeploy or launch duplicates. If the local driver loses
network, rerun the unchanged pipeline/helper to attach saved IDs and cache.
All85tests pass, including mixed-backend/vector/token tampering rejection. The
pipeline will merge validated cloud vectors, fit/dev-select/evaluate/replay/index
automatically. Keep going through final validation as the user explicitly asked.

## Final completion (2026-11-01 20:21 UTC)

The active pipeline described above is now COMPLETE, with all eight stages in
`data/processed/research_experiment/status.json`. Session27304 exited0. Two CUDA
encoder calls finished normally (54/53games,369.6/374.8s); source-to-token/vector
checks passed, then head fit/dev selection/evaluation/replay/index completed.
The final result and limits are summarized at the top of this file and inREADME.
No running project compute remains. The earlier Metal51-vector run stays archived
and excluded. No changes to the scientific inputs/split/objective/penalty grid;
all85tests and validators pass. No purchase, account upgrade, commit or push.

## Evidence browsing (2026-11-01, after completion)

The user could not open local Inputs links in the Markdown index. I added
`scripts/browse_evidence.py`, a read-only loopback viewer for all107 games with
Inputs/Research/Tables navigation, search and original-file downloads. It is
RUNNING at http://127.0.0.1:8765/ in tool session31455, log
/private/tmp/cs230-evidence-viewer.log. Superset browser pane
pane-519ec36e-baf1-4540-836c-32da6da6fc11 is open in this workspace. Leave it
running so the user can inspect evidence; no model/GPU jobs were restarted.
Browser Inputs text matched the original32872-character first-game file exactly
(SHA82df3b23e439ed6bca8fd8add30957795032ce3de2367153cffdede7eb333e65).
All107 games and three document paths per game are present. Research and Tables
routes returned200 and the user began navigating them. README and evidence_index
link to the viewer. If the server stops, restart with the repository.venv command
above. Frozen data, training code, protocols and results were not changed.
