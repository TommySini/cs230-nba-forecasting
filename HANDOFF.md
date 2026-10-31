# CS230 NBA forecasting — handoff

## Current request and location

The user requested and completed a **matched statistics-only versus
statistics-plus-pregame-text language-model experiment** on the 140 eligible
text-covered games. They explicitly chose a compact language-model pair and a
whole-date chronological **108 train / 14 dev / 18 test** split. Details below.
The original gradient-boosting baseline remains intact; the earlier expanded
boosting hyperparameter/threshold experiment remains removed at the user's
request. No purchases, scheduling, agent launch, remote training or GitHub push.

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

## Most valuable next work

1. Scale verified pregame injury reports to the baseline's eligible games and
   build per-game text-plus-statistics records with outcomes stored separately.
   The 160 audited injury excerpts cover only 140 of the 3,375 baseline games;
   no richer narratives are approved. Measure season/team coverage and compare
   all models on identical eligible games.
2. Plan a larger matched study with a fresh final test before further training.
   The initial compact probability-head pilot is complete; retain its recorded
   protocol and inconclusive result. Verify any future backbone's cutoff and
   preserve the disclosure of previously inspected test games.
3. Add ordinary generative SFT and then the proposed joint probability/explanation
   objective after expanding the cohort and defining valid explanation targets.
   These are not implemented. Keep stats/text arms matched and Brier primary.
4. Expand Portland's dated articles using a schedule-defined sample; investigate
   independent pregame snapshots and review content. Keep unverified documents
   quarantined. Current audits do not establish full-season narrative coverage.
5. Consider Stanford Libraries access or prospective capture if historical text
   remains insufficient; neither is set up. Library access information:
   https://guides.library.stanford.edu/gearup/lexisnexis.

Optional paid sources researched, not purchased: BALLDONTLIE $9.99/month for player
game stats; The Odds API at https://the-odds-api.com/ $30/20k credits or $59/100k
credits, with 10 credits per historical region/market query. Recheck prices before
buying. SportsDataIO's historical injuries are game-start snapshots, not a complete
intraday status history. NBA stats endpoint timed out in the initial probe, so the
pilot uses ESPN's daily scoreboard API for results.

## Superset / conversation continuity

The latest work remains uncommitted in the worktree above. A concise continuation
note is saved in the canonical repository's `.remember/remember.md` with that
location. Git-ignored data is not transferred by a merge or a new worktree.

The original relocation retained the scratch-session path as a compatibility
symlink to the canonical checkout. That is separate from this worktree.

To use the expanded audit from the canonical checkout, transfer the ignored data
separately when integrating the reviewed code changes. A fresh Git worktree
contains committed code, but not git-ignored `data/` or `.venv`. A new agent can
continue from this handoff without the full transcript.
