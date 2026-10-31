# CS230 NBA forecasting: dataset audit and baseline

A reproducible **data feasibility audit and statistics-only gradient-boosting
baseline**. The text dataset and final language-model benchmark are still under
development. Uses free public source responses; no paid accounts, API keys,
teacher-model calls or purchases are required.

## Gradient-boosting baseline

The completed baseline uses 3,375 eligible games from the 3,621 cached historical
results. Both teams must have five earlier games in the same season. It uses 18
numeric prior-game features, with no injury/narrative text or target-game stats.

- Train: 2023–24, 1,115 games; select among four candidates on 2024–25, 1,123 games.
- Refit the selected model on those two seasons; test on 2025–26, 1,137 games.
- Test **Brier 0.2099**, **log loss 0.6085**, **ROC AUC 0.7312**, **accuracy 67.3%**.
- Historical home-rate reference: Brier 0.2475 and accuracy 55.0%; boosting reduces
  Brier by 15.2%. The paired day-bootstrap Brier difference interval is
  [-0.0465, -0.0286], conditional on the fitted model and observed season.

This original model is the retained baseline: 100 depth-one trees, learning rate
0.05, minimum leaf size 20, `log_loss`, subsample 1.0, seed 230, no early stopping
or probability calibration, and decision threshold 0.5.

See [the full baseline report](reports/gradient_boosting/findings.md),
[metrics and provenance](reports/gradient_boosting/metrics.json), and
[calibration plot](reports/gradient_boosting/baseline.png).
The baseline includes eligible games through April 10 of each season. The
160-game text audit is a separate sample, with 140 games overlapping the baseline;
future text comparisons must evaluate both models on the same eligible games.
The test season has been inspected more than once. This is not an untouched final
LLM evaluation split.

The pinned modeling environment requires **Python 3.12+** (tested in this
worktree's Python 3.14.6 `.venv`):

```bash
uv pip install --python .venv/bin/python -r requirements-baseline.txt
.venv/bin/python scripts/run_gradient_boosting.py
.venv/bin/python scripts/validate_gradient_boosting.py
.venv/bin/python -m unittest discover -s tests -v
```

Training runs entirely from cached data. Configuration is in
`config/gradient_boosting.json`; the model, separated features/labels, split
manifest and predictions are saved under `data/processed/gradient_boosting/`.

## Matched language-model text experiment

This separate pilot uses the 140 games with both baseline statistics and verified
pregame injury text. The user approved chronological **108 train / 14 dev / 18
test** splits, keeping complete ET dates together. The nominal 75%/12.5%/12.5%
split would divide a game date at the training boundary. The older 40/48/52 counts
were memberships under the original season split, not a requirement for this run.

Both arms use the same September 2024 Qwen2.5-0.5B-Instruct checkpoint, LoRA
adapters and a binary probability head. They have identical initial trainable
weights, game ordering, optimizer settings and five-epoch training budgets. One
arm receives the 18 numeric features; the other receives the same features plus
the verified injury excerpt. Team headings become HOME/AWAY; dates, game IDs,
outcomes and unapproved narratives never enter the prompts. No text is truncated.

One common epoch is selected by average dev Brier across both arms, then frozen
before test evaluation. There is no refit on dev, calibration or threshold tuning.
The threshold stays 0.5. This adapts the language-model forecasting idea; it does
not reproduce BAM's research agents, hindsight-guided report supervision or
forecast ensemble. The original boosting model remains a separate reference.

The test contains only 18 games, already inspected in prior baseline evaluations;
this is an exploratory pilot, not a fresh final benchmark. A paired date-bootstrap
interval describes the Brier difference conditional on these fitted models.
The fixed seed does not guarantee bitwise-identical retraining on the Mac GPU,
which reports a nondeterministic gradient operation; saved predictions are checked
by reloading the selected adapters.

The dev rule selected epoch 1 for both arms. On the identical 18 test games:

| Matched model | Brier ↓ | Log loss ↓ | Accuracy at 0.5 |
| --- | ---: | ---: | ---: |
| Statistics only | 0.2370 | 0.6660 | 55.6% |
| Statistics plus injury text | 0.2390 | 0.6710 | 66.7% |

Text-minus-statistics Brier difference: +0.00199, with a paired date-bootstrap
95% interval [-0.03526, 0.03759]. This pilot gives no clear evidence of improved
probability forecasts from injury text. The accuracy increase is two games;
it does not establish better probabilities. The original boosting reference
scores 0.17485 Brier on these same 18 games, but used a different, much larger
training cohort. All 31 tests and the matched-experiment validator pass.

Config: `config/text_ablation.json`. Frozen protocol, selection, learning curves
and results: [the full matched report](reports/text_ablation/findings.md). Separate inputs,
labels, model adapters and predictions: `data/processed/text_ablation/`.

Install and download the pinned public checkpoint once:

```bash
uv pip install --python .venv/bin/python -r requirements-text.txt
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_dir="data/models/qwen2.5-0.5b-instruct-20240925",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE", "README.md"],
    token=False,
)
PY
```

All experiment stages then run offline:

```bash
.venv/bin/python scripts/run_text_ablation.py --stage prepare
.venv/bin/python scripts/run_text_ablation.py --stage train --device mps
.venv/bin/python scripts/run_text_ablation.py --stage evaluate --device mps
.venv/bin/python scripts/validate_text_ablation.py --device mps
```

Use `--device cpu` on systems without an accessible Apple GPU. Frozen source hashes
include the runner/config and checkpoint; code changes require a new preparation
and training run. Models and datasets remain git-ignored.

## Run

The dataset collectors require Python 3.9+. Install the PDF dependencies in a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/collect_pilot.py
.venv/bin/python scripts/validate_pilot.py
.venv/bin/python -m unittest discover -s tests -v
```

Alternatively, with uv:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/collect_pilot.py
```

Once collected, reproduce the outputs without network requests:

```bash
.venv/bin/python scripts/collect_pilot.py --offline
```

`--stage schedules` stops after building the historical game list and sample.
`--limit 3` runs a small debugging audit and **replaces the processed audit with
that subset**; rerun without a limit to restore the complete 100-game outputs.
The collector caches successful responses and permanent HTTP failures, verifies
cached checksums, throttles requests, and checkpoints processed files after each
game. To retry a cached HTTP failure, remove its `.meta.json` file first. Do not
remove successful cache entries if you need exact offline reproducibility.

The seasonal follow-up uses separate outputs and reuses the raw-response cache:

```bash
.venv/bin/python scripts/collect_pilot.py --config config/seasonal.json
.venv/bin/python scripts/validate_pilot.py --config config/seasonal.json
.venv/bin/python scripts/audit_narrative_sources.py
```

Add `--offline` to either collector to reproduce from cached sources. The seasonal
config selects 10 games in January 15–24 and 10 in April 1–10 of 2024, 2025 and
2026: 60 additional games, with complete daily schedule history from the preceding
October 1. January and April belong to the season starting in the previous year.
Sampling uses seed 230 and ignores outcomes and source availability. Outputs go to
`data/processed/seasonal/` and `reports/seasonal/`, preserving the original pilot.

The additional narrative audit checks four search-discovered Knicks/Trail Blazers
articles and Orlando's current notes index. It is a separate, purposive source
investigation, not a coverage estimate over the random game sample. It joins
articles to the complete seasonal schedule, checks both JSON-LD and CMS
publication/modification times against the cutoff, and checks the header's teams
and game date. Only article blocks are extracted; navigation and related news
are excluded. Even a passing article remains quarantined pending content and
independent archive review. Run it after the seasonal schedule has been collected.

## What is sampled

`config/pilot.json` selects 34 games from November 1–10, 2023 and 33 each from the
same window in 2024 and 2025, using seed 230. Regular-season schedules from October
1 through November 10 in each year provide the preceding results. Sampling is
independent of outcomes, document availability, and model performance.

This early-season sample is intentionally small. It does not establish archive
coverage for entire seasons. Before scaling, audit midseason and late-season
windows too. Select an actual model and verify its training cutoff before setting
the held-out evaluation period; these games are not yet contamination-cleared.

## Files

| Path | Purpose |
| --- | --- |
| `reports/pilot_findings.md` | Findings and next steps from the completed run |
| `reports/coverage.json` | Machine-readable coverage counts |
| `reports/game_audit.csv` | Per-game timestamp and source decisions |
| `reports/seasonal/` | Separate January/April coverage, findings, and per-game audit |
| `reports/narrative_sources/` | Additional team-page timing and archive investigation |
| `data/raw/` | Downloaded source bytes and provenance sidecars |
| `data/processed/game_history.jsonl` | Historical game list, **including outcomes** |
| `data/processed/sample_manifest.jsonl` | Fixed audit sample, **including outcomes** |
| `data/processed/inputs.jsonl` | Pregame feature records and matched injury excerpts |
| `data/processed/labels.jsonl` | Target outcomes, separate from inputs |
| `data/processed/audit.jsonl` | All source decisions and attempted URLs |
| `data/processed/narrative_candidates/` | Unapproved team-note text; not model input |
| `data/processed/seasonal/` | Follow-up inputs, labels, history, manifest, and audit |
| `data/processed/narrative_sources/` | Article audit and quarantined text; not model input |

Raw and processed data are git-ignored. The collector and reports can be shared
without accidentally publishing an archive of third-party content. Check source
permissions before redistribution or training on licensed narrative material.

## Sources and known limitations

**Game results:** ESPN's public daily scoreboard endpoint. It is undocumented and
may change. NBA's `leaguegamelog` endpoint timed out during the initial probe, so
the pilot uses ESPN. We retain only identity, historical scheduled start, final
scores, and season when normalizing games. Live/current news, rosters, standings,
leaders, odds, and target-game box scores never become model features. Historical
start times are taken from the retrieved schedule, not an archived pregame
schedule version; rescheduling must be audited for a final benchmark.

**Prior-game features:** calculated from earlier ET calendar dates in the same
season. Includes win rate, last-five point differential, and rest. Excluding
same-date games conservatively avoids assuming a game had finished just because
its scheduled start preceded another game's cutoff. This pilot does not collect
player-level box scores. Source game IDs allow every aggregate to be reproduced.

**Injuries:** official `ak-static.cms.nba.com/referee/injury/` PDFs. The internal
report timestamp, interpreted in America/New_York with daylight saving, must be
no later than scheduled tipoff minus 60 minutes. The URL filename alone is not
trusted. The parser respects game-date and matchup boundaries, including omitted
repeated times and page-spanning rows. Output is a game-specific text excerpt,
not a fully normalized player injury table. Missing team rows do not imply a
healthy roster; `NOT YET SUBMITTED` remains explicitly flagged.

The URL search probes at most four hourly slots and three filename formats per
slot, stopping after an eligible snapshot within an hour of the cutoff. It is
**not an exhaustive archive discovery or a guarantee of the latest snapshot**.
HTTP 403/404 means a candidate could not be fetched, not that no report exists.

**ESPN previews:** a historical URL labeled `preview` can return an article whose
type is `Recap`, alongside current news. The collector rejects non-preview
articles, wrong games, missing timestamps, and post-cutoff modifications. Even
eligible preview candidates require a content review and are not included by
default. Raw responses contain postgame material and must never be used directly
as model inputs.

**Team notes:** only a known dated Cavaliers PDF URL pattern is implemented in
this pilot. Other teams are explicitly marked `not_attempted_no_adapter`.
Successful PDFs are narrative candidates only. PDF creation/modification metadata
does not prove public availability before the cutoff. Validate the matchup,
publication timing, contents, and usable sections before promoting them to inputs.
This adapter's hit rate is not a leaguewide text-coverage estimate.

**Additional team articles:** `scripts/audit_narrative_sources.py` tests explicit
source URLs in `config/narrative_sources.json`. An early publication date does not
make a later revision safe: any post-cutoff modification disqualifies that
retrieved version. Publisher metadata obtained retrospectively does not establish
an independently archived pregame snapshot. The article audit does not change the
pilot's `narrative_text` fields or its Cleveland-only PDF adapter.

## Next modeling and data decisions

1. Build a training-ready text dataset for the baseline's eligible games. Scale
   the verified pregame injury-report collection, retain timestamps/provenance,
   and report coverage by season/team. There are 160 audited injury excerpts but
   only 140 matching baseline games; no richer narrative is approved yet.
2. Plan a larger matched study before further training. The first Qwen 0.5B
   probability-head pilot is complete; broader model/compute choices and a fresh
   final test remain open. Check each backbone's cutoff and avoid tuning to this
   pilot's already-observed test results.
3. Add the planned ordinary generative fine-tuning comparison to the learned
   probability head after expanding the data. Keep stats-only and stats-plus-text
   arms matched on training/test games; evaluate Brier primarily, with log loss
   and accuracy. The original boosting baseline remains the statistical reference.
4. Once that comparison works, test the proposed joint probability/explanation
   training objective. Explanation targets and richer narratives still need a
   defined collection and validation protocol.
5. Continue dated team-article discovery and independent pregame-version checks;
   keep unverified content quarantined. Prospective capture or licensed library
   sources remain options if historical narrative coverage is insufficient.

Relevant source documentation:

- https://nba.balldontlie.io/
- https://github.com/swar/nba_api
- https://official.nba.com/nba-injury-report-2026-27-season/
- https://guides.library.stanford.edu/gearup/lexisnexis
- https://the-odds-api.com/liveapi/guides/v4/
