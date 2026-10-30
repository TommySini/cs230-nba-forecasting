# CS230 NBA forecasting — handoff

## Current request

The user asked to move this work out of the Superset scratch session into
`Desktop/Repos`, and will open the repository under Superset Projects themselves.
Use the project-local `.venv` for all future Python work. No Superset project
registration, agent launch, paid subscription, or GitHub push is needed.

Canonical path: `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting`.

## Research objective

Forecast NBA home-team win probabilities from pregame text and prior-game stats.
The proposal adapts BAM's *Global Merger-Arbitrage Forecasting with Language Models*
(https://arxiv.org/abs/2607.09921). The user's proposal is in `docs/proposal.pdf`.

Course feedback: demonstrate a simple baseline and a substantive improvement over
ordinary fine-tuning. Zhen specifically endorsed investigating a probability head
and joint training objective. Other feedback emphasized data feasibility, compute
constraints, and keeping postgame evidence out of pregame inputs and evaluation.

Recommended model plan, not implemented: small open-weight LM with LoRA; compare
standard SFT, an outcome-trained probability head, and probability-plus-explanation
joint training. Add a statistics-only logistic baseline. Brier score is primary;
use chronological held-out evaluation after the backbone's training cutoff. Start
with actual binary outcomes; hindsight explanations and market smoothing are later
ablations, not prerequisites. No backbone, GPU budget, or final splits chosen yet.

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
- Inputs and outcome labels separated. No model training, teacher generation,
  player-level box-score ingestion, odds purchase or future-game scheduler yet.

## Code and verification

- `scripts/collect_pilot.py`: cached schedules, sample selection, prior features,
  injury timestamp parsing/matchup extraction, preview rejection, note candidates,
  machine-readable coverage and Markdown report.
- `scripts/validate_pilot.py`: checks all 100 joins, prior-only features, raw injury
  checksums, parsed game boundaries, source timestamps and label separation.
- `tests/test_leakage.py`: 8 passing regression tests, including shared printed
  matchup times, page continuations, DST and future/target outcome exclusion.
- Full collector successfully rebuilt offline; validator passed all 100 games.
- PDF dependencies are pinned in `requirements.txt`; fonttools supports the team
  notes' embedded fonts. The venv is recreated after relocation.

Quick commands (from canonical repo):

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/validate_pilot.py
python scripts/collect_pilot.py --offline
```

`--limit` is a debugging option that replaces processed audit outputs with a
subset. Run the full collector afterward to restore the 100-game outputs.

## Most valuable next work

1. Read `reports/pilot_findings.md` and inspect `reports/game_audit.csv`.
2. Expand dated narrative archives beyond Cleveland; audit publication timing.
3. Test midseason/late-season coverage before scaling to thousands of games.
4. Ask Stanford Libraries about LexisNexis API access and training permissions if
   free narrative sources remain insufficient. The published access process has a
   waiting list: https://guides.library.stanford.edu/gearup/lexisnexis.
5. Start prospective capture of upcoming games when ready; nothing is scheduled.
6. Choose the model and evaluation period before freezing splits.

Optional paid sources researched, not purchased: BALLDONTLIE $9.99/month for player
game stats; The Odds API at https://the-odds-api.com/ $30/20k credits or $59/100k
credits, with 10 credits per historical region/market query. Recheck prices before
buying. SportsDataIO's historical injuries are game-start snapshots, not a complete
intraday status history. NBA stats endpoint timed out in the initial probe, so the
pilot uses ESPN's daily scoreboard API for results.

## Superset / conversation continuity

The old scratch-session path is retained as a compatibility symlink to this repo
after relocation so this running conversation can still resolve its files. The
user will import/open the canonical folder in Projects. A new agent can continue
by reading this file; it does not need the full chat transcript. No agent has been
spawned or project registered automatically.

When opening in Superset, use the repository's **local checkout** to retain access
to the downloaded data. A fresh Git worktree contains committed code, but not
git-ignored `data/` or `.venv`.
