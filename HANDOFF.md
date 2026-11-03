# CS230 NBA forecasting — handoff

Short current state for the next session. Chronological agent history through
2026-11-01 is archived in [`docs/archive/handoff-through-2026-11-01.md`](docs/archive/handoff-through-2026-11-01.md).
Experiment map: [`docs/experiments.md`](docs/experiments.md). Public summary: [`README.md`](README.md).

## Status (2026-11-01)

**Main full-note experiment is complete.** No research/training/GPU job should be
running. Pipeline `20261101T201235Z` finished research, source validation, MLX
embeddings, head training, evaluation, and review index.

| Item | Location |
| --- | --- |
| Findings / metrics | `reports/researched_forecaster/` |
| Research quality | `reports/game_note_research/quality_audit.md` |
| Evidence browser | `.venv/bin/python scripts/browse_evidence.py` → http://127.0.0.1:8765/ |
| Modal app (idle, reusable) | `cs230-nba-research-vllm` in workspace `tsini` |

**Headline result.** Same 14 held-out games: original boosting Brier **0.186** vs
full-note hybrid **0.193** (hybrid worse by ~0.007; paired 95% CI includes zero).
Not evidence that text helps. Limits: tiny / previously inspected test set, 11/14
Cleveland packets. Frozen Qwen2.5-14B; only the 5,121-parameter head was trained.

**Earlier pilots (keep as history).** Stats vs injury text (`text_ablation`) and
boosting vs first-page notes (`boosting_notes`) also failed to show a text gain.
Details in `docs/experiments.md` and their `reports/*/findings.md`.

## What to work on next

1. More verified pregame text beyond the current team/date sample.
2. Fresh final holdout (e.g. 2026–27 games) — do not treat the 14-game pilot as
   the definitive test.
3. Joint probability + explanation training and market-smoothed targets remain
   unimplemented (see README approach).

## Workspace

- **This worktree (preferred for new work):**  
  `/Users/tommasosini/.superset/worktrees/cs230-nba-forecasting/read-then-continue-dataset-09962af6`
- **Canonical clone:** `/Users/tommasosini/Desktop/Repos/cs230-nba-forecasting`
- **GitHub:** https://github.com/TommySini/cs230-nba-forecasting  
  Worktree may be ahead of `origin/main`; push when you want remote to match.
- `data/` is local and git-ignored. Use `.venv/bin/python`.

## Commands agents should default to

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/run_gradient_boosting.py          # stats baseline
.venv/bin/python scripts/evaluate_polymarket.py            # market vs boosting
.venv/bin/python scripts/run_research_experiment.py --stage all   # full pipeline (heavy)
.venv/bin/modal deploy --env main scripts/modal_research_vllm.py  # only if cloud code changed
.venv/bin/python scripts/modal_research_vllm.py --check-deployment
```

Do **not** start `scripts/archive/modal_research_batch.py` or `modal run` ephemeral
apps for the main path. Active cloud path is deploy-once + Python CLI on
`modal_research_vllm.py`. Shared Modal helpers still live in `scripts/modal_research.py`
(legacy MLX benchmark app + volumes/`unpack`); leave that file alone unless
refactoring deliberately.

## Rules that never change

- Pregame inputs only; enforce cutoff timestamps. No raw ESPN → model.
- Outcomes separate from inputs. Unverified narratives stay quarantined.
- Do not claim league-wide note coverage from the current selective sample.
- After collector changes: leakage tests + pilot validator (`collect_pilot.py --offline`).
