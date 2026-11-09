# Sports Forecasting with Language Models

Stanford CS230 project by Tommaso Sini and Anthony Viverito.
Proposal: [`docs/proposal.pdf`](docs/proposal.pdf).
Repo: [github.com/TommySini/cs230-nba-forecasting](https://github.com/TommySini/cs230-nba-forecasting).

**Question.** Statistics already predict NBA games well, and the betting market does better
still. Does pregame text (team game notes) carry information the statistics miss? Can one
open-weight language model, given the statistical forecast and the notes, beat statistics alone,
and how close does it get to the market?

We adapt the forecasting framework of [Jajal et al. (2026)](https://arxiv.org/abs/2607.09921),
"Global merger-arbitrage forecasting with language models", to basketball. The repo calls this
pipeline **BAM** (`scripts/bam_*.py`), after the paper's authors.

## Approach

1. **Statistical forecast.** Gradient boosting on prior-game team and player box-score stats gives
   $p_{\text{stats}}$.
2. **Pregame evidence.** Team game notes whose bytes are shown to be public before the cutoff
   (scheduled tipoff minus 60 minutes): archived captures, publisher timestamps or article dates.
3. **Research.** A frozen Qwen2.5-14B reads every page of the notes and writes a cited brief.
4. **Teacher gold (training only).** A stronger model that can see the outcome and box score
   explains what was foreseeable, then rewrites the brief as if still before the game; a second
   model reviews it against the pregame sources.
5. **Forecaster.** Qwen2.5-14B with LoRA reads the brief and predicts a correction to the
   statistical log-odds, trained jointly on Brier loss and the gold explanations:

   $$\hat p = \sigma\big(\mathrm{logit}\,p_{\text{stats}} + \Delta_\theta(\text{stats}, \text{text})\big),\qquad
   \mathcal{L} = \overline{\mathrm{Brier}} + \lambda\,\overline{\mathrm{CE}}_{\text{gold}}$$

6. **Evaluation.** Identical held-out games after the model's knowledge cutoff; Brier with a paired
   day-clustered bootstrap, against boosting and the market price at note-publication time.

## Results so far

Statistics predict better than a constant home-win rate, and the market predicts
better than statistics. On the **same 1,132 TEST games** in 2025–26:

| Forecast | Brier (lower is better) | Accuracy |
| --- | ---: | ---: |
| Polymarket price 1 hour before tipoff | **0.1941** | **69.3%** |
| Gradient boosting on prior-game statistics | 0.2059 | 67.2% |
| TRAIN home-win rate | 0.2484 | 54.8% |

The market-minus-boosting paired 95% Brier interval is [−0.0180, −0.0055].
On a separate 1,125-game cohort with usable prices at both horizons, the market
still beats boosting eight hours before tipoff: 0.1952 versus 0.2055.
The full statistical baseline scores 0.2058 on 1,137 games. Different cohorts
explain the slightly different baseline scores.

Three text pilots, each with only 14–18 test games, found no demonstrated gain.
The full BAM forecaster has not been trained. The 2025–26 season was previously
inspected, so these are exploratory backtests.

Read the reports for the methods, exact populations and limitations:

- [Data coverage and provenance](reports/data.md)
- [Statistical and market baselines](reports/baselines.md)
- [Text experiments and lessons](reports/text_experiments.md)

## Data

All data is local and git-ignored, under the canonical checkout's `data/`.

| Dataset | Coverage | Built by |
| --- | --- | --- |
| ESPN schedules and results | 16,829 records, including 8 non-franchise exhibitions; 2012–13 to 2025–26 | `collect_pilot.py --stage schedules` (configs `history_2012`, `seasonal`) |
| ESPN box scores (NBA.com where ESPN is empty or invalid) | 16,810 of 16,821 games, 2012–13 to 2025–26 | `collect_rich_stats.py` |
| Game notes v2 | 2,144 team documents for 1,997 games, 2012–26 | `collect_notes_v2.py` ([findings](reports/data.md)) |
| Polymarket prices and trades | 2,410 games, 2024–25 onward | `collect_polymarket.py`, `collect_polymarket_trades.py` |
| BAM snapshot `bam_notes_v2` | 1,922 games with notes, 2012–26 (TRAIN 1,592, DEV 193, TEST 137): numbered pages, split, boosting probability for 1,768 | `bam_data.py prepare --config config/bam_snapshot.json` |

**Fixed splits** (`config/gradient_boosting.json`): TRAIN = 2012–13 to 2023–24, DEV = 2024–25,
TEST = 2025–26. Each dataset is built once, validated, then left alone.

Ground rules, enforced by validators and tests:

- Inputs only at or before the game's cutoff; later document versions are rejected.
- Outcomes stay separate from inputs. Recaps may train targets, never pregame features.
- Raw responses keep URLs, retrieval times and checksums, so every dataset rebuilds offline.

## Layout

```
scripts/
  collect_pilot.py            shared download cache and time helpers; ESPN schedule collector
  collect_notes_v2.py         game notes; validate_notes_v2.py, postgame_screen.py
  collect_rich_stats.py       box scores; box_score_features.py turns them into prior-game features
  run_gradient_boosting.py    the baseline: fit on TRAIN, select on DEV, score TEST; validate_gradient_boosting.py
  collect_polymarket*.py      market prices and trades; evaluate_polymarket*.py
  bam_data.py                 notes_v2 + boosting -> immutable BAM snapshot
  bam_research.py             research prompts and claim checks (bam_claim_guards, bam_note_tables, game_note_tables)
  bam_postgame.py             outcome-side evidence for the teacher (training only)
  bam_api_teacher.py          teacher/reviewer request payloads
  bam_training.py             LoRA + probability-head model code (bam_checkpoint for resume)
  bam_train_contract.py       TRAIN bundle checks; bam_train_only.py worker; bam_aws_preflight.py
  bam_inference.py            vLLM batch worker; bam_budget.py spend ledger
config/                       one config per dataset or experiment
tests/                        unit and leakage tests
reports/                      three research reports and a reading guide
results/                      six retained metrics files; exact results and lineage
docs/                         proposal, AWS plan (bam_aws_architecture, bam_training_first,
                              aws_toolkit_setup), BAM design notes in docs/bam/
infra/aws/                    IAM CloudFormation template
```

## Verify the existing datasets and results

Requires Python 3.12+ and the repo `.venv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-baseline.txt
.venv/bin/python -m unittest discover -s tests

.venv/bin/python scripts/validate_notes_v2.py
.venv/bin/python scripts/validate_gradient_boosting.py --source-revision a75d4cc00b6d89354c9c9b8af4bbad153fae9e4c
.venv/bin/python scripts/validate_polymarket.py
.venv/bin/python scripts/bam_data.py validate data/processed/bam_pipeline/bam_notes_v2
```

The validation commands use the existing local data and do not fit models or
rebuild datasets. The pinned source revision verifies the original boosting
code; current data hashes and prediction replay are checked independently.
Collectors, model runners and evaluators write outputs and require a separately
authorized experiment.

GPU workers install `requirements-bam-training.txt` (training) or `requirements-research-vllm.txt`
(research generation) on the AWS instance, not on the Mac.

Current status and next steps: [`HANDOFF.md`](HANDOFF.md).
