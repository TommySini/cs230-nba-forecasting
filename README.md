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

**Statistics and the market.** 2025–26 regular season (lower Brier is better):

| Forecast | Brier | Accuracy |
| --- | ---: | ---: |
| Polymarket price 1 hour before tipoff | **0.194** | **69.3%** |
| Polymarket price 8 hours before tipoff (typical note publication) | 0.195 | 69.2% |
| Gradient boosting on prior-game team + player box scores (fit on 2012–24) | 0.206 | 67.5% |
| Historical home-win rate | 0.248 | 54.8% |

Boosting is fit on the 12 TRAIN seasons and its settings chosen on DEV 2024–25 (DEV Brier 0.210).
The market beats it by about 0.010 even 8 hours out (paired 95% CI [−0.016, −0.004] on TEST,
[−0.017, −0.005] on DEV); the extra 7 hours of information are worth only about 0.0015. That gap
is the room text has to fill.
Reports: [market](reports/polymarket/findings.md), [market at 8 h](reports/polymarket/horizon_findings.md),
[boosting](reports/gradient_boosting/findings.md), [earlier boosting runs](reports/past/README.md).

**Text: no measurable gain yet.** Three small pilots, on 14–18 test games each, found no gain
from text ([summaries](reports/past/README.md)). The full BAM pipeline has not been trained yet.

**TabPFN experiments.** The `tabpfn-implementation` branch compares the same frozen
252 features on 13,457 TRAIN and 1,123 DEV games. Original TabPFN-2.5 (fixed weights,
four estimators) achieved DEV Brier **0.20783**, versus boosting's **0.21033**;
the paired 95% difference interval **[-0.00512, +0.00015]** includes zero.

Three local follow-ups found no Brier improvement: synthetic augmentation scored
**0.20802**; fine-tuning retained the original weights after internal TRAIN validation
worsened, matching **0.20783**; the combined experiment similarly retained the original
weights and matched synthetic-only at **0.20802**. Both fine-tuning attempts made real
gradient updates. The official synthetic tool generated 1,024 rows using TRAIN-only
categorical bins; its continuous regressor checkpoint was unavailable. Checkpoint
selection used the final TRAIN season, with DEV and TEST excluded from fitting.
These are bounded exploratory results, not a rejection of every possible configuration.
Details and limitations: [TabPFN](reports/tabpfn/findings.md).

## Data

All data is local and git-ignored, under the canonical checkout's `data/`.

| Dataset | Coverage | Built by |
| --- | --- | --- |
| ESPN schedules and results | 16,829 regular-season games, 2012–13 to 2025–26 | `collect_pilot.py --stage schedules` (configs `history_2012`, `seasonal`) |
| ESPN box scores (NBA.com where ESPN is empty or invalid) | 16,810 of 16,821 games, 2012–13 to 2025–26 | `collect_rich_stats.py` |
| Game notes v2 | 2,144 team documents for 1,997 games, 2012–26 | `collect_notes_v2.py` ([findings](reports/notes_v2/findings.md)) |
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
  run_tabpfn.py               local TabPFN TRAIN diagnostic and DEV comparison; validate_tabpfn.py
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
reports/                      current findings; reports/past/ for finished pilots
docs/                         proposal, AWS plan (bam_aws_architecture, bam_training_first,
                              aws_toolkit_setup), BAM design notes in docs/bam/
infra/aws/                    IAM CloudFormation template
```

## Run

Requires Python 3.12+ and the repo `.venv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-baseline.txt
.venv/bin/python -m unittest discover -s tests

.venv/bin/python scripts/validate_notes_v2.py
.venv/bin/python scripts/validate_gradient_boosting.py
.venv/bin/python scripts/evaluate_polymarket_horizons.py
.venv/bin/python scripts/bam_data.py prepare --config config/bam_snapshot.json
```

GPU workers install `requirements-bam-training.txt` (training) or `requirements-research-vllm.txt`
(research generation) on the AWS instance, not on the Mac.

The user authorized local compute specifically for TabPFN. In its worktree, install
`requirements-tabpfn.txt` into `.venv`. The checked-in config pins the package, checkpoint
revision and SHA256, four estimators, and the Apple Silicon `mps` device. Run:

```bash
uv pip install --python .venv/bin/python -r requirements-tabpfn.txt
.venv/bin/python scripts/run_tabpfn.py --stage download-model
.venv/bin/python scripts/run_tabpfn.py --stage prepare
.venv/bin/python scripts/run_tabpfn.py --stage train-smoke
.venv/bin/python scripts/validate_tabpfn.py --stage train-smoke
.venv/bin/python scripts/run_tabpfn.py --stage prepare --cohort dev
.venv/bin/python scripts/run_tabpfn.py --stage dev
.venv/bin/python scripts/validate_tabpfn.py --stage dev
```

Preparation refuses to overwrite an existing bundle. Repeat a model stage only to resume
an interrupted run; validate completed results instead. DEV is exploratory because the
boosting settings were already selected using DEV. These forecasts are not wired into BAM
unless a later model-selection decision authorizes that change. TabPFN-2.5 weights permit
[academic research](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b/LICENSE.txt).

The adaptation study is configured in `config/tabpfn_adaptation.json` and uses the
same runner (`adapt-prepare`, `adapt-generate`, `adapt-fit`, `adapt-dev`, `adapt-report`
stages; variants `synthetic`, `finetune`, `combined`). Existing outputs are immutable.
Validate the completed study without running models:

```bash
.venv/bin/python scripts/validate_tabpfn.py --variant synthetic
.venv/bin/python scripts/validate_tabpfn.py --variant finetune
.venv/bin/python scripts/validate_tabpfn.py --variant combined
```

Current status and next steps: [`HANDOFF.md`](HANDOFF.md).
