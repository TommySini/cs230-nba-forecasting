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

Data files stay out of Git. A fresh clone downloads the pinned dataset from a
[versioned GitHub Release](https://github.com/TommySini/cs230-nba-forecasting/releases/tag/nba-data-4e5ba4eb0554)
into its local `data/`. The checked-in `config/dataset_release.json` pins every
archive's URL, size and SHA-256 checksum. Maintainer worktrees continue sharing
the canonical checkout's data directory.

The `data-reconstruction` branch reads one shared Parquet dataset. Start with the
[data guide](docs/data_reconstruction.md) for the table map, feature dictionary and
a real game traced back to its sources. Features, outcomes and predictions have
separate tables; original downloads remain in raw, saved models in `data/models/`,
and run records in `reports/`. Fresh downloads use this layout directly. In the
maintainer workspace, legacy folders remain temporarily because other worktrees
still read them; local raw regrouping and deletion are deferred.

| Dataset | Coverage | Built by |
| --- | --- | --- |
| ESPN schedules and results | 16,829 source rows, including 16,821 NBA franchise games, 2012–13 to 2025–26 | `collect_pilot.py --stage schedules` (configs `history_2012`, `seasonal`) |
| ESPN box scores (NBA.com where ESPN is empty or invalid) | 16,810 of 16,821 games, 2012–13 to 2025–26 | `collect_rich_stats.py` |
| Game notes v2 | 2,144 team documents for 1,997 games, 2012–26 | `collect_notes_v2.py` ([findings](reports/data.md)) |
| Polymarket prices and trades | 2,410 games, 2024–25 onward | `collect_polymarket.py`, `collect_polymarket_trades.py` |
| BAM notes cohort | 1,922 games with notes, 2012–26 (TRAIN 1,592, DEV 193, TEST 137): numbered pages, split, boosting probability for 1,768 | `bam_data.py validate data/processed` |

**Fixed splits** (`config/gradient_boosting.json`): TRAIN = 2012–13 to 2023–24, DEV = 2024–25,
TEST = 2025–26. Each dataset is built once, validated, then left alone.

Ground rules, enforced by validators and tests:

- Inputs only at or before the game's cutoff; later document versions are rejected.
- Outcomes stay separate from inputs. Recaps may train targets, never pregame features.
- Raw responses keep URLs, retrieval times and checksums, so every dataset rebuilds offline.

## Layout

```
scripts/
  download_data.py           verified dataset download for new clones; optional raw source bundle
  package_data.py            build the versioned release archives from validated data
  nba_dataset.py             shared Parquet readers, feature dictionary, migration and integrity checks
  collect_pilot.py            shared download cache and time helpers; ESPN schedule collector
  collect_notes_v2.py         game notes; validate_notes_v2.py, postgame_screen.py
  collect_rich_stats.py       box scores; box_score_features.py turns them into prior-game features
  run_gradient_boosting.py    the baseline: fit on TRAIN, select on DEV, score TEST; validate_gradient_boosting.py
  run_tabpfn.py               local TabPFN TRAIN diagnostic and DEV comparison; validate_tabpfn.py
  collect_polymarket*.py      market prices and trades; evaluate_polymarket*.py
  evaluate_*orthogonality.py  does market, note or injury text add to boosting? (collect_injuries.py)
  bam_data.py                 shared notes/pages + frozen boosting -> verified BAM inputs
  bam_research.py             research prompts and claim checks (bam_claim_guards, bam_note_tables, game_note_tables)
  bam_postgame.py             outcome-side evidence for the teacher (training only)
  bam_api_teacher.py          teacher/reviewer request payloads
  bam_training.py             LoRA + probability-head model code (bam_checkpoint for resume)
  bam_train_contract.py       TRAIN bundle checks; bam_train_only.py worker; bam_aws_preflight.py
  bam_concepts.py             concept-bottleneck head, teacher concept labels, concept selection
  bam_inference.py            vLLM batch worker; bam_budget.py spend ledger
config/                       one config per dataset or experiment
tests/                        unit and leakage tests
reports/                      three research reports, a reading guide, and the run records the
                              dataset release pins (gradient_boosting/, polymarket/, tabpfn/)
results/                      archived metrics of finished text pilots
docs/                         proposal, AWS plan (bam_aws_architecture, bam_training_first,
                              aws_toolkit_setup), BAM design notes in docs/bam/
infra/aws/                    IAM CloudFormation template
```

## Verify the existing datasets and results

Requires Python 3.12+, macOS/Linux (or WSL), and the repo `.venv`. Run commands from the
repository root. All dependency pins live in [`pyproject.toml`](pyproject.toml).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --group local
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/nba_dataset.py validate
.venv/bin/python scripts/validate_tabpfn.py --stage train-smoke
.venv/bin/python scripts/validate_tabpfn.py --stage dev
.venv/bin/python -m unittest discover -s tests

The default download contains the shared Parquets, audit metadata and saved boosting
model. It supports data inspection and validation of the saved TabPFN comparison
without downloading neural-network weights. Matching files are reused; conflicting
files are never overwritten. See the [data guide](docs/data_reconstruction.md) for
bundle sizes and cache locations.

Full source-level validation also needs the optional raw bundle:

```bash
.venv/bin/python scripts/download_data.py --include-raw
.venv/bin/python scripts/validate_gradient_boosting.py
.venv/bin/python scripts/validate_notes_v2.py
.venv/bin/python scripts/validate_polymarket.py
.venv/bin/python scripts/bam_data.py validate data/processed
```

Orthogonality checks (processed bundle only): does a second signal add to boosting?

```bash
.venv/bin/python scripts/evaluate_orthogonality.py          # market vs boosting, DEV+TEST
.venv/bin/python scripts/evaluate_notes_orthogonality.py    # note TF-IDF vs boosting, TRAIN+DEV
```

Select the dependency group for the environment:

| Group | Purpose | Install into that environment's `.venv` |
| --- | --- | --- |
| `local` | Collection, data validation, analysis, and statistical baseline | `uv pip install --python .venv/bin/python --group local` |
| `training` | GPU training worker | `uv pip install --python .venv/bin/python --group training` |
| `research` | GPU research generation with vLLM | `uv pip install --python .venv/bin/python --group research` |
| `tabpfn` | Optional TabPFN baseline and adaptation (add to `local`) | `uv pip install --python .venv/bin/python --group local --group tabpfn` |
| `validation` | Optional CloudFormation linting | `uv pip install --python .venv/bin/python --group validation` |

GPU workers use separate environments on Linux/CUDA instances: Python 3.12 for `training`
and Python 3.13 for `research`. The research pins require Python 3.13+ because vLLM's
`mistral-common` dependency requires NumPy below 2.4 on older Python, conflicting with
the pinned NumPy 2.4.6. Do not install GPU groups into the Mac `.venv`.
The `local`, `training`, and `research` groups pin
incompatible package versions. Install only one of those groups per environment;
`validation` can be added when needed. The install commands resolve only the selected group.

The user authorized local compute specifically for TabPFN. In its worktree, install
the `local` and `tabpfn` groups into `.venv`. The checked-in config pins the package, checkpoint
revision and SHA256, four estimators, and the Apple Silicon `mps` device.
The baseline is complete; inspect it with:

```bash
uv pip install --python .venv/bin/python --group local --group tabpfn
.venv/bin/python scripts/validate_tabpfn.py --stage train-smoke
.venv/bin/python scripts/validate_tabpfn.py --stage dev
```

The complete unit suite also exercises Torch-based BAM code. With the optional
TabPFN dependencies installed, run `.venv/bin/python -m unittest discover -s tests`.

Preparation records cohort IDs and dataset hashes without copying features or labels.
Completed runs are protected from overwrite; repeat a model stage only to resume
an interrupted run. DEV is exploratory because the
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
