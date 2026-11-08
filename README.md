# Sports Forecasting with Language Models

Stanford CS230 project by Tommaso Sini and Anthony Viverito.
Proposal: [`docs/proposal.pdf`](docs/proposal.pdf).
Repo: [github.com/TommySini/cs230-nba-forecasting](https://github.com/TommySini/cs230-nba-forecasting).

**Premise.** Statistics already predict NBA games well. Gradient boosting on prior-game team
stats closes about 70% of the Brier gap between a naive home-win rate and the betting market,
and adds no information the market price does not already contain. The open question is the
remaining ~30%: does pregame text (team game notes, injury reports, press conferences, analyst
previews) carry information that the statistics miss?

**Question.** Can one open-weight language model, conditioned on the statistical forecast and
pregame text, beat statistics alone—and by how much? Ultimately: does it beat frontier models
given the same context, and the market?

We adapt the merger-arbitrage forecasting framework of
[Jajal et al. (2026)](https://arxiv.org/abs/2607.09921) (BAM) to basketball.

## Approach

1. **Statistical forecast.** Gradient boosting on prior-game team stats gives $p_{\text{stats}}$,
   our strongest non-text model and the starting point for any text model.
2. **Pregame evidence.** For each game, keep only documents timestamped before a fixed cutoff
   (scheduled tipoff minus 60 minutes). Verified team game notes are the main text source so far.
3. **Forecaster.** An open-weight LLM reads the text (and, in the intended design, produces an
   explanation). A probability head predicts a **correction** to the statistical log-odds.
   The correction starts at zero, so the model begins at the statistical forecast and moves only
   when the text justifies it:

   $$\hat p = \sigma\big(\mathrm{logit}\,p_{\text{stats}} + \Delta_\theta(\text{stats}, \text{text})\big)$$

4. **Training (intended BAM-style).** Hindsight-guided explanations from a teacher that can see
   game recaps, paired with probability targets that can blend the market price and the outcome.
   Joint head + explanation training is the design target; it is **not finished yet**:

   $$\mathcal{L}(\theta) = -\sum_t \log p_\theta(z_t \mid x, z_{<t}) + \lambda\,(\hat p_\theta(x) - p^*)^2$$

5. **Measuring what text adds.** Compare systems on identical held-out games after every model's
   knowledge cutoff. Primary metric: Brier, with paired bootstrap. Also ask whether a
   stats-plus-text forecast carries information the market price lacks.

## Results so far

**1. How far do statistics go?** 2025–26 regular season, 1,132 games with both forecasts
(lower Brier is better):

| Forecast | Brier | Accuracy |
| --- | ---: | ---: |
| Polymarket price at the cutoff | **0.194** | **69.3%** |
| Gradient boosting on prior-game team stats | 0.210 | 67.3% |
| Historical home-win rate | 0.248 | 54.8% |

Boosting closes most of the gap from the naive baseline to the market (~0.038 of ~0.054 Brier).
The market still beats boosting by ~0.016 (paired day-bootstrap 95% CI **[−0.022, −0.009]**).
A logistic stack of market + boosting puts essentially **zero weight** on boosting, so this
booster adds no information beyond the price. That leftover gap is the room text has to fill.
Reports: [market](reports/polymarket/findings.md), [boosting](reports/gradient_boosting/findings.md).

**2. Does text help? Not measurably yet.** Three small pilots on previously inspected
test sets (not a fresh final holdout):

| Comparison | Test games | Without text | With text |
| --- | ---: | ---: | ---: |
| Stats vs injury text; Qwen2.5-0.5B + LoRA ([report](reports/text_ablation/findings.md)) | 18 | 0.237 | 0.239 |
| Boosting vs first-page notes; Qwen2.5-0.5B + LoRA ([report](reports/boosting_notes/findings.md)) | 14 | 0.186 | 0.263 |
| Boosting vs full-packet research; frozen Qwen2.5-14B + linear head ([report](reports/researched_forecaster/findings.md)) | 14 | 0.186 | 0.193 |

None shows a clear text gain. The 14B run used verbatim quotes and code-checked
tables after paraphrases failed checks ([quality audit](reports/game_note_research/quality_audit.md));
paired Brier CI vs boosting was **[−0.012, +0.029]**.

**3. BAM-style pipeline (in progress).** Larger note cohort, teacher/quality probes,
and Modal student tooling are in `scripts/bam_*.py` / `reports/bam_pipeline/`.
Gold SFT and joint forecast training are not done yet
([status](reports/bam_pipeline/implementation_status.json),
[API setup](docs/api_teacher_setup.md)).

The current continuation is [AWS-only, TRAIN-first](docs/bam_training_first.md).
A separate TRAIN-only joint adapter/head worker and reviewed-gold data checks
now pass synthetic tests. The 16–32-game NBA learning pilot has not run; credit
eligibility, GPU quota and model activation remain prerequisites
([AWS setup](docs/aws_toolkit_setup.md)). DEV is deferred and TEST excluded.

## Data

| Source | Use | Coverage |
| --- | --- | --- |
| ESPN scoreboard | Results and prior-game team stats | 3,621 regular-season games, 2023–24 to 2025–26 (through April 10) |
| Official NBA injury reports | Injury text; report time vs cutoff | 160 audited games, all matched |
| Team game notes (PDF / HTML) | Pregame narrative from archived pre-cutoff versions | **424 games** in the current BAM snapshot (~13.3k PDF pages, plus HTML); earlier audit stages were 404 source-verified / 256 content-approved ([expansion](reports/bam_notes_expansion/v1/summary.json), [collection](reports/game_notes_collection/findings.md)) |
| Polymarket | Market probability at the cutoff | 2,410 games, 2024–25 onward |

Ground rules (validators + leakage tests):

- Inputs only at or before the game's cutoff. Post-cutoff document versions are rejected.
- Outcomes stay separate from prediction inputs. Recaps may train **targets**, not pregame features.
- Raw responses keep URLs, retrieval times, and checksums for offline rebuilds.
- `data/` is git-ignored (local only).

## Run

Requires Python 3.12+. Use the repo `.venv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-baseline.txt
.venv/bin/python -m unittest discover -s tests

.venv/bin/python scripts/run_gradient_boosting.py   # statistics baseline
.venv/bin/python scripts/collect_polymarket.py      # market prices (--offline to rebuild)
.venv/bin/python scripts/evaluate_polymarket.py     # market vs boosting
```

Full-packet research pilot (`requirements-research.txt`; heavy; Modal H100 for research stage):

```bash
.venv/bin/python scripts/browse_evidence.py         # local viewer → http://127.0.0.1:8765/
.venv/bin/python scripts/run_research_experiment.py --stage all
.venv/bin/modal deploy --env main scripts/modal_research_vllm.py    # only when cloud code changes
.venv/bin/python scripts/modal_research_vllm.py --check-deployment  # do not use ephemeral `modal run`
```

Which script is active vs historical, and which `requirements-*.txt` to install:
[`docs/experiments.md`](docs/experiments.md). Short agent status: [`HANDOFF.md`](HANDOFF.md).
Compute notes: [`reports/game_note_research/compute_options.md`](reports/game_note_research/compute_options.md).

## Layout

```
config/            experiment and collection settings
scripts/           collectors, models, evaluators, validators (bam_*.py = BAM pipeline)
scripts/archive/   superseded drivers (do not run for new work)
tests/             leakage and correctness tests
reports/           findings, metrics, audits (committed)
data/              raw sources, processed datasets, models, credentials (local, git-ignored)
docs/              proposal, experiment map, API teacher setup, archived handoff log
HANDOFF.md         short current status for the next session
```

## Next steps

1. **Finish BAM-style training** on the expanded note cohort: qualified teacher gold, researcher
   SFT, forecast head/joint objective, locked evaluation—without claiming a fresh final test yet.
2. **More verified text** and live capture of 2026–27 games before tipoff for a holdout no
   checkpoint could have seen.
3. **Market and frontier comparisons** on that fresh set: does text close the boosting→market gap,
   and does the hybrid carry information the price lacks?
