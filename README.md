# Sports Forecasting with Language Models

Stanford CS230 project by Tommaso Sini and Anthony Viverito. Proposal: [`docs/proposal.pdf`](docs/proposal.pdf).

**Premise.** Statistics already predict NBA games well. Gradient boosting on prior-game team
stats closes about 70% of the Brier gap between a naive home-win rate and the betting market,
and adds no information the market price does not already contain. The open question is the
remaining 30%: does pregame text (team game notes, injury reports, press conferences, analyst
previews) carry information that the statistics miss?

**Question.** Can one open-weight language model, fine-tuned on both the statistical forecast
and the pregame text, beat statistics alone, and by how much? Ultimately: does it beat frontier
models given the same context, and the market?

We adapt the merger-arbitrage forecasting framework of
[Jajal et al. (2026)](https://arxiv.org/abs/2607.09921) to basketball.

## Approach

1. **Statistical forecast.** Gradient boosting on prior-game team stats gives $p_{\text{stats}}$,
   our strongest non-text model and the starting point for the language model.
2. **Pregame evidence.** For each game, keep only documents timestamped before a fixed cutoff
   (scheduled tipoff minus 60 minutes), grouped into modules: injuries, matchup, press
   conferences, analyst consensus.
3. **Forecaster.** An open-weight LLM with LoRA adapters reads the statistics and the text. A
   probability head on its final hidden state predicts a correction to the statistical log-odds.
   The correction starts at zero, so the model begins at the statistical forecast and moves only
   when the text justifies it. The model also writes a supporting explanation.

   $$\hat p = \sigma\big(\mathrm{logit}\,p_{\text{stats}} + \Delta_\theta(\text{stats}, \text{text})\big)$$

4. **Training.** Hindsight-guided explanations from a teacher model that sees game recaps,
   paired with probability targets that blend the market price and the realized outcome.
   Beyond Jajal et al., we train the head and the explanation jointly:

   $$\mathcal{L}(\theta) = -\sum_t \log p_\theta(z_t \mid x, z_{<t}) + \lambda\,(\hat p_\theta(x) - p^*)^2$$

5. **Measuring what text adds.** Train the same model with and without text and score both on
   identical held-out games played after every model's knowledge cutoff. The value of text is
   the Brier improvement over statistics alone, tested with a paired bootstrap over games. We
   then compare against frontier models and the market, and check whether the stats-plus-text
   forecast carries information the market price lacks.

## Results so far

**1. How far do statistics go?** 2025–26 regular season, 1,132 games where every forecast is
available (lower Brier is better):

| Forecast | Brier | Accuracy |
| --- | ---: | ---: |
| Polymarket price at the cutoff | **0.194** | **69.3%** |
| Gradient boosting on prior-game team stats | 0.210 | 67.3% |
| Historical home-win rate | 0.248 | 54.8% |

Statistics take Brier from 0.248 to 0.210; the market reaches 0.194. The market still beats
boosting (paired Brier difference [−0.022, −0.009]), and a market-plus-boosting combination
gives boosting no weight. That 0.016 gap is the room text has to fill.
Reports: [market](reports/polymarket/findings.md), [boosting](reports/gradient_boosting/findings.md).

**2. Does text help? Not measurably yet.** Three pilots, all on tiny test sets that were
inspected in earlier experiments, so none is conclusive (Brier, lower is better):

| Comparison | Test games | Without text | With text |
| --- | ---: | ---: | ---: |
| Stats features vs stats plus injury text; Qwen2.5-0.5B + LoRA ([report](reports/text_ablation/findings.md)) | 18 | 0.237 | 0.239 |
| Boosting vs boosting corrected from first-page game notes; Qwen2.5-0.5B + LoRA ([report](reports/boosting_notes/findings.md)) | 14 | 0.186 | 0.263 |
| Boosting vs boosting corrected from full-packet research; frozen Qwen2.5-14B + linear head ([report](reports/researched_forecaster/findings.md)) | 14 | 0.186 | 0.193 |

The latest pilot reads every page of 107 archived game-note packets (3,783 pages). Because
the 14B model's own summaries contained factual errors, the forecaster receives verbatim
source passages and code-checked table facts, not generated paraphrases
([quality audit](reports/game_note_research/quality_audit.md)). A frozen 14B encoder turns
that evidence into one vector per game, and a 5,121-parameter linear head corrects the
boosting log-odds. The dev set chose a small correction (L2 = 0.001). On the same 14 test
games, both systems score 78.6% accuracy. The Brier difference is +0.007, with a paired 95%
interval of [−0.012, +0.029], so there is no measurable benefit. Eleven of the 14 test packets are
Cleveland's. Answering the question needs far more verified text and a fresh test set (see next steps).

## Data

| Source | Use | Coverage |
| --- | --- | --- |
| ESPN scoreboard | Results and prior-game team stats | 3,621 regular-season games, 2023–24 to 2025–26 (through April 10) |
| Official NBA injury reports | Injury text, report time checked against the cutoff | 160 audited games, all matched |
| Team game-note PDFs | Narrative and player context from archived pregame versions | 404 games verified, 256 approved ([audit](reports/game_notes_collection/findings.md)); 107 used in full (3,783 pages) |
| Polymarket | Market probability at the cutoff | 2,410 games, 2024–25 onward |

Ground rules, enforced by the validators and leakage tests:

- Every input is timestamped at or before the game's cutoff. Document versions modified after the cutoff are rejected.
- Outcomes are stored apart from inputs. Recaps may only be used to build training targets.
- Raw responses are saved with URLs, retrieval times and checksums, so every dataset rebuilds offline.
- `data/` is git-ignored: third-party content stays local.

## Run

Requires Python 3.12+.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-baseline.txt   # add requirements-text.txt for LLM work
.venv/bin/python -m unittest discover -s tests

.venv/bin/python scripts/run_gradient_boosting.py        # statistics baseline
.venv/bin/python scripts/collect_polymarket.py           # market prices (cached; --offline to rebuild)
.venv/bin/python scripts/evaluate_polymarket.py          # market vs boosting
```

The full-packet research pilot (requires `requirements-research.txt`; research runs on a Modal H100):

To inspect every game's exact Inputs, Research audit and additional Tables, run
`.venv/bin/python scripts/browse_evidence.py` and open
[the local evidence viewer](http://127.0.0.1:8765/). It supports team/date/split
search, rendered reports and original-file downloads without changing experiment artifacts.

```bash
.venv/bin/python scripts/run_research_experiment.py --stage all     # research, encode, fit head, evaluate
.venv/bin/modal deploy --env main scripts/modal_research_vllm.py    # only when cloud code changes
.venv/bin/python scripts/modal_research_vllm.py --check-deployment  # submit jobs this way, not `modal run`
```

Validators (`scripts/validate_*.py`) re-check datasets and results from cached sources. Data collection
(`collect_pilot.py`, `collect_game_notes.py`) and the language-model experiments
(`run_text_ablation.py`, `run_boosting_notes.py`, `run_researched_forecaster.py`) are documented in
their reports, [docs/experiments.md](docs/experiments.md),
[compute options](reports/game_note_research/compute_options.md), and [`HANDOFF.md`](HANDOFF.md).

## Layout

```
config/            experiment and collection settings
scripts/           collectors, models, evaluators and validators
scripts/archive/   superseded drivers (do not run for new work)
tests/             leakage and correctness tests
reports/           findings, metrics and audits (committed)
data/              raw sources, processed datasets, models (local, git-ignored)
docs/              proposal, experiment map, archived handoff log
HANDOFF.md         short current status for the next session
```

See [`docs/experiments.md`](docs/experiments.md) for which scripts are active vs historical.

## Next steps

1. **More verified text.** Extend game-note and injury coverage beyond the current pilot teams
   to hundreds of games, and capture 2026–27 games before tipoff for a test set no model could
   have seen in training.
2. **The stats-plus-text model.** Probability head on the statistical log-odds, joint
   explanation loss, hindsight teacher and market-smoothed targets, compared with standard
   fine-tuning and with the same model trained without text.
3. **The real test.** On 2026–27 games, measure how much text improves on statistics, compare
   with the market and frontier models, and check whether the model carries information the
   market price lacks.
