# Text experiments and lessons

Three small pilots found **no demonstrated improvement from adding pregame
text**. That is a useful result about these implementations, not evidence that
text has no predictive value. Each test contained only 14–18 games, the historical
outcomes had already been inspected, and the methods differed substantially.
The full BAM forecaster with joint probability and explanation training has not
yet run.

## What was compared

| Pilot | TRAIN / DEV / TEST games | Control | Text treatment |
| --- | --- | --- | --- |
| Injury excerpts | 108 / 14 / 18 | Qwen2.5-0.5B + LoRA receiving prior-game statistics | Same initialization, training budget and model, with verified injury text added |
| First-page notes | 80 / 13 / 14 | Earlier statistical booster | Qwen2.5-0.5B + LoRA learns a correction from the booster probability and a 2,700-token front-page selection |
| Full-packet research | 80 / 13 / 14 | Same earlier booster on the same games | Frozen Qwen2.5-14B encodes selected source quotations and code-derived table facts; a small linear head learns the correction |

The injury pilot trained through November 5, 2025, used November 7–January 16
for DEV, and tested January 17–April 10, 2026. The notes pilots trained on games
from November 4, 2024 through December 1, 2025, used December 4–January 31 for
DEV, and tested February 1–April 10, 2026. These are the actual old experimental
partitions, not the project's current fixed whole-season split. They must not
be reused as the current TRAIN/DEV/TEST definition.

## Results on each pilot's identical test games

| Pilot | Games | Control Brier | With-text Brier | Text minus control | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Injury excerpts | 18 | 0.2370 | 0.2390 | +0.0020 | [-0.0353, +0.0376] |
| First-page notes | 14 | 0.1863 | 0.2632 | +0.0769 | [-0.0362, +0.1866] |
| Full-packet research | 14 | 0.1863 | 0.1932 | +0.0069 | [-0.0118, +0.0292] |

Lower Brier is better; a negative difference would favor text. All three point
estimates are worse with text, but all intervals include zero. The intervals
resample game dates and condition on the fitted models; they exclude training
and selection uncertainty. Exact results and selection/prediction hashes are in
the [injury](../results/injury_text_ablation.json),
[first-page](../results/boosting_first_page_notes.json) and
[full-packet](../results/researched_forecaster_14b.json) metrics files.

The injury experiment is the most direct text ablation: the two arms share
initial trainable weights, game ordering and optimization settings. A common
epoch was selected using mean DEV Brier across both arms. Epoch 1 won. Its text
arm had higher accuracy, 66.7% versus 55.6%, despite slightly worse Brier; on
18 games, that does not establish better probability forecasts. The separate
boosting reference used much more training data and is not the matched ablation
control.

The first-page pilot selected epoch 2 using DEV Brier. The zero-correction
starting point was eligible, and there was no DEV refit. Its restricted context
means that failure cannot be attributed to the complete packet. The full-packet
pilot instead kept the 14B weights frozen and fitted a 5,121-parameter correction
head; DEV selected regularization `l2_0.001`. Neither trained the current BAM
joint probability/explanation objective.

The two notes pilots include extra numerical facts and potential recalibration
as well as prose. Their comparisons therefore measure the whole text-conditioned
system, not the isolated value of narrative language. Eleven of the 14 full-packet
test sources came from Cleveland, further limiting generalization.

## Source interpretation was a substantive failure mode

The research-quality trials revealed errors that an apparently grounded summary
could hide: old game-log rows described as the latest game, past DNP entries
treated as current absences, conditional scoring splits presented as current
facts, and field goals made read as points. A larger 32B trial still made table
errors. More model capacity alone did not resolve the issue.

On twelve targeted semantic challenges for a TRAIN packet, the selected 14B
checker passed ten. It accepted a three-point streak assigned to the wrong player
and a claim that confused five made field goals with five points. Thus a second
pass by the same model could not certify its generated summaries. These were
targeted challenges on a small reference case, not an estimate of corpus-wide
error rates.

The revised historical pilot fed the forecaster exact cited source passages with
nearby context and code-derived table facts instead of generated paraphrases.
Known table formats were decoded with date, column and scoring/rebound checks;
unrecognized formats remained unresolved. Citation copying establishes what the
source said, but not that a model understood it or selected all relevant evidence.
Likewise, arithmetic checks cannot establish player identity or temporal scope.

The full historical research run processed 107 packets and all 3,783 physical
pages. That establishes source-processing coverage, not that every fact reached
the forecast input or that the checker was correct. The resulting full-packet
forecast still showed no measured gain. The detailed audit is preserved at Git
revision `67596558a188014803724bd60a158ce6f205ec3e`, path
`reports/past/research_quality_audit/findings.md`. Current safeguards and their
remaining semantic limits belong in [the gold design](../docs/bam/gold_design.md),
not in a second implementation log here.

## The memorization probe does not clear contamination

A separate Qwen2.5-14B diagnostic stopped after the earliest 400 games of 2023–24
and 92 games of 2024–25. It asked for winners and final scores. Its limited recall
under those prompts cannot prove the absence of game-level information in the
weights, especially with different cues or prompts. Its hindsight-strength
comparator was diagnostic only and must not enter forecasting features.
The exact prompts and preliminary outputs remain at the same Git revision,
path `reports/memorization_probe/qwen2.5-14b.json`.

Model identity, documented temporal eligibility and prior test exposure must
therefore remain explicit. A small recall probe does not make an inspected
historical season untouched.

## What the next experiment must establish

The immediate unanswered question is whether the intended BAM implementation can
learn on a small, audited TRAIN cohort at all. The existing
[TRAIN-first plan](../docs/bam_training_first.md) starts with 16–32 TRAIN games,
reviewed research and gold, a frozen statistical forecast, and bounded spending.
It defers DEV and excludes TEST. No new split or result is implied here.

TRAIN-only postgame information may guide target explanations, while the actual
forecast input remains the ordinary pregame brief. Reviewed gold is supervision,
not an extra source at prediction time. Learning checks must keep the uncorrected
booster as a control; later evaluation must compare models on identical games.

Historical pilot code was retired. The retained metrics preserve exact results,
but their former local data locations are absent in this checkout; full replay
requires recovering the original artifacts and code. Read a historical record
without recreating old folders using, for example:

```bash
git show 67596558a188014803724bd60a158ce6f205ec3e:reports/past/injury_text_ablation/findings.md
```

The historical reports record each method's protocol; the `pre-cleanup` tag
retains retired experiment code. Today's scripts are not interchangeable with
those implementations.
