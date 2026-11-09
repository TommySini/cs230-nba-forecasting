# Statistical forecasts and the market benchmark

Prior-game statistics provide a useful NBA forecast, but Polymarket prices are
better on the same games. The current booster scores **0.2058 Brier on 1,137
2025–26 games**. On the 1,125-game cohort with usable prices at both market
horizons, the eight-hour price scores **0.1952**, versus **0.2055** for boosting.
The paired interval favors the market. This is the gap that additional pregame
evidence should be judged against.

Brier score is the mean squared difference between a predicted probability and
the binary outcome; lower is better. Predicting 50% for every game scores 0.25.
Accuracy uses a fixed 0.5 threshold and does not measure probability quality as
directly. All results below are saved results, not newly fitted experiments.

## Statistical model and selection

The [fixed configuration](../config/gradient_boosting.json) assigns TRAIN to
2012–13 through 2023–24, DEV to 2024–25 and TEST to 2025–26. Four predetermined
gradient-boosting candidates are fitted on TRAIN and ranked by DEV Brier. The
selected model has depth 2 and 200 trees. Both DEV and TEST predictions come from
that TRAIN fit, with no DEV refit, probability calibration or TEST-driven tuning.

The 252 features combine basic team history, recent performance and rest with
team/player box-score windows, coverage indicators and home-away differences.
They use only earlier dates in the same season. Median imputation is fitted on
each model's own training rows. There are no current injuries, lineups, text or
market prices among these features. Prior workload is not current availability.

| Candidate depth / trees | DEV Brier |
| --- | ---: |
| 1 / 100 | 0.2129 |
| 1 / 200 | 0.2107 |
| 2 / 100 | 0.2105 |
| 2 / 200 | **0.2103** |

| Split | Games | Forecast | Brier | 95% interval | Accuracy |
| --- | ---: | --- | ---: | --- | ---: |
| DEV | 1,123 | Gradient boosting | 0.2103 | [0.2008, 0.2201] | 66.9% |
| DEV | 1,123 | TRAIN home-win rate | 0.2480 | [0.2438, 0.2521] | 55.0% |
| TEST | 1,137 | Gradient boosting | 0.2058 | [0.1958, 0.2155] | 67.3% |
| TEST | 1,137 | TRAIN home-win rate | 0.2481 | [0.2441, 0.2523] | 55.0% |

The paired TEST difference, boosting minus home rate, has interval
[-0.0523, -0.0326]. These intervals resample ET game dates together, using 2,000
bootstrap draws. They condition on fitted models and observed dates; they do not
cover model-selection uncertainty or dependence across different dates.
Full-precision scores, calibration, season coverage and hashes are in
[`gradient_boosting.json`](../results/gradient_boosting.json).

BAM also needs statistical inputs for TRAIN games. For each season after the
first, the booster is fitted on earlier TRAIN seasons only. This yields 12,308
prior-season forecasts; 1,149 otherwise eligible first-season games have none.
The selected settings still came from DEV, so this is not a fully nested model
selection experiment for every historical TRAIN season.

## The market comparison at the prediction cutoff

The one-hour market forecast is the last home-outcome CLOB history point at or
before the cutoff, at most 60 minutes old. Markets are matched by teams and start
time; selection does not use prices or trading volume. Each row below compares
the same games within that row.

| Split | Matched games | Boosting Brier | Market Brier | Market minus boosting, paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| DEV | 1,121 | 0.2098 | 0.1990 | [-0.0166, -0.0049] |
| TEST | 1,132 | 0.2059 | 0.1941 | [-0.0180, -0.0055] |

Negative differences favor the market. TEST accuracy is 69.3% for the market and
67.2% for boosting on these 1,132 games. The booster's numbers differ slightly
from its full 1,137-game results because five games lack a matched market price.
These populations must not be mixed in a single comparison.

A logistic stack fitted on DEV using both forecasts did not show an additional
TEST gain over the market: Brier remained 0.1941, with paired interval
[-0.0002, 0.0002] versus the market alone. That result concerns this particular
stack and cohort, not proof that statistics can never add information. Details
are in [`polymarket.json`](../results/polymarket.json).

## Does an earlier market price leave more room?

The eight-hour horizon is a fixed proxy for an earlier notes-based decision. It
is not the measured publication time of every packet. This comparison requires
usable prices at both eight hours and one hour: at least five trades and $500
traded in the preceding two hours, a last trade no more than 60 minutes old, and
a history price within 0.05 of the volume-weighted trade price.

| Split | Common games | Boosting | Market at 8 h | Market at 1 h | 8 h minus boosting, paired 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| DEV | 953 | 0.2085 | 0.1973 | 0.1956 | [-0.0172, -0.0051] |
| TEST | 1,125 | 0.2055 | 0.1952 | 0.1938 | [-0.0160, -0.0044] |

All score columns are Brier. The market advantage remains at eight hours. Moving
to one hour improves its point estimate by about 0.0014 on TEST, much less than
the roughly 0.010 gap to boosting. However, the paired eight-hour-minus-one-hour
interval is [-0.0003, 0.0033] on TEST and [-0.0001, 0.0035] on DEV; these data do
not establish a nonzero benefit from the later price. See
[`polymarket_horizons.json`](../results/polymarket_horizons.json).

Neither market table is restricted to the selected notes corpus. A text model
must be compared with boosting and the market on its own identical eligible
games, rather than claiming improvement against these larger-population scores.

## Interpretation and historical qualifications

The evidence supports a strong statistical baseline and a better market
benchmark. It does not yet establish that text closes the gap. Retrospective
schedule, box-score and price retrieval also limits claims about what was
operationally available at the time. The market's tipoff reference is later than
the model cutoff and is not a fair competing forecast. Lifetime trading volume
includes post-cutoff activity and is descriptive only.

Earlier boosters used TRAIN 2023–24 and refitted on TRAIN plus DEV. Their TEST
Briers were 0.2099 with 18 features and 0.2059 with 234 rich features. Those are
historical protocols, not extra candidates selected for the current experiment.
Their original reports remain at Git revision
`67596558a188014803724bd60a158ce6f205ec3e`, under
`reports/past/basic_boosting_2023/` and `reports/past/rich_boosting_2023/`.

The current baseline scored TEST once after its own DEV selection, but this
season was already examined in prior work. These results are exploratory
backtests, not an untouched final benchmark.

## Verify the saved run

The original source hashes match Git revision
`a75d4cc00b6d89354c9c9b8af4bbad153fae9e4c`. Reporting paths subsequently changed;
the metrics, models and predictions did not. This command checks that historical
source, verifies current data/artifact hashes and replays predictions and metrics
with the current code without fitting:

```bash
.venv/bin/python scripts/validate_gradient_boosting.py --source-revision a75d4cc00b6d89354c9c9b8af4bbad153fae9e4c
.venv/bin/python scripts/validate_polymarket.py
```

Running `run_gradient_boosting.py` fits models and overwrites outputs; the market
evaluators also write outputs and the one-hour evaluator fits a stack. They are
not report-refresh commands. Leave frozen experiments intact.
