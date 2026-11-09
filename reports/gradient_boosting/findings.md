# Gradient-boosting statistics baseline

TEST 2025–26: Brier **0.2058**, log loss 0.5989, accuracy 67.3% on 1,137 games, versus 0.2481 for the TRAIN home-win rate. Settings chosen on DEV 2024–25: depth 2, 200 trees.

## Protocol

- Splits by season: TRAIN 2012–13 to 2023–24, DEV 2024–25, TEST 2025–26 (`config/gradient_boosting.json`). Regular-season games from checksum-verified ESPN schedules.
- Features (252): 18 basic team features (games played, season and last-5 win rate, last-5 point differential, rest, back-to-back) plus team and player box-score windows (season, last 5, last 10) with team-coverage indicators; home, away and home-minus-away. All use earlier ET dates of the same season only.
- Eligibility: both teams have at least 5 earlier games and 5 valid earlier box scores. Other missing box scores reduce coverage: team statistics pool the window's valid games, coverage features record the gap, and player statistics are missing unless the whole window has player box scores.
- Models: gradient boosting with median imputation fitted on each model's own training rows. 4 fixed candidates are fitted on TRAIN and chosen by DEV Brier. DEV and TEST forecasts both come from that TRAIN fit; TEST was scored once, after selection. No early stopping, calibration or TEST-driven tuning.
- TRAIN forecasts (`produced_by: prior_seasons`): each TRAIN season after the first is predicted by the selected settings fitted on the earlier TRAIN seasons only, so every forecast BAM receives is fitted on strictly earlier games (the choice among the 4 settings used DEV). The first TRAIN season (2012–13) has none.

## DEV selection

| Depth | Trees | DEV Brier | Log loss | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 0.2129 | 0.6145 | 65.7% |
| 1 | 200 | 0.2107 | 0.6093 | 66.1% |
| 2 | 100 | 0.2105 | 0.6091 | 66.3% |
| 2 | 200 | 0.2103 | 0.6084 | 66.9% |

## Results

| Split | Forecast | Games | Brier | 95% interval | Log loss | Accuracy |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| DEV | gradient_boosting | 1,123 | 0.2103 | [0.2008, 0.2201] | 0.6084 | 66.9% |
| DEV | historical_home_rate | 1,123 | 0.2480 | [0.2438, 0.2521] | 0.6892 | 55.0% |
| DEV | constant_50_percent | 1,123 | 0.2500 | [0.2500, 0.2500] | 0.6931 | 55.0% |
| TEST | gradient_boosting | 1,137 | 0.2058 | [0.1958, 0.2155] | 0.5989 | 67.3% |
| TEST | historical_home_rate | 1,137 | 0.2481 | [0.2441, 0.2523] | 0.6893 | 55.0% |
| TEST | constant_50_percent | 1,137 | 0.2500 | [0.2500, 0.2500] | 0.6931 | 55.0% |

DEV paired Brier difference, boosting minus home rate: [-0.0468, -0.0287] (152 ET dates resampled).
TEST paired Brier difference, boosting minus home rate: [-0.0523, -0.0326] (154 ET dates resampled).

## TRAIN forecasts from earlier seasons

| Season | Games | Brier | Log loss | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 2013–14 | 1,153 | 0.2196 | 0.6309 | 65.0% |
| 2014–15 | 1,151 | 0.2082 | 0.6048 | 68.4% |
| 2015–16 | 1,152 | 0.2091 | 0.6055 | 68.5% |
| 2016–17 | 1,151 | 0.2237 | 0.6383 | 63.8% |
| 2017–18 | 1,150 | 0.2147 | 0.6196 | 65.0% |
| 2018–19 | 1,151 | 0.2144 | 0.6183 | 65.6% |
| 2019–20 | 982 | 0.2239 | 0.6389 | 65.1% |
| 2020–21 | 1,000 | 0.2255 | 0.6425 | 63.2% |
| 2021–22 | 1,152 | 0.2230 | 0.6380 | 64.3% |
| 2022–23 | 1,151 | 0.2277 | 0.6466 | 62.6% |
| 2023–24 | 1,115 | 0.2132 | 0.6144 | 64.2% |
| all | 12,308 | 0.2183 | 0.6268 | 65.1% |

1,149 eligible TRAIN games have no earlier TRAIN season and no forecast.

## Coverage by season

Box scores: `data/processed/box_scores`.

| Season | Split | Games | Box records | Eligible | Excluded: warmup | Excluded: box coverage | With team-box gaps | With player-box gaps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2012–13 | train | 1,229 | 1,228 | 1,149 | 80 | 0 | 65 | 724 |
| 2013–14 | train | 1,230 | 1,229 | 1,153 | 77 | 0 | 54 | 820 |
| 2014–15 | train | 1,230 | 1,230 | 1,151 | 79 | 0 | 0 | 769 |
| 2015–16 | train | 1,230 | 1,229 | 1,152 | 78 | 0 | 105 | 614 |
| 2016–17 | train | 1,230 | 1,229 | 1,151 | 79 | 0 | 103 | 697 |
| 2017–18 | train | 1,230 | 1,224 | 1,150 | 80 | 0 | 155 | 981 |
| 2018–19 | train | 1,230 | 1,230 | 1,151 | 79 | 0 | 0 | 520 |
| 2019–20 | train | 1,059 | 1,059 | 982 | 77 | 0 | 0 | 188 |
| 2020–21 | train | 1,080 | 1,079 | 1,000 | 80 | 0 | 75 | 387 |
| 2021–22 | train | 1,230 | 1,230 | 1,152 | 78 | 0 | 0 | 42 |
| 2022–23 | train | 1,230 | 1,230 | 1,151 | 79 | 0 | 0 | 143 |
| 2023–24 | train | 1,196 | 1,196 | 1,115 | 81 | 0 | 0 | 571 |
| 2024–25 | dev | 1,201 | 1,201 | 1,123 | 78 | 0 | 0 | 425 |
| 2025–26 | test | 1,216 | 1,216 | 1,137 | 79 | 0 | 0 | 268 |

## Limits

- Historical box scores were retrieved retrospectively. Earlier ET game dates are used, with explicit stat_available_at_utc/completed_at_utc enforced when present; historical correction times and exact result availability are not independently archived. Possessions and ratings are box-score estimates, not official NBA figures.
- No injury, lineup, odds or text inputs. Prior-game workload is not current-game availability.
- The 2025–26 TEST season was examined in earlier experiments, so this is an exploratory backtest. The previous protocol (TRAIN 2023–24 only, final refit on 2023–25) scored 0.2099 (basic 18 features) and 0.2059 (rich features) on it: [basic](../past/basic_boosting_2023/findings.md), [rich](../past/rich_boosting_2023/findings.md).
- Bootstrap intervals are conditional on the fitted models.

## Reproduce

```bash
.venv/bin/python scripts/run_gradient_boosting.py
.venv/bin/python scripts/validate_gradient_boosting.py
```

Runtime 216 s. Inputs, labels, exclusions, predictions (`p_boost`, `produced_by`) and `model.joblib` (fit game IDs and settings) are in `data/processed/gradient_boosting/`; full metrics, hashes and versions in `metrics.json`.
