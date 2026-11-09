# Polymarket market baseline

Market-implied home-win probability at the same prediction cutoff as the boosting baseline (scheduled tipoff minus 60 minutes), scored on identical games.

Held-out 2025–26 games with both forecasts: **1132** of 1137. Polymarket Brier **0.1941** (accuracy 69.3%) versus boosting **0.2059** (accuracy 67.2%). Paired Brier difference, market minus boosting: **[-0.0180, -0.0055]**; negative favors the market.

## Coverage

Polymarket NBA game moneyline markets for the 2024-25 and 2025-26 regular seasons; Polymarket listed no per-game NBA markets during 2023-24.

| Season | Games | Cutoff price | No market | Ambiguous | Stale / no pregame price | Fetch failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024–25 | 1204 | 1199 | 5 | 0 | 0 | 0 |
| 2025–26 | 1220 | 1211 | 8 | 0 | 1 | 0 |

Median price point is 10 minutes before the cutoff (maximum 49; limit 60). Matched-market lifetime volume: median $579,192, 10th percentile $56,124.

## Test season 2025–26 (1132 games, 2025-10-31 to 2026-04-10)

| Forecast | Brier ↓ | 95% interval | Log loss ↓ | ROC AUC ↑ | Accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| polymarket_cutoff | 0.1941 | [0.1820, 0.2056] | 0.5689 | 0.7669 | 69.3% |
| gradient_boosting | 0.2059 | [0.1958, 0.2157] | 0.5993 | 0.7395 | 67.2% |
| historical_home_rate | 0.2484 | [0.2444, 0.2525] | 0.6899 | 0.5000 | 54.8% |
| polymarket_tipoff_reference | 0.1936 | [0.1817, 0.2050] | 0.5678 | 0.7682 | 69.7% |

Paired differences (negative Brier / positive accuracy favors the first forecast):

- `polymarket_cutoff_minus_gradient_boosting`: Brier [-0.0180, -0.0055]; accuracy [-0.2%, +4.3%]
- `polymarket_cutoff_minus_historical_home_rate`: Brier [-0.0661, -0.0426]; accuracy [+10.9%, +18.2%]

Markets with lifetime volume ≥ $50,000 (971 games): Polymarket Brier 0.2008, boosting 0.2115.

## Validation season 2024–25 (1121 games, 2024-11-01 to 2025-04-10)

| Forecast | Brier ↓ | 95% interval | Log loss ↓ | ROC AUC ↑ | Accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| polymarket_cutoff | 0.1990 | [0.1889, 0.2092] | 0.5814 | 0.7552 | 69.0% |
| gradient_boosting | 0.2098 | [0.2005, 0.2188] | 0.6072 | 0.7255 | 67.0% |
| historical_home_rate | 0.2480 | [0.2438, 0.2522] | 0.6891 | 0.5000 | 55.0% |
| polymarket_tipoff_reference | 0.1987 | [0.1884, 0.2089] | 0.5808 | 0.7558 | 69.5% |

Paired differences (negative Brier / positive accuracy favors the first forecast):

- `polymarket_cutoff_minus_gradient_boosting`: Brier [-0.0166, -0.0049]; accuracy [-0.0%, +4.3%]
- `polymarket_cutoff_minus_historical_home_rate`: Brier [-0.0592, -0.0385]; accuracy [+10.5%, +17.4%]

Markets with lifetime volume ≥ $50,000 (1087 games): Polymarket Brier 0.2005, boosting 0.2119.

## Does boosting add information beyond the market?

A two-feature logistic stack, logit(market) and logit(boosting), was fitted on the 2024–25 season and scored once on 2025–26. Coefficients: market 0.990, boosting 0.001, intercept -0.018. Test Brier 0.1941; paired difference versus the market alone [-0.0002, 0.0002]. An interval below zero would mean the boosting features carry information the market price lacks.

## Protocol and limits

- Discovery: Polymarket Gamma `/markets`, NBA tag, `moneyline` type, closed markets, paged by end-date window. Games match on both team names and a market start within 180 minutes of the ESPN scheduled start. Primary `nba-{away}-{home}-{date}` slugs win over low-volume `-moneyline` duplicates; selection never uses volume or prices.
- Price: last CLOB price-history point (10-minute fidelity) for the home outcome token at or before the cutoff, at most 60 minutes old. The request ends at scheduled tipoff, so no in-game prices are fetched. `polymarket_tipoff_reference` is the last pregame point and is later than the model cutoff; it is a reference, not a fair competitor.
- Prices come from exchange trading history retrieved retrospectively through Polymarket's API; the API's sampling is not independently archived. Lifetime volume includes post-cutoff trading and is used only to describe liquidity.
- Label consistency: market resolution agrees with the ESPN outcome for 2410 games, disagrees for 0, unresolved 0.
- The 2025–26 test season was already examined for the boosting baseline; this is a repeated evaluation on that season. Polymarket listed no per-game NBA markets in 2023–24, so the training season has no market comparison.
- Polymarket keeps price history for closed markets, so the same collector can score a later held-out period (for example 2026–27) after games finish.

## Reproduce

```bash
.venv/bin/python scripts/nba_dataset.py validate
.venv/bin/python scripts/evaluate_polymarket.py
.venv/bin/python scripts/validate_polymarket.py
.venv/bin/python -m unittest tests.test_polymarket -v
```
