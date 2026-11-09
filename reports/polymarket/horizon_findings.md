# Market price at note-publication time versus one hour before tipoff

Game notes are typically published about 8 hours before tipoff, so the fair market reference for a notes-based forecast is the price 8 hours before tipoff. Each price is the last CLOB price-history point at or before tipoff minus the horizon (at most 60 minutes old). It counts only when the preceding 2 hours hold at least 5 trades and $500 traded, the last trade is at most 60 minutes old, and the price is within 0.05 of the trades' volume-weighted price. Only games usable at every horizon are scored, against the existing out-of-sample boosting predictions.

## Validation season 2024-25: 953 of 1123 games

| Forecast | Brier ↓ | 95% interval | Log loss ↓ | Accuracy |
| --- | ---: | --- | ---: | ---: |
| market_8h_before | 0.1973 | [0.1854, 0.2103] | 0.5775 | 69.9% |
| market_1h_before | 0.1956 | [0.1839, 0.2083] | 0.5734 | 70.3% |
| gradient_boosting | 0.2085 | [0.1984, 0.2193] | 0.6044 | 67.6% |

Paired Brier differences (negative favors the first):

- `market_8h_before_minus_gradient_boosting`: [-0.0172, -0.0051]
- `market_1h_before_minus_gradient_boosting`: [-0.0188, -0.0070]
- `market_8h_before_minus_market_1h_before`: [-0.0001, +0.0035]

## Test season 2025-26: 1125 of 1137 games

| Forecast | Brier ↓ | 95% interval | Log loss ↓ | Accuracy |
| --- | ---: | --- | ---: | ---: |
| market_8h_before | 0.1952 | [0.1835, 0.2067] | 0.5715 | 69.2% |
| market_1h_before | 0.1938 | [0.1820, 0.2052] | 0.5682 | 69.3% |
| gradient_boosting | 0.2055 | [0.1955, 0.2153] | 0.5984 | 67.5% |

Paired Brier differences (negative favors the first):

- `market_8h_before_minus_gradient_boosting`: [-0.0160, -0.0044]
- `market_1h_before_minus_gradient_boosting`: [-0.0178, -0.0055]
- `market_8h_before_minus_market_1h_before`: [-0.0003, +0.0033]
