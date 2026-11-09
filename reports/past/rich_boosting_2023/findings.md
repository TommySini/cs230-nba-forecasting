# Rich-statistics gradient boosting

Primary DEV-selected richer model: **team_player_selected**. On the same 1,137 held-out games, Brier is **0.205928** versus **0.209868** for the original features. Difference (richer minus original): **-0.003940**, paired date-bootstrap 95% interval **[-0.007222, -0.000679]**. Negative favors richer features.

## Same-capacity feature comparison

All rows below use the original fixed 100 depth-1 trees and otherwise identical training settings.

| Features | Count | Brier | Log loss | AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 18 | 0.209868 | 0.608507 | 0.7312 | 67.3% |
| team | 162 | 0.206724 | 0.601318 | 0.7399 | 67.3% |
| team_player | 234 | 0.205928 | 0.599430 | 0.7422 | 68.2% |

## Matched DEV-selected comparison

Every feature arm receives the same four candidates (depth 1/2 × 100/200 trees). The richer arm and its settings were selected using DEV, before TEST scoring.

| Features | Selected depth / trees | DEV Brier | TEST Brier | TEST accuracy |
| --- | --- | ---: | ---: | ---: |
| original | 1 / 100 | 0.216867 | 0.209868 | 67.3% |
| team | 1 / 100 | 0.216344 | 0.206724 | 67.3% |
| team_player | 1 / 100 | 0.216318 | 0.205928 | 68.2% |

## Data and temporal boundaries

| Split | Original eligible | Team-source eligible | Common paired cohort | Any missing player window |
| --- | ---: | ---: | ---: | ---: |
| train | 1115 | 1115 | 1115 | 571 |
| validation | 1123 | 1123 | 1123 | 425 |
| test | 1137 | 1137 | 1137 | 268 |

TRAIN: 2023–24; DEV: 2024–25; TEST: 2025–26, through April 10. Final models refit TRAIN + DEV with settings fixed. Within TEST, earlier completed games update rolling features but model weights are not refit.

Source coverage: 3,613 complete team box scores, 3,456 complete player box scores. Providers: {"ESPN": 3607, "NBA": 6}. Audited official NBA pages repair invalid ESPN records; the original raw records and replacement-source mapping remain preserved.

The source audit excludes 8 All-Star games involving non-franchise teams that ESPN labeled with regular-season type metadata. They were already excluded by the original five-game warmup and do not change the original eligible forecasting cohort. Original history and baseline artifacts remain intact.

Features are calculated only from earlier Eastern-calendar dates in the same season. Full prior TEAM box-score coverage is required. Team ratios use pooled box-score counts; possession-derived ratings and pace are estimates, not official NBA ratings. Player features summarize prior workload and rotation continuity; actual current-game participants, final lineups and injury outcomes are excluded.

Missing player windows retain their actual game membership and carry explicit coverage indicators. Their missing numeric values use medians learned only from TRAIN during DEV selection, then TRAIN+DEV for final fitting. No TEST data informs imputation. Ambiguous minute placeholders are not treated as zero minutes.

Historical public box scores were retrieved retrospectively. Source hashes and dates are preserved, but original publication times and later stat corrections are not independently archived. Only normalized numeric fields reach the feature builder.

Current-game injury availability was not added: the existing report sample covers only 140/3,375 original eligible games and has additional timestamp/coverage issues. A missing injury report is not interpreted as a healthy roster.

Original full-test prediction reproduction on the complete cohort: **True**. If the matched cohort differs, the original-feature arms are refit on its same TRAIN/DEV games; the published full-cohort result is separately retained in metrics.json.

## Interpretation limits

This TEST season has already been examined in prior experiments. Treat this as an exploratory backtest. The paired interval is conditional on fitted models and does not include training uncertainty or correlation across calendar days. Added features can fail to improve prediction; all declared arms are reported. No text model was trained or evaluated by this run.

## Verify the frozen run

```bash
.venv/bin/python scripts/validate_rich_gradient_boosting.py
```

The validator reconstructs every retained numeric source and feature row, checks
TRAIN/DEV fitting membership and imputation, reproduces all saved TEST forecasts
and metrics, and verifies that the original baseline artifacts are unchanged.
See `validation.json` for the completed checks.

The existing prediction lock prevents accidental retraining. To run a new
experiment, copy `config/rich_gradient_boosting.json` to a new config with new
output and report directories, then use `--config PATH --stage all`. Preserve the
current source snapshot and its manifest; rebuilding it in place changes the
provenance timestamps and invalidates the frozen run. `feature_definitions.csv`
and `feature_missingness.csv` document the full feature set and coverage.
