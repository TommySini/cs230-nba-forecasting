# Original boosting versus boosting plus pregame game notes

The original statistics-only booster is unchanged. The hybrid consumes its probability and archived game-note evidence, then adds a learned correction to its log odds. Exactly these two systems are compared on the same games.

| Split | Games | First date | Last date |
| --- | ---: | --- | --- |
| train | 80 | 2024-11-04 | 2025-12-01 |
| dev | 13 | 2025-12-04 | 2026-01-31 |
| test | 14 | 2026-02-01 | 2026-04-10 |

Training probabilities are chronological out-of-sample: expanding-month booster fits use only earlier dates, and the frozen original booster is used after its final fitting date. The original fixed hyperparameters are reused without a new search. Their earlier historical selection is not a nested prospective experiment. Hybrid dev and test games were fitted by neither system.

All hybrid-cohort games occur after the pinned September 2024 Qwen2.5-0.5B-Instruct checkpoint. Older collected packets remain in the source audit only. LoRA rank 4 on query/value projections, zero-initialized residual probability head, binary cross-entropy, five epochs, seed 230. Adapter/head learning rates .0001/.001, effective batch 16, AdamW weight decay .01, clipping 1. Epoch 0 is the unchanged starting prediction and is eligible for selection. No dev refit or probability calibration; classification threshold .5.

Epoch **2** selected using dev Brier only. Note evidence uses a fixed front-page source-order selection with a 2700-token budget; source and selected lengths are recorded per game. The entire packets remain preserved, but are not all passed to the model.

| Model | Brier ↓ | Log loss ↓ | Accuracy |
| --- | ---: | ---: | ---: |
| original_boosting | 0.186296 | 0.558920 | 78.6% |
| boosting_plus_notes | 0.263225 | 0.775461 | 57.1% |

Hybrid minus boosting Brier: **+0.076929**; paired date-bootstrap 95% interval [-0.036184777655606466, 0.18663209646543877]. Negative favors the hybrid. This interval conditions on fitted models and observed dates, excluding training/selection uncertainty and longer-range dependence.

This is a BAM-inspired compact probability forecaster, not a full reproduction of research agents, hindsight-supervised reports, GPT-4o or ensemble forecasts. Rich notes add player/team statistics as well as narrative. A two-system improvement cannot isolate text from recalibration or these extra numeric facts.

Archive coverage is selective by team/date; see the collection audit. These are previously inspected historical test-season games, so results are exploratory. One seed and a small held-out cohort limit conclusions. MPS may use nondeterministic training kernels; prediction replay is checked within tolerance.

Reproduce with `scripts/collect_game_notes.py --offline`, `scripts/review_game_notes.py`, then `scripts/run_boosting_notes.py --stage prepare/train/evaluate --device mps` (one stage per command), and `scripts/validate_boosting_notes.py --device mps`.
