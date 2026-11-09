# TabPFN statistical baseline

Stage: **dev**. The paired DEV interval includes zero; this run does not establish an advantage.

Both models predict the same 1,123 query games using 252 frozen pregame features. TabPFN context: 13,457 earlier TRAIN games. TEST is excluded.

| Model | Brier | Log loss | Accuracy |
| --- | ---: | ---: | ---: |
| tabpfn | 0.207831 | 0.601664 | 66.8% |
| rich_gradient_boosting | 0.210332 | 0.608395 | 66.9% |

Brier difference (TabPFN minus boosting): -0.002500; paired ET-date bootstrap 95% interval [-0.005122, +0.000148]. Negative favors TabPFN.

TabPFN uses the pinned synthetic-only v2.5 default-2 checkpoint with fixed pretrained weights, 4 ensemble members and no fine-tuning. Model inputs contain only the baseline feature matrix; outcomes serve as context for earlier TRAIN games and as separate evaluation targets for query games.

Comparator: preserved full-TRAIN DEV-selected booster forecasts. DEV is exploratory: the saved booster's settings were selected on DEV. The bootstrap is conditional on these fitted models and does not include training uncertainty.

Local device: mps; recorded elapsed time: 236.6 s. The source manifest, checkpoint, packages, cohort and per-game forecasts are pinned in the run artifacts.
