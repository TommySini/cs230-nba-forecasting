# TabPFN statistical experiments

All comparisons use the same 1,123 DEV games (2024–25) and 252 frozen pregame features. TRAIN is 2012–24; TEST is excluded. Lower Brier/log loss is better.

| Model | Brier | Log loss | AUC | Accuracy | Brier difference vs frozen TabPFN (95% interval) |
| --- | ---: | ---: | ---: | ---: | --- |
| rich_gradient_boosting | 0.210332 | 0.608395 | 0.7237 | 66.9% | — |
| tabpfn | 0.207831 | 0.601664 | 0.7303 | 66.8% | — |
| synthetic | 0.208017 | 0.602082 | 0.7296 | 67.1% | +0.000186 [-0.000243, +0.000606] |
| finetune | 0.207831 | 0.601664 | 0.7303 | 66.8% | +0.000000 [+0.000000, +0.000000] |
| combined | 0.208017 | 0.602082 | 0.7296 | 67.1% | +0.000186 [-0.000243, +0.000606] |

## Observed outcome

None of the three prespecified variants improved DEV Brier over frozen TabPFN.
Synthetic-only and combined are slightly worse on Brier/log loss, although accuracy
increased by three of 1,123 games. Their Brier difference interval includes zero.

Both fine-tuning attempts performed real gradient updates, but every completed
epoch worsened log loss on the separate 2023–24 TRAIN validation season. Early
stopping retained the initial pretrained weights. Fine-tuning-only predictions
therefore match original TabPFN exactly; combined predictions match synthetic-only
exactly (maximum per-game probability difference is zero in both comparisons).

| Fine-tuning arm | Updates | Epochs | Initial inner log loss | Epoch 1 | Epoch 2 | Selected weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Real TRAIN only | 96 | 2 | 0.604215 | 0.606617 | 0.607731 | Original pretrained |
| Real TRAIN + synthetic | 104 | 2 | 0.605741 | 0.612362 | 0.617062 | Original pretrained |

The official package warns that one estimator during training/validation versus
four for final inference, with fixed preprocessing seeds, can affect generalization.
This configuration was fixed before the experiment and shared by both arms.
These results cover one seed and one bounded configuration, not all possible
fine-tuning or synthetic-generation approaches.

All runs completed locally. Synthetic generation took 62.7 seconds; DEV inference
114.3 seconds. Real-only fine-tuning took 265.3 seconds plus 135.2 seconds for DEV;
combined fine-tuning took 284.9 seconds plus 134.5 seconds for DEV. All three saved
result validators pass. The full suite ran 371 tests with one optional test skipped.

## Protocol

The original TabPFN-2.5 result used fixed pretrained weights and all 13,457 TRAIN games as context, with four estimators on Apple MPS. Its Brier difference versus boosting was -0.002500 (paired 95% interval [-0.005122, +0.000148]); this did not establish an advantage.

The adaptation study uses the official TabPFN unsupervised generation and fine-tuning tools. Synthetic-only and combined share 1,024 generated rows, using a deterministic sample of 1,024 real 2012–22 TRAIN games as the generator context. The continuous regressor checkpoint could not be downloaded (HuggingFace TLS timeouts; official fallback HTTP404), so the official classifier-only joint sampler uses up to eight TRAIN-fitted quantile bins per primitive feature. Bin medians are decoded and engineered arithmetic/missingness identities reconstructed. This is a coarse binned generation experiment, not an evaluation of the continuous-regressor generator. Fine-tuning uses the earlier TRAIN seasons for weight updates and the 2023–24 TRAIN season for checkpoint selection by log loss. Final prediction context includes all original TRAIN games, plus the same synthetic rows in augmentation arms. Both fine-tuning arms use the same fixed training settings and budget.

Fine-tuning is bounded to five epochs, learning rate 1e-5, 256-row meta-batches, one estimator during updates/inner validation, patience two, and four estimators for final prediction. The selected checkpoint may remain the original pretrained weights if no epoch improves inner-TRAIN log loss; per-run metadata records that explicitly.

Official documentation: [data generation](https://docs.priorlabs.ai/capabilities/data-generation), [fine-tuning](https://docs.priorlabs.ai/capabilities/fine-tuning).

Intervals use 2,000 paired ET-calendar-day bootstrap replicates. These are exploratory, unadjusted comparisons of three variants, conditional on the fitted models; they do not include training/generation seed uncertainty or dependence across days. DEV had already been inspected and used to select the booster. Synthetic rows are model-generated examples, not new observed games.

Frozen baseline artifacts remain in `data/processed/tabpfn/dev/`; experiment manifests, separate synthetic features/labels, training checkpoints and per-game predictions are under `data/processed/tabpfn/adaptation/`. Recompute scores with `validate_tabpfn.py --variant <name>`.
