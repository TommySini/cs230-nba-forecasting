# Matched pregame-text pilot

Does verified pregame injury text improve predictions beyond prior-game statistics?

## Controlled comparison

Both arms use the same pinned Qwen2.5-0.5B-Instruct backbone, a binary probability head, and LoRA on query/value projections. The head starts at zero, and both arms have identical initial trainable weights, game ordering, optimizer settings and training budgets. Statistics are serialized identically; only the injury-text block changes. Team headings are replaced by HOME/AWAY. Player names remain in the text arm. No game dates, IDs, target outcomes, raw ESPN content or unapproved narratives enter either prompt.

This is a compact language-model adaptation of the forecasting idea, not a BAM replication. BAM uses research agents, a fine-tuned GPT-4o forecaster, hindsight-guided report supervision and an ensemble. Here we directly train a binary probability head on game outcomes, with no research agents, generated reasoning targets, market smoothing or ensemble.

## Split

The user approved whole-date chronological 108/14/18 instead of an exact 105/17/18 split that would cut through a game date.

| Split | Games | First date | Last date |
| --- | ---: | --- | --- |
| train | 108 | 2023-11-03 | 2025-11-05 |
| dev | 14 | 2025-11-07 | 2026-01-16 |
| test | 18 | 2026-01-17 | 2026-04-10 |

Checkpoint revision `7ae557604adf67be50417f59c2c2f167def9a775` is dated 2024-09-25, before dev/test. The original 40/48/52 counts described the old season split only; this experiment trains new models on the new split. Historical games outside the cohort supply prior-game statistics but are not supervised training examples.

## Training and selection

Five epochs per arm; LoRA rank 4, alpha 8, dropout 0.0; adapter learning rate 0.0001, head rate 0.001; AdamW weight decay 0.01; batch 4 with accumulation 4; seed 230. Binary cross-entropy trains both arms. No calibration or threshold search; threshold 0.5. Full prompts must fit 1024 tokens; truncation is rejected.

One common epoch (**1**) is selected by the mean dev Brier across the two arms, so both selected models receive the same number of training passes. No refit on dev. `selection.json` locks the chosen weights and dev results before test evaluation.

## Results on the identical 18 test games

| Model | Brier ↓ | Log loss ↓ | Accuracy at 0.5 |
| --- | ---: | ---: | ---: |
| stats_only | 0.236997 | 0.666035 | 55.6% |
| stats_plus_text | 0.238987 | 0.671024 | 66.7% |
| original_boosting_reference | 0.174847 | 0.532168 | 77.8% |

Paired Brier difference (text minus stats): **0.001990**. A negative value favors text. Paired date-bootstrap 95% interval: [-0.03525847762702463, 0.037594191635378055]. This conditions on fitted models and sampled dates; it excludes training and selection uncertainty.

The original boosting model is a reference only: it was trained on a different, much larger cohort. Only the two language-model arms form the matched experiment.

## Limits

Only 18 test games and 14 dev games; both estimates and epoch selection are unstable. This is one fixed training seed and one small language model. The text consists of injury excerpts, not rich narratives, and missing rows do not prove a healthy roster. Text includes player identity as well as injury information. These test games were inspected in previous baseline evaluations; repartitioning does not make them an untouched final benchmark. A positive or negative result cannot establish whether all pregame text helps in general.

## Reproduce

```bash
uv pip install --python .venv/bin/python -r requirements-text.txt
.venv/bin/python scripts/run_text_ablation.py --stage prepare
.venv/bin/python scripts/run_text_ablation.py --stage train --device mps
.venv/bin/python scripts/run_text_ablation.py --stage evaluate --device mps
.venv/bin/python scripts/validate_text_ablation.py --device mps
```

The pinned public checkpoint must first be downloaded to the model path in the config. Thereafter all stages are offline. Inputs, separate labels, adapters, split manifest and predictions are in `data/processed/text_ablation/`. The original baseline and raw sources are preserved.

Sources: [BAM architecture and training](https://arxiv.org/html/2607.09921v1#S4), [Qwen2.5 release](https://qwenlm.github.io/blog/qwen2.5/), [LoRA/PEFT](https://huggingface.co/docs/peft/en/quicktour).
