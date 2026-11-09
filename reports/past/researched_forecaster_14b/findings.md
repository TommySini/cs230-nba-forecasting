# Full-note research plus original boosting

All archived PDF pages are processed. The frozen research model selects prose evidence and makes a separate full-page support-check pass, using preserved local and Modal executions. That checker failed two of twelve targeted factual challenges, so generated summaries are audit material only. The forecaster receives selected verbatim source passages with nearby context and code-derived table facts. Recognized Miami and Cleveland recent-player tables are decoded and averaged by code, with scoring/rebound identities and dates checked. Unrecognized table formats remain available in the source audit. Passage selection can still omit useful information, and an encoder can misinterpret correct source quotations.

The frozen September 2024 Qwen2.5-14B model encodes every retained evidence item across complete report chunks. It receives exact source quotations with nearby context and code-checked table facts; generated paraphrases and their inferred scopes/caveats are not prediction inputs. A 5,121-parameter linear head learns a correction to the original booster log odds. Only the head is trained; the 14B weights are unchanged. Regularization is selected using dev Brier, with a zero-correction candidate eligible. No dev refit, probability calibration or threshold search.

| Split | Games | First date | Last date |
| --- | ---: | --- | --- |
| train | 80 | 2024-11-04 | 2025-12-01 |
| dev | 13 | 2025-12-04 | 2026-01-31 |
| test | 14 | 2026-02-01 | 2026-04-10 |

Dev selected **l2_0.001**, Brier 0.332851.

| Model | Test Brier ↓ | Log loss ↓ | Accuracy |
| --- | ---: | ---: | ---: |
| original_boosting | 0.186296 | 0.558920 | 78.6% |
| researched_hybrid | 0.193206 | 0.575385 | 78.6% |

Hybrid minus original Brier: **+0.006910**; paired date-bootstrap 95% interval [-0.011765788935868264, 0.029195763578169193]. Negative favors the hybrid. This interval excludes model-fitting and selection uncertainty.

**Exploratory repeated historical test.** These 14 outcomes were inspected in the earlier pilot; they are not a fresh benchmark. Choices in this revision used training-packet research quality and dev performance, not these test outcomes. Coverage remains selective by team/date, with 11 of 14 test packets published by Cleveland.

The two-system comparison measures the full research-conditioned correction, including extra numeric facts and possible recalibration. It does not isolate narrative text alone. Cited reports make the research inspectable, but the learned vector head does not establish which passages caused a probability adjustment.

Training booster probabilities are chronological out-of-sample using the earlier frozen fold lineage. Both systems use exactly the same held-out games. The original booster remains unchanged.
