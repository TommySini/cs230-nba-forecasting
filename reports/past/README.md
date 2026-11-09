# Finished pilots

Findings from early experiments whose code has been removed. Each folder keeps the
findings write-up (and metrics where there were any). The full code, configs, protocol
files and per-game outputs are at git tag `pre-cleanup`.

| Pilot | Test games | Without text | With text | Folder |
| --- | ---: | ---: | ---: | --- |
| Stats vs stats + injury-report text; Qwen2.5-0.5B + LoRA | 18 | 0.237 | 0.239 | [injury_text_ablation](injury_text_ablation/findings.md) |
| Boosting vs boosting + first-page notes; Qwen2.5-0.5B + LoRA | 14 | 0.186 | 0.263 | [boosting_first_page_notes](boosting_first_page_notes/findings.md) |
| Boosting vs boosting + full-packet research; frozen Qwen2.5-14B + linear head | 14 | 0.186 | 0.193 | [researched_forecaster_14b](researched_forecaster_14b/findings.md) |

Brier, lower is better. None shows a text gain; the 14B paired CI against boosting was
[−0.012, +0.029]. All three used small, previously inspected 2025–26 test games.

- [dataset_pilot](dataset_pilot/findings.md): the first 100-game feasibility audit of injury
  reports, previews and Cleveland notes.
- [research_quality_audit](research_quality_audit/findings.md): why the 14B research switched
  from paraphrases to verbatim quotes and code-checked tables.
- [basic_boosting_2023](basic_boosting_2023/findings.md) and [rich_boosting_2023](rich_boosting_2023/findings.md):
  the earlier boosting baselines (TRAIN 2023–24 only, final refit on 2023–25), 0.2099 and 0.2059
  TEST Brier on 2025–26 with 18 basic and 234 rich features. Replaced by the single
  `run_gradient_boosting.py` pipeline with TRAIN 2012–13 to 2023–24.
