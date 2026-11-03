# Archived scripts

Not part of the active experiment path. Kept so history and one-off Modal
deployments remain inspectable.

| File | Why archived |
| --- | --- |
| `modal_research_batch.py` | Old MLX full-research Modal app. Superseded by `scripts/modal_research_vllm.py`. Do not deploy or run for new work. |

Active Modal entry point: `scripts/modal_research_vllm.py`. Shared volume/`unpack`
helpers still live in `scripts/modal_research.py` (do not move that file here
without a deliberate import refactor).
