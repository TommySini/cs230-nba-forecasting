# A bounded $100 experiment

**November 7 update:** the user has now set a **$200** experiment cap; the $100
allocation below is historical planning. The next phase is TRAIN-only fitting
diagnostics and full TRAIN learning, with DEV deferred and TEST excluded. The approximately 200 gold
reports below were a cost proposal, not a fixed training-set limit. Follow the
[training-first protocol](../../../docs/bam_training_first.md) for the current
execution order and implementation prerequisites.

2026-11-03. Planning only: no paid requests, cloud jobs, budget changes, or production changes.

A lean experiment appears feasible, conditional on a small research-quality and
joint-training throughput pilot. The unchanged multi-stage workflow does not fit.
The following are **spending allocations, not verified completion estimates**.

| Stage | Allocation |
| --- | ---: |
| Full-note research and semantic checks | $35.00 |
| About 200 accepted TRAIN hindsight/gold reports and independent review | $20.00 |
| One joint forecaster/head training run, loading, validation and evaluation | $25.00 |
| Previously recorded completed estimates plus uncertain reservation | $8.59 |
| Qualification, repairs, CPU/storage and remaining contingency | $11.41 |
| Total within the existing ledger's accounting scope | **$100.00** |

The $8.59 is the existing ledger's $6.77 completed estimate plus $1.82 uncertain
reservation, not an invoice. Earlier project costs are unknown. This allocation
leaves at most $91.41 for additional work in that scope; it does not certify the
entire project's historical expenditure. Do not assume free credits, tax treatment,
provider availability, successful first responses, or unmeasured cache discounts.

## Keep and simplify

- Read all pages/sections of the eligible notes, then produce a compact cited
  research brief. Full-source coverage does not require forwarding the whole PDF
  to every downstream model. Preserve unreadable-page and undecoded-table flags.
- Use one frozen prompted researcher for every split. Do not train researcher
  adapters or run their chronological out-of-fold training/generation. Freeze
  prompts on TRAIN and require a documented pre-DEV knowledge cutoff for every
  held-out reader, reviewer and repair model.
- Use a stronger modern API teacher only on a representative TRAIN subset for
  postmortems and as-of-cutoff gold. The outcome can guide which pregame evidence
  matters; it must never enter a prediction input. Target about 200 accepted gold
  examples, not a gold report for every training game. Review them independently.
- Train on all eligible outcomes (currently about 1,506 TRAIN games), with a
  separately normalized explanation loss on that gold subset. Use one already
  integrated Qwen2.5-14B LoRA plus residual probability head. This is a cost and
  implementation choice, not evidence it outperforms GPT-oss or Gemma.
- Keep the original selected boosting hyperparameters, reconstruct prior-only
  forecasts, and compare the two systems on identical DEV/TEST games using Brier.
  Choose a small epoch budget before full fitting; no model grid or multiple seeds.

Sparse gold and the frozen-reader route are not implemented in production.
The full dataset also still requires the source/history checks in the companion
expansion review. Provisional research counts are 1,632/201/147; potential
forecast counts are 1,506/190/140. Historical test exposure remains a limitation.

## Why the allocation is plausible, and what is still unmeasured

The cached source audit counted 105,983,157 Qwen tokens across 1,980 games. Adding
1,000 formatting/prompt tokens per game gives 107,963,157 input tokens; allowing
2,000 output tokens per game gives 3,960,000 output tokens. These are **not provider
token counts**. At GPT-4.1-mini's derived Batch rates of $0.20/$0.80 per million
input/output tokens, that illustrative first research pass is **$24.76**.
The $35 allocation therefore has about $10.24 for semantic checks and overhead at
that proxy size, before using contingency. Exact API tokenization, chunking,
layout preservation, model access and actual quality must be checked first.

GPT-4.1-mini is a candidate, not selected or qualified: its documented June 1,
2024 knowledge cutoff precedes DEV, its context is about one million tokens,
and the snapshot is marked deprecated. Account access and the shutdown schedule
must be checked before starting. GPT-4o-mini is a possible distinct eligible
reviewer (October 2023 cutoff), but its lower cost is not proof it catches the
errors that matter. Review contextual meaning, dates, player identity, table
columns and omissions; source-string matching alone is insufficient.

The $20 gold allocation assumes compact reports with linked source evidence and
bounded postmortem, synthesis and independent-review calls. It is not a measured
price for 200 accepted targets. The old, much longer gold format projected about
$17.60–$32.94 for 200 targets even before retries at the Sol/Terra Batch rates.
Exact requests and a pilot must justify compacting the format without losing
useful analysis. Do not quietly reduce the accepted target count or skip review.

Modal currently lists H100 at $0.001097/second, CPU at $0.0000131/core/second and
memory at $0.00000222/GiB/second. Six physical CPU cores and 64 GiB with an H100
give an illustrative allocated rate of $4.743648/hour, so $25 permits about
5.27 hours at that rate. Actual resource usage, loading and evaluation count.
Existing synthetic training measurements suggest compact reports and a short
one- or two-epoch fit could fit; they do not prove the actual joint run's runtime.
Long-sequence multi-epoch training is not covered by this promise.

Before the full run, make representative TRAIN requests, verify the research
quality, tokenize the actual requests/targets, and measure the actual joint
training graph. Reserve worst-case token/output/timeout costs before admitting
each stage and retain uncertain charges. Stop before exceeding the cap if those
checks fail. **Spending can be capped; successful completion and improved Brier
cannot be guaranteed in advance.** No API account is configured yet.

Sources, checked 2026-11-03:

- [GPT-4.1-mini model card and prices](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
- [Batch API: 50% discount](https://developers.openai.com/api/docs/guides/batch)
- [Current API rates](https://developers.openai.com/api/docs/pricing)
- [GPT-4o-mini model card](https://developers.openai.com/api/docs/models/gpt-4o-mini)
- [Modal pricing](https://modal.com/pricing)
- Local `token_volume.md`, `design_audit.md`, and `../api_cost_scenarios_20261102.json`.
