# API cost scenarios — frozen 424-game corpus, 2026-11-02

**A scientifically valid full run is not demonstrated within the $150 cap.** With the now-measured BF16 short-run rates and implemented heldout-final omission, the retained light-teacher Modal scenario is **$276.26–$731.86**. Substituting TRAIN postmortem/gold plus independent review at Sonnet/Opus prices gives **$298.78–$723.42**. These are conditional workload allowances, not invoices or completion guarantees.

**Current ledger:** $150 total cap − $6.77 completed recorded costs − $1.82 cancelled/uncertain costs − $5 buffer = **$136.41 available**. The teacher-235B maximum $6.14 is already included, not an additional deduction. Root reports no active GPUs, no configured API account/key, and no actual NBA training/evaluation yet. Ledger values are not independently verified provider invoices. No cap increase is assumed.

This worksheet is offline: no paid calls, credentials, GPU work or account access. No provider/model is chosen and no quality approval is implied. Original budget reports are unchanged.

**Batch rates supplied as verified for this worksheet** ([OpenAI pricing](https://developers.openai.com/api/docs/pricing), [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)); no online re-verification was performed. USD per million tokens:

| Price label | Input/M | Output/M |
|---|---:|---:|
| OpenAI GPT-6.1 Sol | $1.00 | $5.00 |
| OpenAI GPT-6.1 Terra | $0.50 | $2.50 |
| OpenAI GPT-6 Luna | $0.05 | $0.25 |
| OpenAI GPT-6 Astra | $5.00 | $25.00 |
| Claude Sonnet 5.5 | $1.00 | $5.00 |
| Claude Opus 5.5 | $2.00 | $10.00 |
| Claude Haiku 5.5 (<100k input/request) | $0.05 | $0.25 |
| Claude Haiku 5.5 (>100k input/request) | $0.25 | $1.25 |

Haiku’s higher input tier costs five times the lower tier for both input and output. Four Qwen-sized gold-review proxies below 100k do not establish full-corpus provider-token tier membership. No cache discount or free credit is assumed. Bill all output/reasoning usage, not only visible JSON.

**Exact work versus projection.** Research cohort: 366 TRAIN / 21 DEV / 37 TEST = 424. Forecast cohort: 261 / 21 / 37 = 319. Exact selection chunks: 13,089 TRAIN + 729 DEV + 1,516 TEST = 15,334. Export remains 1,098 TRAIN targets and 1,272 student input rows; three SFT fits entail 6,912 three-epoch sequence passes. All sources, required examples and date folds remain intact.

The v4 failed diagnostic measured 136 initial selections (612,133 input / 152,759 output tokens), 186 selection repairs (1,243,005 / 216,170), 37 initial module/partial calls (311,007 / 89,720), seven module repairs (112,651 / 24,811), 34 first semantic reviews (373,700 / 75,135), and eight later reviews (100,789 / 22,228). These are Qwen-token observations containing known factual errors, not accepted gold and not exact API billed-token counts.

**Implemented saving:** 174 DEV/TEST prompted final generations and their 174 reviews are now omitted; reviewed partial hierarchy and final requests needed by the student remain. Root reports 25 pipeline tests for that change, within the final 318-test suite. Student generation/review and TRAIN gold targets are not omitted. The retained v4 structural projection becomes 4,808 prompted generations, rather than 4,982. The exact omitted count is known; timing/token mix is still proxy-based. Prorating the prior generation/review allowances gives a modest **$3.15–$7.95** reduction, not measured billing savings. Not all partial/final work is redundant.

**Stage token worksheet.** Tokens are millions; the common $1/$5 rate is only a pricing comparator. It is not a single-model self-review design. Selection rows are alternatives; repairs are additional.

| Stage / scope | Requests | Input M | Output M | At $1/$5 | Basis |
|---|---:|---:|---:|---:|---|
| Narrative initial selection, all 424 | 15,334 | 69.02–69.02 | 17.22–17.22 | $155.14–$155.14 | Exact chunks × v4 means |
| Reference-only selection alternative | 15,334 | 69.61–69.61 | 2.20–2.20 | $80.61–$80.61 | Offline representation proxy |
| Extra selection repairs at v4 volume | 15,334 | 140.15–140.15 | 24.37–24.37 | $262.01–$262.01 | Additional repair-token projection |
| Initial partial/direct prompted modules | 3,893 | 32.72–32.72 | 9.44–9.44 | $79.92–$79.92 | v4 structure, heldout finals omitted |
| Additional final synthesis | 915 | 10.98–21.96 | 1.37–2.75 | $17.84–$35.69 | 12k–24k input / 1.5k–3k output assumed |
| Independent prompted-module reviews | 4,808 | 52.85–52.85 | 10.62–10.62 | $105.97–$105.97 | One per remaining projected generation |
| Actual-student module reviews | 1,272 | 13.98–13.98 | 2.81–2.81 | $28.04–$28.04 | Prompted-review means are proxies |
| TRAIN postmortem | 366 | 4.18–9.47 | 0.73–1.83 | $7.84–$18.62 | Context proxies + output assumptions |
| TRAIN gold generation | 366 | 7.37–11.72 | 1.83–3.29 | $16.52–$28.19 | Context proxies + output assumptions |
| TRAIN independent gold review | 366 | 10.19–12.29 | 1.10–2.93 | $15.68–$26.93 | Context proxies + output assumptions |

The 3,893 initial and 915 synthesis requests use the v4 direct/synthesis mix to allocate the exact 174 omissions (29 direct / 145 synthesis); that mix and the remaining hierarchy are projections, not actual future-corpus counts. More synthesis levels and repairs are additional.

TRAIN output assumptions are 2k–5k postmortem, 5k–9k combined gold, and 3k–8k independent review tokens/case, including billed reasoning where applicable. Four sizing samples give postmortem input 11,423–25,875 tokens, empty-postmortem gold contexts 18,134–27,033 plus nominal 2k–5k insertion, and gold-review contexts 27,849–33,574. They combine existing erroneous partial/final outputs; no complete approved gold set exists. Canonical quote attachment, actual output shape and tokenizer differences can expand them. Several Qwen input+output/reasoning shapes exceed the old 32k helper context; API context/output/batch limits still need verification.

**Why $35 is not the workflow.** 15M input + 4M output is $35 at Sol/Sonnet rates or $70 at Opus rates. That is illustrative token arithmetic. This worksheet’s TRAIN generation-only package is 11.55–21.19M input plus 2.56–5.12M output. Adding independent gold review gives 21.74–33.48M input plus 3.66–8.05M output, before retries. Selection, module hierarchy/reviews, SFT, student decoding/reviews and forecasting still remain.

**Narrow TRAIN-only API teacher substitution.** Replace only postmortem/gold generation and independent gold review; retain every other stage. Model pairs below are price allocations, not recommendations or quality approvals. Combined amounts use measured-short-run BF16 extrapolation, the $5–15 student load/tail/repair allowance, and the implemented heldout omission.

| Teacher / independent reviewer price labels | API TRAIN package | Combined remaining-work scenario |
|---|---:|---:|
| OpenAI GPT-6 Luna / Claude Haiku 5.5 (<100k input/request) | $2.00–$3.69 | $245.05–$626.44 |
| OpenAI GPT-6.1 Sol / OpenAI GPT-6.1 Terra | $32.20–$60.28 | $275.25–$683.03 |
| Claude Sonnet 5.5 / Claude Opus 5.5 | $55.73–$100.67 | $298.78–$723.42 |
| Claude Opus 5.5 / Claude Sonnet 5.5 | $64.40–$120.56 | $307.45–$743.31 |
| OpenAI GPT-6 Astra / Claude Opus 5.5 | $153.16–$287.93 | $396.22–$910.68 |

Before API charges, retained non-teacher Modal work is $243.05–$622.75. Even deleting all student decoding cost leaves $197.66–$514.36 under the retained phase allowances. Thus cheaper TRAIN gold alone cannot fit the available $136.41. These are retained-model planning floors, not universal hardware-runtime lower bounds.

**Reference-only selection savings are conditional.** Serialization projects 152,009 old output tokens to 19,511 for all proposed references (87.2% less), or 15,696 for matched-retained sizing (89.7% less). No model generated these compact selections; recall, topics, repair rate and quality remain unmeasured. Old final-input 865,644 includes repair chat; fresh input 617,359 is not a pure saving relative to the initial 612,133.

Initial-only all-corpus selection at the common API rate projects from $155.14 narrative to $80.61 reference-only. The refreshed all-reference helper increases initial module calls **39→59** and inputs **318,084→483,919** (about 52% more). The matched-retained sizing control gives **39→33** and **318,084→258,049** (about 19% less); reproducing the rejected-summary retention filter would drop potentially useful original evidence and is not an approved shortcut. These helper baselines differ from the actual 37-request v4 wave.

An intentionally optimistic output-dominated Modal selection scenario gives about $13.15 initial-only after amortized call overhead. Prefill, repairs and shifted downstream work remain unknown. Even that selector plus free student decoding leaves about $158.64 of retained low-end work before API charges.

**Measured BF16 update.** The completed H100 probe used a six-update synthetic-trained adapter: 12/12 natural EOS outputs completed at 53.84 / 60.58 / 62.25 aggregate output tokens/s with roughly 4k–8k prefixes and ~448–469 tokens/report. Fixed-256 batch-four decoding was 59.60 tok/s, 1.87× the measured NF4 8k rate of 31.82. Adapter effect was verified (maximum common-token logprob delta 2.073); no actual NBA SFT or factual-quality result is implied.

Applying the observed short-run rate range to the assumed 1.908–3.816M report-output tokens gives **8.5–19.7 H100 hours / $40.39–$93.39 generation-only**. With an unmeasured $5–15 load/tail/repair allowance: $45.39–$108.39. Long 1.5k–3k outputs, larger prefixes, uneven tails, repairs and paired NF4/BF16 text equivalence remain unmeasured. Earlier 100–400 tok/s / $6–25 hypotheses are removed from primary scenarios.

**Broader counterfactual, not teacher-only.** Move all TRAIN selection, prompted modules and module reviews to APIs too. Keep heldout reviewed partials/request construction, all actual-student reviews, SFT and forecast/control on Modal. No newer API model is used for historical DEV/TEST prediction/review. Exact TRAIN chunk counts are used; other stage allocations and reference-only quality remain unmeasured.

| TRAIN teacher / reviewer allocation | TRAIN API research cost | Whole workflow with measured-short-rate BF16 extrapolation |
|---|---:|---:|
| OpenAI GPT-6 Luna / Claude Haiku 5.5 (<100k input/request) | $14.55–$17.13 | $135.26–$313.76 |
| OpenAI GPT-6.1 Sol / OpenAI GPT-6.1 Terra | $235.75–$281.67 | $356.45–$578.30 |
| Claude Sonnet 5.5 / Claude Opus 5.5 | $401.45–$464.23 | $522.16–$760.86 |

The cheapest unapproved allocation has a $135.26 low case, only $1.15 below the currently available $136.41, and a $313.76 high case. It leaves no credible room for qualification, tokenizer/repair uncertainty or overruns. Its tiny models have not passed source/temporal/numeric or independent-review quality gates. This is arithmetic sensitivity, not a scientifically valid funded completion plan.

**Remaining design saving is not credited:** gold-agent assessment may show that some TRAIN initial semantic reviews can be superseded by a strongest-teacher/independent-gold-review path with an explicit unverified-draft channel. No such approval exists here; preserve final gold/student quality and source/evidence context. Partial hierarchy remains necessary—do not delete all 4,808 projected generations/reviews.

**Best next spending decision within this cap:** do not launch the full corpus. After the user chooses/configures an API account/model, a separately priced TRAIN-only qualification tranche capped at at most $5 could test reference selection, gold and independent review against the existing manual evidence. This is a proposed reservation, not authorization or a promise of usable gold; it comes from the same $136.41, and would not establish funding for the full run. Exact requests must be tokenized and full per-call reservations admitted first. No further scope/cap increase is assumed.

Other blockers: no approved complete gold/export length distribution; partial hierarchy and repairs unknown; real SFT target lengths not measured (6,912 passes, provisionally $32–66); actual-student reviews remain roughly $17–43; forecast/control/evaluation $7–24 and CPU/transport $3–10. No silent source/evidence/target truncation or dropped games to meet price. Preserve OOF dates, strict cutoff/provenance, separate hindsight and checkpoint-eligible heldout models.

The JSON records detailed stage/rate arithmetic, 0.8×–1.5× tokenizer and 1×–1.5× retry sensitivities, provenance hashes and the new BF16 result. Inputs: original `remaining_modal_budget_424.{md,json}`, `expanded_cohort_work_units.json`, refreshed `reference_selection_{prototype,matched_retained}_sizing.json`, v5 gold-context proxies, the NF4 report, and `data/probes/bam_cloud/20261102_student_vllm_probe_int32/result.json`. Original reports were not edited.
