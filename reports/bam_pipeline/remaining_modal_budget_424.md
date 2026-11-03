# Revised remaining budget: frozen 424-game cohort

**The updated light-teacher workload allowance is about $305–869, using candidate
batch-four student decoding. This is a planning envelope, not a probability
interval or a commitment to finish within that amount.** A $500 cap remains a
useful staged ceiling; it is not enough to promise the complete experiment.
The 32B/72B teacher path used for these workload comparisons has not passed the
quality gate. The 235B alternative still has no completed throughput result at
the time of this revision. No new GPU work was performed for this revision.

## Verified cohort and work-unit boundaries

Frozen snapshot: `data/processed/bam_pipeline/bam_v2_20261102_expanded`.
The input checksum and source-page checksums were verified without opening labels.

| Scope | Train | Dev | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Research | 366 | 21 | 37 | 424 |
| Forecast eligible | 261 | 21 | 37 | 319 |
| Research-only | 105 | 0 | 0 | 105 |

There are 454 active source packets, 13,260 page/article-section instances and
**15,334 initial selection chunks**, counted with the actual current rule:
`len(line.text)+30`, splitting at 18,000 characters, including empty-page
dispositions. The median is 33 chunks/game, maximum 92. This count is exact for
the frozen sources; token lengths and v5 model outputs are not yet measured.

Current exporter inspection confirms `module_requests.json` contains only final
`stage='module'` requests. `export_student_inputs` requires exactly three final
modules; `export_student_examples` produces three approved gold targets. Thus
successful complete export currently means **1,098 training examples and 1,272
student input rows**. The training schema can carry optional `unit_id`, but the
current exporter emits no partial units. Recheck the actual JSONL row counts at
export; a future partial-unit exporter would require a new budget/scientific plan.

The upstream research workload is substantially larger than three outputs/game:
the v4 initial module wave had **35 partial requests and two direct final requests
across four games**. Scaling that observed structure gives 3,710 partials plus 212
direct finals for 424 games. Another 1,060 final synthesis outputs are needed to
reach all 1,272 final modules: **at least 4,982 prompted module generations before
repairs or additional synthesis levels**, with corresponding independent reviews.
These module counts are structural extrapolations, not exact final-corpus counts;
selection content determines the hierarchy. Count and reprice each actual wave.

## Measured v4 units and tokens

All measurements came from the failed diagnostic; they are throughput evidence,
not accepted research. Prompt/output means below are totals divided by responses.
Reasoning tokens are included in the 32B output counts.

| V4 wave | Requests | Input tokens total / mean | Output tokens total / mean | GPU generation batch seconds |
| --- | ---: | ---: | ---: | ---: |
| Initial 14B source selection | 136 | 612,133 / 4,501 | 152,759 / 1,123 | 297.36 |
| Two 14B selection repair waves | 186 | 1,243,005 / 6,683 | 216,170 / 1,162 | 507.69 |
| Initial 14B module/partial wave | 37 | 311,007 / 8,406 | 89,720 / 2,425 | 195.25 |
| Module repair wave | 7 | 112,651 / 16,093 | 24,811 / 3,544 | 133.89 |
| First 32B semantic-review wave | 34 | 373,700 / 10,991 | 75,135 / 2,210 | 257.51 |
| Later 32B review waves | 8 | 100,789 / 12,599 | 22,228 / 2,779 | 208.83 |

The initial module prompt median was 9,408 tokens, maximum 9,961. The first review
prompt median was 11,946, maximum 14,552. Tiny repair waves were inefficient:
one review consumed 73.44 generation seconds. Local call-to-result elapsed exceeded
batch generation by 84–103 seconds/call, including startup, queue and transport.
For planning, use about 95 seconds/call amortized across up to 256 requests; this
is not a billable-duration measurement. Avoid multiplying that overhead by games.

## Phase assumptions and allocated-resource costs

Use $4.743648 per H100 container-hour for the current six CPU cores and 64 GiB RAM;
two H200s plus twelve cores and 128 GiB cost $10.668096/hour at published rates.
Actual CPU/RAM above reservations and other billing modifiers can increase cost.
[Modal pricing](https://modal.com/pricing).

The hour ranges below are deliberately rounded allowances after startup/repair
provisions, not strict worst-case bounds. GPU-hour sums do not equal elapsed wall
time when generation runs four ways in parallel. Training stays one worker.

| Stage | Work/token basis | H100 container-hours | Approx. USD |
| --- | --- | ---: | ---: |
| 14B source selection | Exact 15,334 initial chunks; about 69M prompt / 17.2M output tokens if v4 means hold; repair volume ranges from much less than v4 to near v4 | 11–32 | $52–152 |
| 14B prompted partials/finals | At least 4,982 extrapolated units; initial partial/direct outputs about 9.5M tokens, final synthesis another 1.6–3.2M before retries/intermediate levels | 8–20 | $38–95 |
| 32B prompted-module review | Approximately one review per accepted generation plus repaired reviews; base roughly 11M output tokens at v4 review means | 11–28 | $52–133 |
| TRAIN postmortem + gold | 366 of each; scenario 2k–5k postmortem output including reasoning and 5k–9k gold output, about 2.6–5.1M total; long source prompts not benchmarked | 4–13 | $19–62 |
| 72B independent gold review | 366 combined reports; scenario 3k–8k output each, about 1.1–2.9M; assume 100–200 aggregate output tokens/s only as an unmeasured planning scenario | 3–10 | $14–47 |
| Research NF4 SFT | Exact date-fold structure below, 6,912 sequence passes, measured 8.7k–16.4k synthetic shapes plus checkpoint/load allowance | 6.7–14 | $32–66 |
| NF4 batch-four student generation | 1,272 initial final-module outputs; scenario 1.5k–3k output each, 1.9–3.8M tokens, plus bounded same-adapter repairs and uneven tails | 15–50 | $71–237 |
| 32B actual-student review | 1,272 initial reviews, about 2.8M output tokens at v4 means, plus review/student repairs | 3.5–9 | $17–43 |
| Forecast reader/control/LoRA/evaluation | 319 encodings; 261×3=783 training examples with separate prefix-Brier and explanation-CE passes; 21-game dev selection and one 37-game evaluation | 1.5–5 | $7–24 |
| CPU-only fetch/export/transport allowance | Model already cached; repeated accumulated exports and bookkeeping remain | — | $3–10 |
| **Conditional light-teacher total** | **No actual reference approvals reused** | **63.7–181 H100 hours** | **$305–869** |

For source selection specifically, actual chunk counts with the v4 initial time
give 9.31 generation hours; including all v4 repair work gives 25.21. Adding the
amortized call overhead produces roughly $52 initial-only or $138 with that
repair volume. The table allows additional variation. V5 latest-only repair and
byte-identical partial stopping should reduce waste, but their savings are not
yet measured on this corpus.

The initial module wave averaged about 460 aggregate output tokens/s; including
small repair waves lowered this to about 348. The first 32B review wave averaged
292, versus 209 including small review waves. These contextual rates motivate
the allowances; they are not universal throughput constants for gold or larger
contexts. Quality-driven branching can exceed them.

## Exact SFT and OOF partitions

| Partition | Fit games | Fit module rows | Three-epoch sequence passes | Initial student generation rows |
| --- | ---: | ---: | ---: | ---: |
| Cold fold 0 | 0 | 0 | 0 | 411 |
| Expanding fold 1 | 137 | 411 | 1,233 | 384 |
| Expanding fold 2 | 265 | 795 | 2,385 | 303 |
| Final training fit / dev-test generation | 366 | 1,098 | 3,294 | 174 |
| **Total** | — | — | **6,912** | **1,272** |

Three adapters are fitted, not four; the first partition is cold prompted.
Measured SFT step times imply about 5.58 update-hours if every sequence resembles
the 8,704-token synthetic probe, or 11.57 at 16,384 tokens. Those probes used
512-token targets; real approved gold length distributions must replace the
shape assumptions after export. No examples or token spans may be dropped to
meet the estimate.

Measured batch-four inference at 4k/8k prefixes was 42.97/31.82 aggregate tokens/s
for fixed 256-token outputs. Directly applying those rates to 1.5k–3k real report
lengths gives 12.3–33.3 hours before load, repairs, uneven tails or longer KV caches.
The 15–50-hour allowance accounts for some of those effects. **Long output and
16k-prefix performance remains unmeasured.** Current production batch two would
instead imply roughly 22.5–49.2 hours before those extras; the table assumes a new
batch-four configuration, which is a recommendation, not an applied change.
See [the completed probe](student_decode_throughput_probe.md).

A six-hour fold call can require continuation. The final SFT fit alone can take
about 5.51 hours at the maximum measured shape, then still needs 174 generations.
The existing exact checkpoints and per-module caches must resume this work. The
three-attempt ceiling is a separate limit: very slow cold-fold decoding or several
quality repairs could exhaust it without finishing, even if money remains.

## Heavy teacher and budget gates

The 235B two-H200 probe has no completed result yet. Its cost ratio for the same
work is `2.249 × heavy_elapsed / light_elapsed`. Substituting it for 32B module
reviews, postmortem/gold, and student reviews affects **18.5–50 light-H100 hours**
in this table. At equal elapsed time, that subtotal changes from roughly $88–237
to $197–533, adding about $110–296. A different measured speed changes this
linearly. This is a sensitivity calculation, not evidence that either teacher
passes quality or that the stronger model has equal speed.

The dominant uncertainties are: final source/partial hierarchy depth; independent
review reliability and resulting repairs; 235B startup/long-context throughput;
real gold target lengths; and long, uneven NF4 student generations. Full-corpus
quality failures must stop work, not be counted as inexpensive completion.

Use the explicit-cap `BudgetLedger` and durable attempt keys, reserving full
timeout cost plus 15%, retaining the $5 buffer. Reconcile completed estimates,
retain cancellation/unknown costs until evidence resolves them, and admit every
retry separately. Re-estimate after the first source tranche, actual module waves,
gold export tokenization, and the first cold student partition. The ledger is an
admission guard, not provider billing verification or permission to raise a cap.

Underlying work units and arithmetic are in `expanded_cohort_work_units.json` and
`remaining_modal_budget_424.json`. The latter records the frozen snapshot hash,
exact fold counts, exporter scope and all phase allowances.
