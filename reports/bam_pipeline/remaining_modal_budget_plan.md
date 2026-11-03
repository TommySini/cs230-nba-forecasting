# Remaining Modal execution budget

**Superseded for current planning by [the frozen 424-game budget](remaining_modal_budget_424.md),
which uses exact source-chunk counts, separate partial/final generation units,
exact date-fold counts and measured candidate batch-four decoding.**

**Subsequent measurement:** the candidate cohort is now 424 games. The isolated
student decode probe measured only 21.6–23.5 aggregate tokens/s at the current
batch-two setting, and 31.8–43.0 at candidate batch four. The original $40–180
student-decoding allowance below is too optimistic if batch two is retained.
See [the measured probe and updated conditional costs](student_decode_throughput_probe.md).
The table below remains the original planning record, not an updated total quote.

**A $500 remaining cap is a reasonable staged spending ceiling, not a promise
that the full experiment will finish. $150 supports bounded partial work; $300
could finish only a favorable lighter-teacher path.** No cap response has been
received at the time of this estimate. This report authorizes no compute and
does not change model roles, training parameters or experiment gates.

The latest immutable snapshot is 371 games (320 train / 17 dev / 34 test; 300
forecast eligible). The data agent's latest approved union is 405 games (354 /
17 / 34), not yet frozen. Additional source validation may change those counts.
Plan below uses approximately 405 games. No outcome values were read.

## Published rates and actual worker allocations

Modal's published rates, checked for this report, give the following arithmetic.
These are regular Function rates without regional or non-preemptible premiums.
Load time is billable. CPU/RAM billing uses the larger of requested and actual
usage, so reservation-based totals are not hard invoice bounds.
[Modal pricing](https://modal.com/pricing),
[resource units and billing](https://modal.com/docs/guide/resources).

| Resource | Published hourly rate |
| --- | ---: |
| H100 SXM5 | $3.949200 / GPU |
| H200 SXM | $4.539600 / GPU |
| Physical CPU core | $0.047160 |
| RAM GiB | $0.007992 |
| Current light generation/training: 1 H100 + 6 cores + 64 GiB | **$4.743648 / container-hour** |
| Current heavy probe: 2 H200 + 12 cores + 128 GiB | **$10.668096 / container-hour** |
| Model fetch: 4 cores + 16 GiB | $0.316512 / container-hour |

Four concurrent light workers consume approximately $18.97/hour while all are
active. They reduce elapsed time, not total GPU work. No credits are assumed;
these are gross incremental usage estimates. Existing monthly usage, remaining
credits, excess storage/egress and other apps must be reconciled separately.

## What was measured

All timings below come from the retained failed v4 four-game reference, before
gold or NBA training. Its research was not quality approved. Batch seconds count
each GPU generation batch once, dividing repeated per-response durations by batch
size. Local elapsed uses `call.json` to `result.json` filesystem timestamps; it
includes queuing/transport and is a conservative elapsed proxy, **not a Modal
billing record**. Copying or touching files can invalidate that proxy.

| Reference phase | Calls / responses | Generation seconds | Local elapsed seconds |
| --- | ---: | ---: | ---: |
| 14B selection, initial | 1 / 136 | 297.36 | 400.72 |
| 14B selection, two repair waves | 2 / 186 | 507.69 | 696.91 |
| 14B modules/partials, incomplete hierarchy | 2 / 44 | 329.13 | 521.25 |
| 32B semantic review, incomplete hierarchy | 3 / 42 | 466.33 | 723.35 |

The excess over generation is 84–103 seconds per call. Larger batches amortize
this; allocating that overhead once per game would overestimate full-run cost.
Conversely, using generation seconds alone omits real startup work. Model hashes,
checkpoint load and short repair waves remain material.

Selection alone, scaled from four to 405 games, gives approximately 8.36 H100
generation hours for an initial pass or 22.64 hours including the v4 repairs.
At 256 requests/call, adding roughly 95 seconds per call gives approximately
$46 initial-only or $123 with v4 repair volume. A $150 selection allowance permits
some changed prompt/coverage overhead; it is not a worst-case guarantee.
Byte-identical partial-retry stopping should reduce waste, but it has not been
measured on the final corpus. The four games are not a random sample of note sizes.

V5 source instructions and expanded snapshot metadata changed. Exact v4 response
cache hits are expected to be zero. No old approval or stage completion is reused.

## Phase allowances for the lighter teacher path

The following are **planning allowances, not a statistical interval or a measured
full-run forecast**. They assume the 32B teacher and 72B independent gold reviewer
could pass the new quality gate; that has not been established. The reference
found false approvals, so their lower prices cannot justify bypassing quality.

| Remaining phase, approximately 405 games | USD allowance | Basis / uncertainty |
| --- | ---: | --- |
| 14B source selection | $46–150 | Measured initial and repaired source workload, startup included |
| 14B prompted modules and synthesis | $30–75 | Measured partial hierarchy, allowance for remaining synthesis and repair |
| 32B prompted-module semantic review | $50–110 | Roughly 10–13 reports/reviews per game including partials; observed large-wave timing plus repair margin |
| TRAIN postmortem and gold generation | $15–50 | About 354 postmortems and 354 combined gold reports; not yet timed |
| 72B independent gold review | $10–40 | About 354 reports, potentially repaired; not yet timed on this workload |
| Research NF4 SFT across fitted folds | $30–70 | Measured 8.7k–16.4k sequence shapes, three epochs, checkpoints/load allowance |
| OOF/final student module generation | $40–180 | About 1,215 reports; Transformers NF4 decoding **unmeasured** |
| Independent actual-student review | $15–45 | 1,215 module reviews plus bounded same-adapter repairs; no gold substitution |
| Frozen reader/control, forecast LoRA and evaluation | $10–30 | Original fixed three epochs, separate Brier/CE passes, dev selection and one test evaluation |
| CPU fetch/export/transport allowance | $3–10 | CPU-only operations and repeated exports; not a storage/egress invoice estimate |
| **Sum of allowances** | **$249–760** | Neither a completion guarantee nor a predicted mean |

The module generation sample had 44 outputs, median 2,693.5 tokens and range
450–6,037. The student may produce different lengths. To make the decoding
allowance explicit: 1,215 outputs of 1,500–3,000 tokens at an assumed aggregate
30–60 tokens/second would take about 8.4–33.8 H100 hours ($40–160), before retries
and loading. **Those decoding rates are scenario assumptions, not measurements.**
Reprice this line from the first completed cold fold before committing later folds.
If effective decoding throughput is lower, $180 can be exceeded without a bug.

There are four chronological **generation partitions**: cold fold, two expanding
fits, final-fit heldout generation. Only three adapters are fitted; the cold fold
does not train. The 371-game snapshot gives 6,039 three-epoch sequence passes.
Scaling its date-distribution proportions to 354 training games gives roughly
6,681 passes: about 5.4 hours at the measured 8,704-token synthetic shape or 11.2
hours at 16,384 tokens, before overhead. Final fit alone has 3,186 passes, about
2.57 or 5.33 hours. Targets in these probes had 512 tokens; real gold targets vary.

The final six-hour training call may need a continuation because it also generates
153 dev/test module reports. That is acceptable only through the existing exact
checkpoint/completed-adapter and per-module caches. Repeated calls must not refit
a completed adapter or regenerate successful siblings. Keep training concurrency
one: fold aggregate files share a run directory.

## Stronger 235B teacher: wait for its measured throughput

The two-H200 235B probe was still pending when this estimate was assembled. There
is no defensible full-corpus runtime estimate yet. For the same completed work,
its cost relative to one H100 is:

`2.249 × (235B elapsed seconds / lighter-teacher elapsed seconds)`.

Half the elapsed time means approximately 1.12 times the cost; equal time means
2.25 times; twice the time means 4.50 times. This is rate arithmetic, not a claim
about model speed. Prompted and student review plus postmortem/gold generation
account for roughly $80–205 of the lighter-path allowance. At equal elapsed time,
moving all those roles to the heavy teacher adds about $100–256. Moving only
review roles has a different, smaller footprint. Do not extrapolate a 16-claim
probe as though it already measured multi-claim long-context reports or gold.

A single 30-minute heavy probe reserves about $5.33 at current allocations, plus
buffer. Its quality result and startup-versus-generation timing should determine
the next bounded tranche; model size alone is not evidence of better review.

## Meaning of a $150 / $300 / $500 remaining cap

| Remaining gross-usage cap | Practical commitment |
| --- | --- |
| $150 | Bounded quality/reference work and a source-stage tranche. Do not commit to full-corpus training. Preserve partial artifacts and stop at the cap. |
| $300 | A favorable lighter-teacher full path is conceivable, but the empirical support is insufficient to promise it. Re-estimate after source/module quality and the first student decode partition. |
| $500 | Reasonable staged ceiling for a serious full attempt. Still not a guarantee, especially with a heavy teacher or slow NF4 generation. Pause before new calls when the reserved remainder will not fit. |

If the accepted cap cannot cover the declared pipeline, preserve the completed
stage and report the missing work. Do not quietly drop games, epochs, reviews,
source pages or evaluation controls to make a smaller budget appear complete.

## Minimal ledger and stop rules

No new task framework is needed. Add one run-owned `budget_ledger.json` keyed by
the existing durable call key, with cap basis, rates, function/model resources,
call ID, submission time, observed worker start/finish, estimated completed cost,
outstanding reservation and status. Keep an append-only event file if desired.
Reconcile this estimate with actual Modal usage, which remains authoritative.

1. Atomically reserve before dispatch; a reconnect to the same call consumes the
   same reservation. A failed new attempt is another charge, so record it separately.
   Completed cached results consume no new reservation. Never count concurrent
   controller wall time once when several containers are active.
2. Use worker elapsed timestamps where present; add startup allowance. Inference
   currently has batch timings but no full-function timestamps in its result, so
   use persisted call-to-result elapsed as a conservative proxy until a future
   wrapper records actual worker start/finish. Do not treat summed batch seconds
   as a billable-duration total.
3. Reserve the full declared timeout at the worker's allocated hourly rate before
   each new call: two-hour light call $9.49, six-hour training call $28.46,
   30-minute two-H200 call $5.33. Add a 15% operating buffer for startup/resource
   variance. Four light calls reserve approximately $43.64 with that buffer.
   This can underfill concurrency near the cap; that is preferable to overspending.
4. Admit only when `completed_estimate + outstanding_reservations + next_reservation`
   remains below the authorized remaining cap minus a small export/recovery reserve
   (for example $5). Check every spawn/retry, not only at stage boundaries.
   Stop submitting new calls on exhaustion; let already reserved calls finish and
   persist results. Never auto-increase the cap or launch a replacement unreserved.
5. Re-estimate after the new quality reference, after the first 32–64-game source
   tranche, after gold export yields real token lengths, and after cold-fold decode.
   Stop on quality failures regardless of available money. Preserve test-label
   isolation and selection locks across all pauses.

The ledger is an admission guard, not a hard invoice guarantee: actual CPU/RAM,
other workloads and startup can differ. Modal's Workspace usage budget is a
monthly gross-usage cap before credits; its spend limit instead caps net charges.
Environment budgets require Team/Enterprise. To enforce a remaining-dollar limit
at the platform, account for usage already incurred and other active apps rather
than setting the month's limit directly to the remaining amount.
[Modal budgets documentation](https://modal.com/docs/guide/budgets).

No budget settings, code, deployments, downloads or GPU work were changed by this
audit. Machine-readable rates, retained call IDs/timings and allowance arithmetic
are in `remaining_modal_budget_plan.json` beside this report.
