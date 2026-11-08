# A $100 lean BAM-style experiment: design audit

2026-11-03. Offline feasibility review only. No paid requests, cloud jobs, training,
account setup, dataset import or production configuration changes were performed.
API prices and model availability are outside this audit; use the companion pricing
assessment before any spending decision.

**A smaller BAM-style design is scientifically coherent, but completion of the
expanded full-note experiment for $100 has not been demonstrated.** Freeze one
prompted researcher, obtain hindsight-guided gold for 200–300 TRAIN games, and
train one existing Qwen2.5-14B LoRA plus probability head. This removes researcher
distillation, its chronological out-of-fold fits, and adapted-researcher decoding.
It retains the main question: whether a jointly trained forecaster using pregame
research improves on the same statistical forecast. It does not reproduce the
full BAM research-training workflow or establish that cheaper research is good.

The retained original 424-game workflow already had a **$276.26–$731.86** conditional
allowance; merely replacing its teacher with a cheap API does not make the new
1,980-game workflow fit $100. Savings here require the explicit redesign below.

## Scope and evidence status

The expanded cohort remains provisional: **1,632 TRAIN / 201 DEV / 147 TEST**
research games, with **1,506 / 190 / 140** potential forecast games. TRAIN ends in
2023–24; DEV is 2024–25; TEST is 2025–26. Counts can change after source
reconstruction, readable-page checks, header validation and statistical-history
verification. Research-only games must not quietly enter outcome training, nor
should a budget estimate silently drop them from a stated 1,980-game research run.

Preserve the old and new source union, exclude known post-cutoff revisions unless
an earlier version is recovered, and bind selected text to exact source bytes.
Keep archive-proven and publisher-timestamped source tiers explicit. The expansion
is selected team/date coverage, with source-distribution shift between TRAIN and
evaluation. It is not league-wide narrative coverage. Historical TEST also
overlaps previously inspected/evaluated games; this remains an exploratory
backtest, not a fresh final benchmark.

## Proposed minimum design

1. **Freeze full-note research.** Use one fixed, pinned, knowledge-cutoff-eligible
   researcher throughout TRAIN/DEV/TEST. Every model doing held-out selection,
   synthesis, semantic review or repair must qualify before the first DEV game,
   October 23, 2024, under the agreed documented-cutoff criterion. A modern
   TRAIN-only teacher may not become a hidden held-out reviewer. Freeze the
   research prompt and acceptance policy using TRAIN examples; save accepted
   research before hindsight gold is generated.
2. **Preserve complete source coverage.** Enumerate every source page/section and
   chunk, with input hashes and a processing disposition. Retain evidence
   references, opposing evidence, decoded numeric tables and explicit unreadable
   pages/undecoded tables. “Once per game” means one cached research workflow;
   it means one request only when the entire input and output reservation fit the
   qualified model's usable context. Larger packets need fully accounted chunks
   and synthesis. Full notes may be summarized; silently dropping pages, source
   text or unfinished outputs is not a valid cost saving.
3. **Make 200–300 TRAIN gold targets.** Select from the forecast-eligible TRAIN
   games before inspecting outcomes, stratified by season, publisher/source tier,
   team coverage, pregame probability and packet-length bins. Include difficult
   long/numeric packets rather than only cheap cases. A 250-game sample is 16.6%
   of the potential 1,506-game forecast TRAIN set. A stronger modern teacher can
   see TRAIN postmortems/recaps in a separate target-only store. Its gold must
   express an ideal as-of-cutoff explanation, grounded entirely in pregame
   sources; hindsight guides attention, never supplies forecast-input facts.
   Independently review each accepted gold against the pregame evidence and
   preserve rejection/repair history. Sample size counts accepted targets.
4. **Fit one Qwen2.5-14B forecaster.** Retain the pinned implementation, rank-8
   LoRA and zero-initialized residual log-odds head. Use outcome Brier on every
   eligible TRAIN game and completion-only explanation loss on the gold subset.
   The probability head reads only the pregame prefix; gold is used in its
   separate teacher-forced training pass. No researcher SFT, adapted-researcher
   generation, new model integration or multi-model search is included.
5. **Keep the comparison matched.** Reconstruct prior-only boosting probabilities
   for the extended history using the original selected hyperparameters. The old
   final booster included 2024–25 in fitting and cannot supply honest inputs for
   the proposed DEV season. Give both arms identical booster probabilities and
   held-out membership. Predeclare epochs and selection on DEV, retain the zero
   correction candidate, save TEST predictions before scoring, and report paired
   date-bootstrap Brier differences. No TEST-driven rerun or tuning is included.

The sparse objective needs explicit normalization. One defensible specification
is mean Brier over all N TRAIN games plus a fixed weight times mean per-example
completion loss over the M gold games. Simply masking missing gold in the current
all-game loop reduces the effective explanation weight by M/N; that is a different
objective unless explicitly intended. Declare the normalization and weight before
DEV selection. Avoid post-hoc oversampling of wins, errors or appealing gold.

The frozen researcher has no target-game gold fit, so removing its OOF training
does not introduce the researcher target-leakage problem that those folds solve.
This does not waive source timestamps, model eligibility or booster prior-only
training. Freeze and record the research model's empty task-training set.

## What remains unimplemented or unqualified

The existing `_forecast_steps` in `scripts/bam_training.py` requires an accepted
explanation for **every** TRAIN label. Sparse explanations require a deliberate
change to masking, loss normalization, export/checkpoint identities and validation;
the proposed run cannot be obtained merely by supplying a smaller gold file.
Existing research pipeline assumptions about adapted reports/three modules also
need a truthful frozen-research import path. Do not fabricate adapter provenance.

The current forecast entry point requires frozen-reader embeddings and fits its
diagnostic control before the LoRA fit. Reusing that path adds an encoding pass
whose cost must be counted; making the control optional is a separate small
implementation change. No such changes are made by this audit.

Qualification must check full-source recall and actual claims on representative
TRAIN cases, including player identity, current versus historical injuries,
table columns/units and sample time windows. Exact citations and schema validity
alone failed to establish correctness in earlier probes. Budget independent
semantic checks and bounded repairs, not just the first researcher response.
Replacing all-report semantic review with a sample audit is an explicit decrease
in quality assurance; it cannot inherit the old “approved research” guarantee.

Gold/report lengths are not measured for the new cohort. The old gold proxy was
5k–9k output tokens, so some prefix-plus-gold sequences may exceed the training
limit. Specify a complete, useful explanation format, size it with the exact
tokenizer, and reject overflow before spending. Neither source nor target
truncation is an approved fallback. No accepted NBA SFT set or fitted model-quality
result exists yet.

## Is six H100 hours a defensible allowance?

The actual pinned NF4/LoRA **research SFT** probe measured 1.5113 / 2.9041 / 4.4486 /
6.0250 seconds per update at 4,608 / 8,704 / 12,800 / 16,384 total tokens. It used
synthetic prefixes, 512-token targets, microbatch one and an optimizer update
every example. Those are shape measurements, not the real joint objective.

The forecaster runs a separate prefix Brier forward/backward on all games, plus
a completion CE forward/backward on gold games. For 1,506 TRAIN and 250 gold,
one epoch therefore has 1,506 prefix passes and 250 completion passes. Treating
each as one same-shaped probe pass gives the following **illustrative arithmetic**:

| Probe-shaped pass assumption | One epoch | Two epochs | Three epochs |
| --- | ---: | ---: | ---: |
| 4,608 tokens | 0.74 h | 1.47 h | 2.21 h |
| 8,704 tokens | 1.42 h | 2.83 h | 4.25 h |
| 12,800 tokens | 2.17 h | 4.34 h | 6.51 h |
| 16,384 tokens | 2.94 h | 5.88 h | 8.82 h |

These are neither bounds nor measured forecaster runtimes. Prefix-only Brier
avoids the LM output head, actual gold can be much longer than 512 tokens, and
accumulation differs. Loading, validation, frozen-reader encoding, checkpoint
writes, recovery, final prediction and any explanation generation are extra.
At 200–300 gold games, two maximum-shape epochs span approximately **5.71–6.05 h
before overhead**. A six-hour reservation cannot credibly promise that case.

Six hours is a reasonable **spending ceiling for a qualified one- or two-epoch
run** when actual lengths and a short joint-graph probe establish sufficient
headroom. It is not evidence that three epochs fit or that one epoch is adequate.
Choose the intended epoch count from measured runtime before the full fit; keep
checkpoint/resume semantics and reserve any continuation within the same cap.
If admission fails, stop rather than drop long games or silently change the
training protocol. A partial checkpoint is not a completed planned experiment.

## Budget boundary

At the previously recorded allocated-resource rate of $4.743648 per H100 hour,
six hours reserves **$28.46**. This report does not refresh that price. Keep
qualification, data/CPU/transport, model loading, encoding, DEV/TEST inference,
storage and retries inside the total. Do not count free credits or cache hits
until actually available and applicable.

| Meaning of the $100 ceiling | Room after six-hour fit and a $10 contingency |
| --- | ---: |
| $100 additional future spending | $61.54 for all remaining stages |
| $100 inclusive of known $6.77 completed + $1.82 uncertain old-cap ledger | $52.95 for all remaining stages |

Earlier project bills are unknown, so neither interpretation proves total
historical project spending was at most $100. The old ledger is an estimate,
not a provider invoice. Keep uncertain charges reserved until reconciled.

The remaining amount must cover **all** frozen full-note research, its review and
repair, 200–300 accepted gold targets with postmortems and independent review,
qualification and final inference/CPU work. It is a finite allowance, not a cost
estimate for those stages. Obtain exact provider-token request sizes and bounded
output/retry reservations before deciding whether they fit. The old 424-game
selection worksheet alone had 69.02M input tokens, including repeated prompts;
“one call per game” is no proof of small inputs. Do not scale that proxy into an
allegedly exact new-corpus bill.

A hard admission ledger can enforce spending at most $100 even when the work
does not finish. Promise the cap; promise completion only after source/token
sizes, researcher quality and joint-training throughput support it.

## Scientific tradeoffs

This design preserves full-source access, hindsight-guided supervision, joint
forecaster/head training, all eligible outcome labels and a matched boosting
comparison. It sacrifices learned researcher adaptation, gold for most games,
multiple model candidates, broad hyperparameter/seed searches and potentially
some training epochs. A small gold subset may underrepresent rare source styles
or basketball situations; one Qwen fit can fail to learn useful corrections.
Neither lower cost nor added game count establishes improved Brier.

With one trained text model versus boosting, any gain is a gain of the combined
system. It does not isolate the contribution of textual information from extra
calibration/modeling capacity as well as a matched trained no-text ablation would.
The original booster should remain the primary fixed baseline; disclose the
missing ablation rather than claim a causal text contribution. Source/season
breakdowns and paired uncertainty remain useful, but small strata are descriptive.
No results from this proposal should be presented as completed training findings.

Sources: `HANDOFF.md`, `README.md`, `docs/experiments.md`,
`reports/bam_pipeline/api_cost_scenarios_20261102.md`,
`reports/bam_pipeline/claude_expansion_review_20261103/{recommendation,data_audit,provenance_review}.md`,
`reports/bam_pipeline/training_throughput_probe.md`,
`reports/bam_pipeline/remaining_modal_budget_424.md`, and
`scripts/bam_training.py` (prediction validation, token overflow guards,
forecast entry point and joint optimizer).
