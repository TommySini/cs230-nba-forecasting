# Offline TRAIN-only API teacher boundary review

An API teacher can replace the training-target postmortem/gold backend without entering heldout research inference. It must be a separate role from the fixed historical model that reviews actual student reports. No API calls, keys, installations or production changes were made for this review.

The current `Pipeline.teacher` has several jobs: it reviews initial modules for all splits, generates TRAIN postmortems/gold, and reviews actual student reports before forecasting. Simply replacing that object with a modern API model would send heldout research through the API and would conflict with the current model-date gate. The roles need to be separated explicitly.

## Minimal initial boundary

Keep the local pinned student and historical research critic unchanged for every prediction-input pathway. Add an optional `training_target_backend` that can receive only:

- `postmortem` requests for game IDs in the immutable TRAIN manifest, using the existing normalized, separately stored postgame records.
- `gold` requests for those same TRAIN IDs, combining the three real researcher outputs with the separate TRAIN postmortem exactly as the current builder does.
- `semantic_review` requests whose target artifact is a TRAIN gold artifact, bound by its exact target hash. This review receives the full pregame evidence context and candidate, never the training-only postmortem.

The adapter must verify game membership from the trusted snapshot, not merely a request's `split` string. It should reject selection, heldout module generation/review, student prediction-report review, forecast and evaluation requests. A later choice to use API review on preliminary TRAIN modules needs its own explicit training-target purpose and protocol flag; it should not silently alter the reviewer of actual forecasting inputs.

The local research student remains shared across the three module calls. API gold is a target, never a replacement for a generated student report. Existing chronological fold exclusions, same-game training exclusions, development selection and locked test evaluation remain in force.

## Configuration and artifact identity

A backend specification needs provider, exact requested model/snapshot name, returned model identity, allowed roles/splits, context/output budgets, request timeout/retry bounds, and the response-format policy. Use two independent generator/reviewer identities; the current code's independent-review check must remain. API credentials belong in the chosen account's secret/environment mechanism and must never enter configs, requests, artifacts or logs.

Record API teacher release/identity separately from the prediction-time checkpoint dates. `effective_minimum_eval_date` should continue to bind every model used on heldout prediction inputs. A modern TRAIN-only teacher is a retrospective distillation dependency: describe it explicitly. Source/date/citation checks reduce leakage risk but cannot prove that a newer teacher's pretrained knowledge has no influence on its choice of TRAIN explanations. Do not call the entire experiment a historically deployable pre-test-checkpoint system without that qualification.

Keep current backend-generic request JSON and parent/source/input hashes. Add provider request/batch IDs and returned usage/model metadata beside immutable raw responses. After retrieval, use the same source/citation/coverage validators, exact finding IDs, independent semantic verdict checks and repair machinery. Reusing a cached API response requires exact request/model/config identity; a provider model alias alone is insufficient evidence of a fixed checkpoint. Preserve malformed/truncated/error outputs and retry lineage. Never regenerate a completed request merely because its network retrieval was interrupted.

## Context and spending

The now-compacted postmortem requests fit the current four-game 32k budget. Existing real-report sizing proxies need approximately34k–43k for gold and40k–45k for gold review, so select a context budget that fits the complete request plus completion/reasoning allowance. Final gold output allowance is8192tokens. No source truncation or independent per-module hindsight report should replace the combined gold task.

Use the user's single remaining$150 cap across Modal and API. API admission should reserve explicit token-cost estimates (actual input count plus bounded output/reasoning, pinned pricing and margin) before submission and reconcile returned usage afterward. The current GPU ledger uses elapsed-time allocation estimates; do not disguise an API token estimate as an hourly GPU rate. Outstanding batch requests, ambiguous submission failures and retries must retain reservations until their state is resolved. The earlier illustrative dollar estimate is not an authorization to spend beyond the combined cap or an invoice guarantee.

## Required boundary tests before integration

1. A dev/test game rejected even when the request falsely claims TRAIN; no provider call occurs.
2. A TRAIN gold review rejected if its target hash/game/source differs, and serialized review messages contain no postmortem or label fields.
3. Unknown task/backend fields fail closed; current normalized-source and output schemas remain allowlists.
4. Every original researcher module and every postmortem observation survives the combined gold prompt; canonical source records and raw responses stay unchanged.
5. API teacher targets never enter prediction inputs, while student generation provenance and fold exclusions remain exact.
6. Interrupted submission/retrieval reconnects to the same provider request; duplicate billed retries and over-budget calls are prevented or explicitly recorded as uncertain.
7. A modern training-target teacher does not disable checkpoint-date checks for heldout student/reviewer models.

The first quality decision should use the same four complete TRAIN report audits and source expectations already prepared. Neither a provider name nor a larger model replaces source-level review. The current235B isolated probe remains a separate fixed-input diagnostic; no quality result was available when this review was written.
