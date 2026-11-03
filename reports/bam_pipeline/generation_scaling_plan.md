# Generation scaling and exact-response reuse

The practical first change is four concurrent light-generation calls, retaining
one training call. No GPU jobs were started for this audit. The helper is
`scripts/bam_generation_dispatch.py`; eight focused tests and an independent
review passed. Production integration belongs to the root controller.

## Integration contract

```python
cloud._connect()  # Initialize SDK handles before threads.
jobs = [{"args": original_generate_arguments, "identity": original_call_identity}, ...]
results = dispatch_batches(jobs, cloud.invoke, max_inflight=4)
```

The original batch arguments, order, run IDs and durable identities remain
unchanged. Results return in input order even if calls finish out of order. On
an exception, no further jobs are submitted after the coordinator observes it;
already running calls are drained so DurableCloud saves their results.
`DispatchFailed` exposes indexed `results`, `failures` and `not_started`.
Restart with identical jobs: completed results are checked and reused, recorded
calls are reattached, and only never-started jobs are spawned. Partial successful
responses inside a failed semantic workflow remain raw evidence, not approval.

At inspection, `bam_cloud.generate` still had `max_containers=1`; raising that
cap to four is necessary or parallel client calls merely queue. Leave training
at one. Every generation batch already has a distinct hashed cloud run ID and
directory, which keeps concurrent writes separate. A heavy-review function can
have its own max-one container limit; route its calls serially through that
function. A two-H200 review call is two physical GPUs, additional to the four
light H100s if both pools run concurrently. Record that resource policy explicitly.

Keep one controller per experiment. The existing durable client has a narrow
spawn-to-call-record crash window: if the remote spawn succeeds and the controller
dies before writing `call.json`, restart cannot discover that call automatically.
Do not claim service-level exactly-once execution. A small hardening is a durable
spawn-intent record and a fail-closed/manual reconnect when intent exists without
a call ID. Normal reconnect after a recorded call ID is covered by the new test.

## Batching policy

Use the same model within each raw generation dispatch. Keep the existing
hierarchical barriers: source selection, module generation, independent review,
then repairs/synthesis. Begin with `batch_games=64`, `request_batch_size=256`, and
the existing worker `request_chunk_size=64`, `max_num_seqs=12`. These are proposed
execution settings for a new frozen run, not changes made by this helper.

The current default of 512 requests per remote call often leaves fewer than four
jobs in a module/review wave. A 256 limit gives more opportunities to fill four
slots while amortizing model load over several internal chunks. For small waves,
accept fewer busy slots instead of launching tiny calls just to fill GPUs. Each
remote call currently starts a new subprocess and reloads its model; replacing
that with resident serving is a larger change and unnecessary for the first
scaling improvement. Do not run entire game-batch controllers concurrently:
they mutate shared stage state and are not covered by this helper's safety model.

## Measured reference workload, not a full-run forecast

The failed v4 four-game diagnostic retained these generation totals. Batch seconds
are summed as `response.batch_seconds / response.batch_size` so a batch is counted
once, not once per response. They exclude startup, checkpoint file hashing, model
load, logging/transport, and idle time between dependency stages.

| Work completed before the quality/context failure | Calls | Responses | Generation batch seconds |
| --- | ---: | ---: | ---: |
| 14B selection, initial pass | 1 | 136 | 297.361 |
| 14B selection, two repair waves | 2 | 186 | 507.693 |
| 14B module/partial generation | 2 | 44 | 329.134 |
| 32B independent semantic reviews | 3 | 42 | 466.334 |

At the same four-game source workload per game, 371 games would imply 20.74
single-H100 generation hours for selection including those retries, or an ideal
5.19 hours across four GPUs. Initial selection alone gives 7.66 single-GPU hours,
or an ideal 1.92 hours across four. These are rough workload extrapolations from
four nonrandom reference games, not runtime promises. Notes vary substantially;
the module/review workflow did not finish and no gold or training phase ran.
Stopping byte-identical partial repairs removes unnecessary repetition without
approving rejected facts; that fix is owned by the root driver.

## Exact cache, deliberately narrow

```python
cache = ExactResponseCache(cache_directory)
cache.import_batch(
    original_requests, prior_result_path, prior_result_manifest_path,
    origin={"run_id": origin_run, "call_id": origin_call},
    expected_identity=trusted_current_full_runtime_identity,
    prompt_token_hashes={request_id: current_prompt_token_hash, ...},
)
hit = cache.lookup(
    new_request, expected_identity=trusted_current_full_runtime_identity,
    prompt_token_sha256=current_prompt_token_hash,
)
# None, or {"response": unchanged_raw_response, "reuse": explicit_origin_record}
```

Import verifies the actual prior result bytes against their durable result
manifest, every request/source hash, token-ID hash, complete model-file/runtime
identity and response hash. Copied origin evidence has its own content hash.
Lookup requires exact full request equality, including source/audit metadata and
budgets, and exact model revision/file hashes, worker/helper hashes, effective
sampling/settings, completion policy and library versions. The caller must
independently establish the expected runtime identity; copying an untrusted old
identity is not verification. No guessed runtime compatibility is supported.

Only `status=complete` plus `finish_reason=stop` responses are indexed. Responses
are not rewritten or rebound. Re-ingest their raw contents under the current
mechanical validators; run independent review again when review requests change.
Never copy old approved artifacts, completion markers or quality gates.

No actual response reuse has been performed. The v5 base source instructions
changed, so all v4-to-v5 exact request cache hits are expected to be zero. Expanded
snapshot source/audit metadata also differs. The root deliberately is not wiring
a zero-benefit import path into this run; the tested helper remains available for
future genuinely identical requests.

## Remaining bottlenecks and restart constraints

- Every inference call hashes all pinned checkpoint/tokenizer files, then starts
  a fresh model subprocess. More small calls increase this fixed cost. Keep the
  integrity checks; measure their elapsed time separately before changing them.
- Local prompt tokenization is repeated by preparation, save/preflight and raw
  response validation. A bounded in-process cache keyed by tokenizer-file identity
  plus exact messages can remove CPU duplication later without relaxing checks.
- Each 64-game batch waits for its entire review/repair hierarchy. This limits
  cross-batch utilization but keeps shared pipeline mutations sequential. Four
  independent raw batches are the low-risk first concurrency boundary.
- Student OOF generation uses Transformers NF4/adapter generation with batch size
  two, left padding and a fixed batch waiting for its longest output. Four light
  vLLM workers do not accelerate this path. Do not silently replace it with a
  different backend; adapter-serving migration needs separate parity/provenance
  validation. Actual output-token throughput remains unmeasured.
- Research fitting and generation share a fold call. Periodic exact optimizer
  checkpoints and per-module generation caches preserve progress, but generation
  contributes to the six-hour call budget. Graceful deadline failures resume the
  same immutable fold; completed siblings are retained. Do not parallelize folds
  in the same cloud run: they update a shared generated-modules aggregate.
- The 371-game preview contains 320 train, 17 dev and 34 test games. With three
  modules per accepted training game and three epochs, chronological fit pools
  would contain 0, 363, 690 and 960 examples, totaling 6,039 sequence passes.
  At the measured synthetic 8,704-token shape this is 4.87 hours of fit updates
  across folds; at 16,384 tokens, 10.11 hours. The final fit alone is 2.32 or 4.82
  hours respectively, before generation/checkpoints/load. These are shape-based
  projections with a 512-token synthetic target, not real gold-length estimates.
- Every fold export currently walks/checksums and archives the accumulated run,
  excluding adapter and optimizer tensors by default. Repeated full exports grow
  with earlier-fold artifacts; preserve immutable exports, but incremental
  manifests/selected-file retrieval is a possible later CPU/transport improvement.

## Isolated stronger-teacher probe review

Read-only diff confirms `bam_teacher_probe_worker.py` differs from the frozen
inference worker only in its description and `tensor_parallel_size=2`. The
separate cloud wrapper requires the exact pinned teacher spec, 1–40 TRAIN-only
requests, two H200s, max one container, zero idle containers, a 1,800-second limit
and a 1,740-second child watchdog. Requests and results remain immutable and the
volume commits in `finally`. Existing full-token/context checks, raw reasoning
retention, complete-answer checks and per-request provenance are preserved.

No code blocker found. FP8 is taken from the pinned checkpoint configuration;
the worker's BF16 setting controls the activation dtype. Download-independent
weight hashing and model loading still count toward the diagnostic timeout, so
startup failures must be distinguished from model-quality results. No claim of
quality improvement is made before that bounded comparison completes.
