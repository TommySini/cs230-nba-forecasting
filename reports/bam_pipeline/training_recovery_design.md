# Proposed bounded training recovery

The helper is integrated into research SFT and forecast adapter/head training.
Twelve helper tests and actual CPU-loop interruption tests cover adapter/head,
AdamW, shuffle/dropout RNG, dev-label identity and selected-best provenance.
Production deployment and the `smoke-resume` CUDA interruption/replay stage
completed. Actual H100 research and forecast/head replay was bitwise identical;
independent artifact checks are recorded below. Preserved reference artifacts
remain unchanged; the revised run uses new worker/protocol hashes.

## Current behavior and failure boundary

The driver submits one `research-generate` call for each date fold and the final
fit. Training calls now have a 21,600-second Modal limit and a 21,300-second
child watchdog; inference remains separately bounded at two hours. Training
receives a 21,180-second internal budget and checks for a checkpointed stop with
120 seconds remaining. Completed adapters and modules are cached with hashes.
The previous worker saved no optimizer/model state until all fitting epochs
completed, so timeout retries repeated the entire fit. The revised worker saves
at optimizer boundaries and resumes from the committed pointer. The existing
driver still permits at most three failed attempts.

## Minimal optimizer checkpoint contract

Save at an optimizer-step boundary, after clipping, `optimizer.step()` and
`zero_grad(set_to_none=True)`. Keep the existing microbatch size, accumulation,
loss normalization, shuffle algorithm, epochs and learning rate unchanged.
The standalone helper uses a three-minute default interval (at most five),
checked at optimizer boundaries. Worker integration should also honor an execution
deadline with enough reserve for one accumulation group, checkpoint serialization
and volume commit. Measure
group duration and reserve conservatively; never stop midway through a group.

Each immutable `research/<fold>/checkpoints/step_<number>_<unique suffix>/` contains:

- Adapter-only tensors in `adapter.pt`; exact scientific config and pinned base
  identity are bound in the manifest (the existing immutable run config retains
  the full PEFT construction settings).
- AdamW state and parameter-group identity in a separate tensor checkpoint.
- Epoch index, current shuffled row-index order, next accumulation-group offset,
  completed optimizer-step count, loss sum/count and completed epoch history.
- Python shuffle-generator state, Python global RNG state, torch CPU RNG state
  and all CUDA RNG states; no gradient-scaler state is needed for BF16 autocast.
- Hashes of scientific config, ordered input rows, tokenized examples, worker,
  tokenizer and base revision, plus all checkpoint files and their sizes.

Write into a temporary sibling directory. Flush files, write its manifest last,
then rename into the immutable step directory. Advance `latest.json` atomically
only after all manifest hashes verify. Emit `checkpoint_committed` only after
the parent has committed the volume. Retain at least the latest two complete
checkpoints; incomplete temporary directories are never candidates for resume.

On restart, validate all identity/file hashes before loading state. Construct the
same base and trainable PEFT modules, load the adapter and optimizer states, then
restore RNG states **after** initialization/loading (which consume or reset RNG).
Continue at the saved group offset without reshuffling or replaying updates.
Validate optimizer state against named trainable parameters and refuse any
different target ordering, hyperparameters or model revision. Completed
`training.json` remains the terminal fit artifact and binds the final adapter.
The helper requires a live `named_parameters()` mapping; it derives optimizer
order from parameter object identities, so same-shaped reversed parameters
cannot silently exchange Adam moments. The manifest binds tensor devices,
torch version, CUDA availability/device count/device names, and CUDA RNG count.
CPU/CUDA runtime transitions fail before mutation. For forecast integration,
include head tensors in saved trainable state and supply distinctly named head
parameters alongside model parameters. Restore the model/head training modes
before resumed updates. The completed CUDA integration smoke is described below;
the CPU tests alone would not establish NF4 kernel reproducibility.
Default local artifact exports should include checkpoint manifests and cursors,
not every optimizer/adapter tensor copy; the full resume state stays on the
committed cloud volume and is downloadable explicitly if needed.

## Preferred deployment: measured training timeout and ordinary resume

After measuring representative sequences, extend only training-capable worker
calls to a realistic bounded duration (the measured 4k/8k results suggest six
hours, pending the maximum-length probe), with a shorter
child watchdog that leaves time for final checkpoint/volume commit. Preserve the
two-hour pure generation limit. Modal supports a maximum 24-hour function
timeout; the chosen limit must follow the measured workload rather than simply
using that maximum. Parent documentation check:
https://modal.com/docs/guide/timeouts .

The current three-attempt retry budget can remain if each retry resumes actual
progress and the measured fit fits comfortably within the extended call. This
is the preferred minimal change: no additional continuation API is necessary
unless the measured workload or an operational time limit requires it. Include
checkpoint cadence and elapsed-time reporting in logs so long calls remain
inspectable. The caller continues awaiting completion with its durable call ID.

Throttle ordinary Modal volume commits to every 15–30 seconds while still
writing/flushing every local log line. Commit immediately after an explicit
completed-checkpoint or completed-batch marker, and on exit/finally. Current
commit-on-every-line behavior can block draining hundreds of model warnings for
minutes; it changes transport overhead, not inference math. Preserve request
commit before launching the subprocess and final result commit afterward.

The maximum-length SFT probe does not cover forecasting's simultaneous prefix
Brier and teacher-forced CE graphs. Once actual report/target lengths exist,
measure that path or perform mathematically equivalent separate backward calls
before the same optimizer step, so only one graph remains live at a time.

## Optional bounded continuation if the measured workload requires it

Pass execution deadline/continuation ID through worker CLI or a separate
execution argument; do not put them in scientific config hashes. A graceful
budget stop returns `status=checkpointed`, checkpoint hash and next cursor.
The driver persists this result, verifies/downloads its manifest, and submits a
new execution ID tied to that exact checkpoint. A completed stage is not marked
complete merely because a continuation checkpoint exists. Interrupted calls use
the same recovery contract and never mutate source/target input files.

Keep a separate bounded continuation count derived from measured expected
runtime, with a total wall-clock budget. Do not spend the three failure retries
on successful checkpoint continuations. Detect no-progress continuations by
identical optimizer-step counters. Separate generation from fitting after fit
completion when remaining time cannot safely cover a generation batch.

## Same-adapter output-length recovery

Persist every generated completion immediately, including raw token IDs/text,
EOS/finish reason, actual/requested output tokens, original input hash, fold,
base/adapter identity, and attempt number. A length-limited completion is a
rejected artifact, never a valid research report. Preserve successful sibling
outputs when one request in a batch fails.

Predeclare an output budget ladder, for example 4,096 then 8,192 tokens, with at
most two length attempts. The second attempt uses the identical source prompt
and the exact same fold adapter; verify input plus output fits the model context.
The ladder belongs to the new scientific protocol so changed output policy
cannot silently reuse old caches. Cache keys bind attempt budget and adapter.
Do not drop source tokens, insert gold, or switch to the base model.

Once a completion ends normally, apply existing mechanical and independent
semantic review. Factual repair remains a distinct bounded source-only
same-adapter process; it must not disguise truncation. Exhausted length attempts
remain visibly rejected with all raw outputs available for inspection.

## Verification before full fitting

Use a small deterministic CPU fixture to compare uninterrupted training with a
save/reload continuation at a group boundary, including the next shuffled epoch
and RNG sequence. Check malformed/checksum-mismatched checkpoints fail before
updates; incomplete directories never resume; stage completion cannot be forged
by a continuation; and length recovery preserves original prompt/adapter hashes
and rejects exhausted outputs. Then run a short CUDA interruption/restart check
with the actual NF4/PEFT path. Numerical tolerance must account for CUDA kernels;
identical step counts, data order and restored optimizer/RNG state are mandatory.

## Completed H100 interruption/replay validation

Run `bam_resume_smoke_v4_20261102`, call
`fc-01M6CSFV4MT7BZSZS2T6SY08D7`, completed in 144.91 seconds on an NVIDIA H100
80GB HBM3 with torch 2.6.0+cu124. It used the actual pinned 14B NF4/LoRA loader,
dropout and checkpoint helper. Each synthetic research and forecast/head path
performed six updates over two epochs, interrupted after update four, then
loaded a fresh model and resumed from the saved state.

| Path | Compared tensors | Bitwise equal | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| Research adapter + optimizer + RNG | 1,538 | 1,538 | 0 |
| Forecast adapter/head + optimizer + RNG | 1,546 | 1,546 | 0 |

Independent local inspection verified all 43 exported files against the export
manifest, the latest checkpoint pointers against their manifest hashes, and
cursor files against their manifests. Final `adapter.pt`, `optimizer.pt` and
`rng.pt` hashes were identical between uninterrupted and resumed paths for both
stages. Scientific identities, shuffle order, loss history and final cursors
also matched. The interruption cursor was epoch index 1, global step 4, next
group offset 1; the final cursor was epoch index 1, global step 6, offset 3.
Forecast selection retained the identical synthetic dev-selected epoch 2 and
identical selected adapter/head file hashes.

Artifacts and the independent audit are under
`data/probes/bam_cloud/20261102_resume_smoke_v4/`, including
`independent_replay_audit.json`, `artifact_export_manifest.json`, and
`artifacts/synthetic_resume_smoke/result.json`. The verified export archive hash
is `3c401d5a3909d35c10b449fb92d75b23023ba1b438d07e44b72d823fe0b9e5de`.
Worker: `36c0095db4e44e6dd36bcceb4140dad8d281d0dedc50166b61a62fef71fe96c8`;
helper: `106fd62fc78fb5fcdfa12d6a08f264271c3c68de91766b920c415f76961f8c2d`.

This validates short synthetic interrupted/reloaded training, not NBA forecast
performance. The earlier maximum-length probe covers research SFT memory;
maximum-length forecasting memory and actual corpus runtime remain separate.
