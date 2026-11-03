# Compute handover assessment — November 1, 2026

**Current path:** deploy `scripts/modal_research_vllm.py` once, then submit jobs
with `.venv/bin/python scripts/modal_research_vllm.py` (see **Modal measurements**
and **Adopted backend** below). The full-note experiment completed; do not start
`scripts/archive/modal_research_batch.py` or ephemeral `modal run` apps for new
production research.

Sections before **Modal measurements** are **historical planning / local timing**.
Benchmark timings under MLX CUDA vs vLLM are retained for comparison.

## Historical: preferred route (planning notes)

Use the user's existing university/cloud credits on one suitable NVIDIA GPU.
An L40S (48 GB), A100 (40/80 GB) or H100 (80 GB) has capacity for the current
28 GiB MLX memory limit. Availability, quotas, queue time and credit eligibility
depend on the provider. Start with a short benchmark; select from GPUs actually
available under those credits rather than opening another paid account.

MLX 0.32.4 provides a Linux CUDA backend, and its release history includes
quantized matrix multiplication and matrix-vector support. This offers a way to
test the **same pinned four-bit September 2024 checkpoint** without replacing the
model. A working CUDA installation has now been tested for this exact job.
See [installation requirements](https://ml-explore.github.io/mlx/build/html/install.html)
and [MLX releases](https://github.com/ml-explore/mlx/releases).

For context only, the checked Runpod public listing shows A100 80 GB at $1.79/hour,
H100 PCIe at $2.89/hour, H100 SXM at $3.99/hour, and L40S at $1.09/hour, before any
separate storage charges. These are advertised rates, not a booked instance or
a total experiment quote. [Runpod pricing](https://www.runpod.io/pricing).
CS230's project page describes AWS GPU credits, but the user's actual provider,
credit balance and usable GPUs have not been confirmed.
[CS230 project resources](https://cs230.stanford.edu/project/).

## Historical: prepared MLX/CUDA device benchmark

`scripts/benchmark_research_device.py --stage prepare` creates the small
`data/probes/research_compute_bundle/` snapshot. Its archive contains ten files,
about 50 KB compressed, with all 15 pages of the reviewed Miami **training**
packet. It excludes target labels, booster probabilities and credentials.
The exact model can be downloaded separately from its pinned public revision
(7.75 GiB). File hashes and unchanged request hashes are checked before execution.

After provisioning an authorized GPU and copying this private bundle, run from
the bundle root with a compatible Linux/CUDA installation:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements-research-cuda.txt
.venv/bin/python scripts/benchmark_research_device.py --stage check
.venv/bin/python scripts/benchmark_research_device.py --stage run --download-model
```

The benchmark measures extraction/checking time and peak GPU memory, repeats the
twelve factual challenges, compares generated output hashes with local references,
and exercises the encoder's cached hidden-state operations. Its outputs stay in
`probe/results/`; it cannot replace main experiment responses. Exact text equality
across GPU backends is not assumed. The known semantic-checker failures remain
visible; source quotations/code-checked figures are still required for forecasting.

## Historical: full research transfer snapshots

`scripts/export_research_compute.py` creates a separate, timestamped snapshot of
the frozen research code, all pregame source units, protocols and completed
responses. It does not upload anything. Raw ESPN, target outcomes, boosting
features, account credentials and model weights are excluded. The snapshot has
a file manifest; the same public checkpoint is downloaded on the compute host.

Run only `--stage research` remotely, then return the research outputs and record
the remote runtime identity. The local source validator must pass against the
original archived PDFs before any returned evidence enters forecast preparation.
Keep all encoder vectors on one verified backend. Preserve the local run until
the remote benchmark and transfer are working; coordinate the eventual cutover
so two processes do not write the same results. A new snapshot should be taken
at cutover to include additional local progress.

## Modal measurements (current deploy path)

The existing Modal workspace `tsini` is authenticated. `modal==1.6.3` was added
only to the project venv without changing its 59 existing package versions.

**Current launch workflow:** deploy `scripts/modal_research_vllm.py` once with
`.venv/bin/modal deploy --env main scripts/modal_research_vllm.py`, then submit
jobs with `.venv/bin/python scripts/modal_research_vllm.py` and the existing
snapshot/resume options. The Python CLI uses named function lookups and never
starts a temporary app. `--check-deployment` only checks registration. Later
deploys update `cs230-nba-research-vllm` in place; zero idle containers and the
existing shared volumes are retained. The benchmark/cutover entries below are
historical runs, not instructions to start another app with `modal run`.
The superseded MLX batch driver is under `scripts/archive/`.

The private `cs230-nba-research-models` and `cs230-nba-research-runs` volumes hold
the verified public checkpoint and label-free probe outputs. No account plan
upgrade or credit purchase was performed; the user's credit balance is unknown.

The successful MLX/CUDA run `20261101T090405Z` used one NVIDIA H100 80 GB.
All pinned checkpoint hashes match. The 15-page training packet took **144.1 s
for extraction and 30.0 s for support checking**, versus local measurements of
125.0 s and 79.4 s (about 1.17x overall, not a compelling speedup).
Nine of fifteen extraction texts matched the local reference exactly. The
5120-dimensional encoder kernel probe was finite. Semantic challenges still
failed the same two of twelve cases; this is not factual certification.
Results: `data/probes/modal_benchmark/20261101T090405Z/`.

The GPU job releases its container on completion. Earlier bounded attempts
exposed a symlinked-venv check and mismatched CUDA shared libraries; both were
fixed before the successful probe. Runtime CUDA libraries must be resolved as
one matched pip-provided set rather than mixed with the older base-image cuBLAS.

`scripts/modal_research_vllm.py` now benchmarks a separate CUDA-optimized engine.
It reconstructs FP16 weights from the **same pinned quantized checkpoint**,
preserving the original model files. This restores the quantized levels, not the
original pre-quantization Qwen weights. Exact prepared input token hashes are
checked; inference arithmetic and batch scheduling differ and are recorded in
an execution manifest. No new backbone or outcome-based model choice is made.
Full execution requires reviewing its training-only timing and factual checks.
Returned research must pass local source checks before entering forecasting.

Modal's checked rates are H100 $3.95/GPU-hour, plus CPU and host memory usage.
The isolated benchmark has an 1100-second subprocess deadline; full jobs have
a 6900-second deadline and retain partial artifacts on failure. See
[Modal pricing](https://modal.com/pricing), [GPU configuration](https://modal.com/docs/guide/gpu),
and [vLLM offline inference](https://docs.vllm.ai/en/latest/getting_started/quickstart/).

## Adopted backend and full run

The vLLM benchmark `20261101T093252Z` completed on H100: **22.72 s extraction
+ 18.10 s checking = 40.82 s**, about **5.0x** faster than the measured local
204.4 s. It failed the same two factual challenges (wrong streak attribution and
the deliberately false Bam scoring claim), with all schema/source checks passing.
No outcome or test score entered this compute choice. The frozen encoder still
runs through the original local MLX checkpoint for every game.

The conversion uses `convert_research_weights.py`: direct affine unpacking with
FP32 arithmetic and one FP16 rounding, matching MLX GPU sample rows exactly for
all 338 matrices. The full cloud conversion rechecks those reference hashes.
It produces 15 weight shards / 29,540,067,328 tensor bytes, with file hashes kept
in the execution identity. MLX's own converter stalled; those attempts were
stopped. Two CPU preemptions interrupted the direct conversion before a complete
retry succeeded. Future conversion calls use a non-preemptible CPU function;
this applies Modal's 3x CPU/memory multiplier to that short step, not the GPU job.

At cutover, 18 games were complete locally. The private research-only snapshot
`20261101T094350Z.tar.gz` contains 1,879 files / 8,163,544 bytes. An explicit
allowlist, hashes and source audit confirmed 107 public PDFs from `nba.com` and
`cdn.nba.com`, archived pregame, plus derived research/code/configuration. No
target labels, booster inputs, raw ESPN, credentials or private user documents.
An initial automated review blocked export as potentially sensitive; the audit
was supplied and the same action was then approved. See `modal_transfer_audit.json`.

Full run **20261101T094719Z** stopped after 45 completed games when one checker
response repeatedly omitted the eighth required verdict. Its GPU job exited;
completed responses were retained. Resume **20261101T183024Z** is active in
[Modal](https://modal.com/apps/tsini/main/ap-5rdJuJFYrSD9wXS4zYGvhi), reusing the
same remote cache and original audited snapshot. The retry now explicitly lists
required and missing IDs; it does not invent or silently discard verdicts. The
live run passed the failed request and reached 46/107 completed games. Four
regression tests cover the fix; all 76 tests pass. Model, initial prompts,
sources, inference settings and forecasting experiment remain unchanged.
The original local writer was stopped, preserving its responses and complete
snapshot. The execution plan is `cloud_execution_plan.json`. Research resumes
after preemption and the H100 subprocess has a 6,900-second limit. Results return
to `data/probes/modal_vllm/20261101T183024Z/`; `--finish-local` then imports,
validates original sources and runs the existing preparation/encoding/head
training/dev selection/evaluation/replay pipeline. The local driver must remain
running for these follow-up steps. No revised Brier result exists yet.

At 09:41 UTC, Modal reported $1.31 metered usage, covered by credits with $0
billed then. This is a timestamped setup-usage observation, not the final run
cost or the account's remaining credit balance.

## Encoder timing and interrupted-call recovery (November 1)

A separate, training-only encoder probe sent the exact token IDs for all three
chunks of game `espn:401704720` to the same original four-bit MLX checkpoint on
an H100. It checked checkpoint hashes, package versions, output membership,
complete chunks and the normalized 5,120-dimensional vector. No labels or head
fitting entered the probe. Results are in
`data/probes/encoder_device_probe/comparison.json`.

Local encoding took 48.4748 seconds; CUDA MLX took 45.8896 seconds, excluding
cloud startup. The pooled vectors had cosine similarity 0.99998739, with maximum
component difference 0.00084372. This is close numerical agreement, not bitwise
equality. The initial decision was to retain local encoding because this was not
a useful speed gain. That decision was revised after the sleep interruptions
described below. The probe's vector remains separate from the experiment.

Research initially failed at 45 games because a support-check response omitted
one verdict. Explicit missing-ID retry feedback fixed that request without
inventing a decision or changing the validation rules. A client network failure
later cancelled the temporary-app resume after 83 completed games were logged;
its saved cache recovered through 87 games. A named-deployment resume reached
100 games, then received a cancellation during a deployment update. All completed
raw responses remain cached. The final recovery uses the same public-source
snapshot and remote directory, with one explicitly reviewed fourth invocation
(`--invocation-budget 4`); each invocation still has a bounded 6,900-second GPU
subprocess and no automatic retry loop. Research model, inference parameters,
source prompts, split, head architecture and dev-selection rule are unchanged.
The CLI now uses a spawned call and saves its ID before waiting, allowing the
same cloud call to be reattached after local connectivity failures. Do not
redeploy while a call is running.

## Uniform cloud encoder after Mac sleep interruptions

The local pipeline validated all research and prepared all forecast inputs, then
encoded 51 games. At approximately 19:45 UTC the Mac began repeated maintenance
sleep cycles on battery, greatly extending wall-clock inference time. Read-only
power logs confirmed the sleep/wake pattern; no power settings or unrelated
processes were changed. The encoder was stopped deliberately and its complete
partial directory and protocol were preserved under
`data/probes/local_encoder_interrupted/`.

Before any head fitting, `encoder_execution_plan.json` freezes the decision to
encode all 107 games with the same original four-bit MLX checkpoint on CUDA.
Two H100 workers run in the existing deployment, each bounded to 6,900 seconds,
with no idle containers. They receive only exact pregame prompt token IDs and
model metadata, not target outcomes. Requests and retrievable function-call IDs
are saved under `data/probes/full_encoder_cuda_20261101/`. All tokens, complete
chunks, weights, runtime versions, pooling and vector provenance are checked.
Local and CUDA vectors cannot be mixed. The original cohort, source prompts,
head objective, regularization candidates and dev selection remain unchanged.
Numerical arithmetic differs from the archived local run, as measured above;
the selected result will use one backend uniformly. The plan and encoder
implementation hashes are frozen in the forecaster protocol.

## Completed result

Pipeline `20261101T201235Z` completed in 495 seconds after switching to the
uniform cloud encoder, including cached research replay, all source checks,
encoding, head fitting, dev selection, evaluation and final replay. The two
encoder calls returned 54 and 53 validated vectors in 369.6 and 374.8 seconds.
Median packet encoding after warmup was 4.83 and 4.33 seconds; summed actual
encoding time across both workers was 643.46 seconds. The single-packet timing
probe included first-use compilation and did not predict steady batch throughput.

All 107 vectors use MLX CUDA with the original quantized checkpoint. The exact
input file is byte-identical to the interrupted local run, and 17 scientific
protocol fields are unchanged. Only execution/backend bookkeeping and its
implementation changed before fitting. Raw cloud requests/results, code hashes
and runtime identities are retained; no local vectors enter this result.

Dev selected L2=.001 (Brier .347535 -> .332851). Test Brier is .186296 for
original boosting and .193206 for the researched hybrid, a difference of
+.006910 with paired date-bootstrap 95% interval [-.011766,.029196]. Both have
78.6% accuracy. This exploratory 14-game comparison does not establish a gain
from notes. See [the result](../researched_forecaster/findings.md).
All85 tests and both source/forecaster validation stages pass. A final read-only
Modal status check found zero backlog, running inputs or containers for
`execute`, `encode_reports` and `convert_model`; the deployment remains available.
