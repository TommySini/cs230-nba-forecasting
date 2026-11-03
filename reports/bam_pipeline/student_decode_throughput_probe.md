# Measured NF4 student decoding throughput

**Batch four improved aggregate throughput by 1.83× at 4k input tokens and 1.48×
at 8k, with at most 20.56 GB peak allocated memory in these measurements.**
Use batch four as the candidate for a new configuration at the tested lengths;
the production configuration has not been edited. This is an inference throughput
probe with an initialized, zero-update adapter, not actual research SFT or NBA
forecasting evidence.

| Exact prompt tokens per sequence | Batch size | Median seconds / 256-token generation | Median aggregate output tokens/s | Peak allocated GB | Peak reserved GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 2 | 21.805 | 23.54 | 12.67 | 14.93 |
| 4,096 | 4 | 23.828 | 42.97 | 15.30 | 16.84 |
| 8,192 | 2 | 23.747 | 21.56 | 15.30 | 16.84 |
| 8,192 | 4 | 32.184 | 31.82 | 20.56 | 22.44 |

GB is decimal. Each shape had a 16-token warmup followed by two timed repetitions,
with CUDA synchronization and peak-memory reset. Every timed sequence generated
exactly 256 tokens. Timings include prefill, decoding and the production helper's
output transfer to Python. Aggregate throughput counts every sequence in the
batch; per-sequence latency increases with batch size. Medians of seconds and
throughput are calculated separately from the two measurements.

## Exact tested path

The worker called unchanged `bam_training._load_model` and
`_transformer_completion_batch`: pinned Qwen2.5-14B-Instruct revision
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, bitsandbytes NF4 double quantization,
BF16 compute, SDPA, KV cache on, inference mode, and greedy decoding. Hardware was
one NVIDIA H100 80GB HBM3. Gradient checkpointing was off for inference.

A valid rank-8/alpha-16 LoRA adapter was created through the configured training
loader, checked for zero initialized B matrices, saved, and reloaded through the
production adapter-inference loader. **No optimizer updates occurred.** Adapter
initialization took 49.42 seconds and inference reload took 16.40 seconds.
The synthetic adapter is not eligible for any research or forecast input.

Only this benchmark disabled EOS/forced EOS and set `min_new_tokens` to the timed
256-token budget (16 during warmup). Production stopping behavior is unchanged.
Four distinct, reproducible outcome-free synthetic prefixes were verified locally
with the pinned tokenizer. All input/output token hashes, adapter file hashes,
library versions, model/worker identities and raw logs are retained.

## Interpretation and budget correction

Batch four is preferable for aggregate throughput at these measured shapes. Its
memory use leaves substantial H100 headroom here, but this does **not** validate
16,384-token prefixes or 4,096/8,192-token continuations. Real variable-length
batches can waste work waiting for the longest completion; bucketing by prefix
length is a possible later execution change. No backend, stopping rule, model,
adapter or training objective should be changed on the strength of this test.

The earlier remaining-budget report assumed 30–60 aggregate tokens/s for the
student path. That range was too optimistic for the current batch-two setting:
observed batch-two throughput is about 21.6–23.5. The latest candidate cohort is
424 research games, or 1,272 three-module reports. Extrapolating the short measured
rates to 1,500–3,000 tokens/report yields the following **conditional arithmetic**:

| Generation setting | Projected inference hours | Allocated-resource cost |
| --- | ---: | ---: |
| Current batch two | 22.5–49.2 | about $107–233 |
| Candidate batch four | 12.3–33.3 | about $59–158 |

These estimates exclude initialization, checkpoints/volume commits, repairs,
uneven batch tails and slowdown from longer continuation KV caches. Prefill is
amortized differently for long outputs. Do not treat the projections as measured
long-report performance or a completion guarantee. They use $4.743648/hour for
the configured H100 + six physical CPU cores + 64 GiB RAM at published rates.
[Modal pricing](https://modal.com/pricing).

For context, the v4 diagnostic's 44 module/partial outputs had median 2,693.5
tokens. Applying that length to every future report would imply roughly $192–209
at batch two or $105–142 at batch four before the excluded costs. Student output
lengths and quality are still unknown; this is a scenario, not a substitute for
the first actual fold's measurements. The earlier $40–180 student-generation
allowance needs revision if batch two remains fixed.

## Bounded execution and artifacts

- Call: `fc-01M6CY92P45A1A76TE7QKPEHGA`.
- Isolated app: `cs230-nba-bam-student-decode-probe`.
- Run: `bam_student_decode_probe_20261102`.
- Local artifacts: `data/probes/bam_cloud/20261102_student_decode_probe/`.
- Controller: `scripts/probe_bam_student_decode.py`.
- New cloud/worker files: `bam_student_decode_probe_cloud.py` and
  `bam_student_decode_probe_worker.py`.

The call ID was persisted before waiting. Exactly one call ran, capped at ten
minutes with one H100 and no retries. It finished successfully; the isolated app
completed. Worker elapsed was 280.87 seconds; wrapper elapsed was 296.49 seconds.
At the reserved resource rate this is approximately **$0.39**, an estimate rather
than an invoice. The ten-minute allocation bound was approximately $0.79 before
resource-usage variance. No models or packages were downloaded; cached offline
weights and the existing training image were used. Frozen production files remain
unchanged.
