# Isolated H100 training throughput probe

The pinned 14B NF4/LoRA SFT path completed synthetic updates at the configured
16,384-token maximum on one NVIDIA H100 80GB HBM3. This verifies the research
SFT memory path; it does not establish research quality, generation throughput,
or the combined forecast Brier-plus-explanation graph's memory use.

| Prompt tokens | Target tokens | Total tokens | Warmup / timed updates | Median seconds/update | Peak allocated GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 512 | 4,608 | 2 / 3 | 1.5113 | 27.89 |
| 8,192 | 512 | 8,704 | 2 / 3 | 2.9041 | 39.48 |
| 12,288 | 512 | 12,800 | 1 / 2 | 4.4486 | 51.07 |
| 15,872 | 512 | 16,384 | 1 / 2 | 6.0250 | 61.21 |

GB above is decimal. At maximum length, timed peak reserved memory was 63.70 GB;
the first maximum-length warmup temporarily reserved 80.98 GB. Allocated memory
peaked at 61.21 GB in both warmup and timed updates. CUDA synchronization brackets
each measured forward, completion-only CE, backward, gradient clipping and AdamW
update. Logs/volume commits and model initialization are outside step timings.
Model loading took 38.71 seconds in the first successful call and is recorded
separately in each result.

The exact production loader was imported unchanged: Qwen/Qwen2.5-14B-Instruct,
revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, NF4 double quantization,
BF16 compute, rank-8/alpha-16 LoRA, dropout 0.05, SDPA, nonreentrant gradient
checkpointing, microbatch size one, and 12,582,912 trainable parameters.
Packages were torch 2.6.0+cu124, transformers 4.51.3, peft 0.15.2 and
bitsandbytes 0.45.5. The worker hash was
`ea49a22ffd3e0414fd5e20f09f57dfab39436c61c7f8797dd6582f7f7e106acb`.

Synthetic prefixes contain only repeated outcome-free source statements. Every
completion has exactly 512 tokens, including its chat terminator. No NBA labels,
reports, gold targets or forecast results were used. Real target lengths vary;
these timings are shape-based planning measurements, not a representative
distribution of real examples. The probe updates AdamW every microbatch;
production accumulates eight microbatches, so optimizer overhead differs.

## Runtime implication for the original 229-training-game snapshot

Chronological fold fits contain 258, 483 and 687 module examples, respectively;
three epochs require 774, 1,449 and 2,061 sequence passes. Cold-fold fitting is
empty. The complete set of fits therefore requires 4,284 sequence passes.

| Assumed sequence shape for every example | Final fit only | All fits only |
| --- | ---: | ---: |
| Measured 4,608 total tokens | 0.87 hours | 1.80 hours |
| Measured 8,704 total tokens | 1.66 hours | 3.46 hours |
| Measured 12,800 total tokens | 2.55 hours | 5.29 hours |
| Measured 16,384 total tokens | 3.45 hours | 7.17 hours |

These projections exclude tokenization, model load/reload, checkpoint writes,
volume commits and generation. Generation counts are 258, 225, 204 and 81 module
reports across the cold, two fitted and final folds. Output generation was not
benchmarked. At the measured 8,704-token shape the final fit leaves only about
915 seconds of the current 6,900-second watchdog for all other work; at the
maximum shape it cannot finish before that watchdog.

A six-hour training-call limit, periodic exact checkpoints and ordinary resume
is a reasonable measured starting bound. Retain rank 8, microbatch one,
accumulation eight and the 16,384-token training limit; do not reduce examples or
truncate sources to fit the old timeout. Keep pure inference calls separately
bounded. The forecast joint graph needs its own check once real input/explanation
lengths exist; the SFT measurement does not cover simultaneous graphs.

## Durable artifacts

- `data/probes/bam_cloud/20261102_training_throughput_v2/`: request, call ID,
  raw execution log, response and results for 4,608/8,704 tokens.
  Call: `fc-01M6CQDK3GTBBFNJ4YCXZQNTAQ`.
- `data/probes/bam_cloud/20261102_training_throughput_v3/`: the same artifacts
  for 12,800/16,384 tokens. Call: `fc-01M6CQK2XR3M6YAA6EF8XQPV08`.
- `data/probes/bam_cloud/20261102_training_throughput/`: preserved first-attempt
  request/call/failure metadata. This attempt failed during startup because an
  imported helper module was not mounted; training did not run. The isolated
  app was explicitly stopped before the self-contained wrapper was retried.

Both successful isolated apps completed; production code and its running
reference protocol were not modified. Results contain per-step timings,
token-vector hashes, library versions and peak memory, not just this summary.
