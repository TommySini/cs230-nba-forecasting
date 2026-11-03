# BF16-base student LoRA inference probe

The isolated vLLM path works with the actual six-update **synthetic** research adapter from the CUDA replay test. It delivered **53.84–62.25 aggregate output tokens/s** for natural completions at batch four, and **59.60 tokens/s** for a separately marked fixed-256-token batch. This is a useful measured improvement, approximately **1.87×** the earlier NF4 batch-four 8,192-token benchmark (31.82 tokens/s), rather than the unmeasured 100–400 tokens/s planning scenarios. It does not establish NBA report quality or identical NF4/BF16 outputs.

Successful call: `fc-01M6D1XJBV9E9RV95BR101DT92`, run `bam_student_vllm_probe_int32_20261102`. Raw request, result, execution log, and independent audit are in `data/probes/bam_cloud/20261102_student_vllm_probe_int32/`. No production worker, driver, historical artifact, or model weight was changed.

| Case | Prompt tokens per request | Output tokens, batch total | Seconds | Aggregate output tokens/s |
|---|---:|---:|---:|---:|
| Natural EOS, batch 4 | 4,093 | 1,852 | 34.40 | 53.84 |
| Natural EOS, batch 4 | 8,193 | 1,875 | 30.95 | 60.58 |
| Natural EOS, batch 4 | 8,180–8,193 | 1,792 | 28.79 | 62.25 |
| Fixed-length benchmark only, batch 4 | 8,180 | 1,024 | 17.18 | 59.60 |

The twelve natural completions were 427–502 tokens (median 456.5). All twelve had verified pinned EOS token IDs, unchanged input token IDs, and the expected adapter identity. Offline re-tokenization with the pinned local tokenizer independently reproduced every prompt-token hash and request hash; decoding the returned tokens reproduced every saved completion. The fixed-length batch deliberately disabled EOS and used min/max 256 tokens, so it makes no natural-completeness claim. This single fixed-length timing is not a confidence interval or an exact paired comparison with the earlier NF4 prompts.

Two additional one-token controls used exactly the same synthetic prompt, once with the base and once with the nonzero trained adapter. The maximum absolute common-token logprob difference was **2.072977**. This proves the adapter affected computation; matching or differing greedy text was not required. Eighteen total generations ran. No NBA training or NBA outcome data was used.

## Runtime and provenance

The cached official checkpoint was `Qwen/Qwen2.5-14B-Instruct`, revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`. Original model and tokenizer files were hash-verified. The synthetic checkpoint manifest and adapter tensor hash were checked before a lossless CPU conversion from the existing `adapter.pt` state dictionary to safetensors. Adapter content hash: `930280e9e2614bcc31f1c1101c80a1a2c996d0751eee13520b44c55caef9495f`.

Training used the earlier NF4 double-quantized base with BF16 compute; this inference experiment uses the original **BF16 base and BF16 LoRA execution**. The precision and runtime differences are explicit. There is no token-equivalence claim. Recorded inference versions are vLLM 0.32.0, Transformers 5.19.0, Torch 2.13.0 and tokenizers 0.23.3. Configuration: one H100, six CPU cores, 64 GiB RAM, TP1, rank8, alpha16, q/k/v/o projections, eager execution, at most four concurrent sequences, 8,192 prefill tokens, 32,768 context, and 0.85 GPU memory fraction. Each completed group of four is written before the next group and causes a volume commit marker.

Engine construction took 36.15 seconds; the child completed in 184.33 seconds, including checks, initialization, controls and all generations. Engine profiling reported 27.9 GiB weights/non-Torch memory, 0.95 GiB peak activation and 38.45 GiB allocated KV cache. These are initialization-profile values, not a separately sampled maximum across all benchmark generations.

Two earlier attempts remain preserved:

- `fc-01M6D0V2MAXFVPPNFC8CEC6MS5` stalled before weights loaded. The observed 595 MiB/0% GPU state and successful spawn-method correction are consistent with the documented fork/thread-pool incompatibility; no stack trace proved the exact deadlock site. It was cancelled with container termination; the call became terminal and the app had no remaining containers. Its full **$1.82 reservation remains uncertain** pending billing reconciliation.
- `fc-01M6D1HS0C9KWZA4D4ZMKW6C28` initialized successfully, then failed with a concrete int32 adapter-ID overflow. Its verified completed response was reconciled at **$0.26 estimated cost**. The corrected backend uses a positive int32 process-local handle while retaining the full SHA256 and rejecting differing adapter hashes sharing a handle within a batch.

The successful ten-minute call reserved $0.91 and reconciled at **$0.37 estimated cost**, using local elapsed plus 30 seconds and 15% margin. These are conservative local estimates, not provider invoices. Total accounted for these three attempts is $0.63 completed estimates plus the retained $1.82 uncertain reservation. All spending remains in the explicit shared $150 ledger. No further GPU call was launched.

The API choices follow [vLLM LoRA documentation](https://docs.vllm.ai/en/stable/features/lora/). The runtime fix follows [vLLM multiprocessing guidance](https://docs.vllm.ai/en/stable/design/multiprocessing/) and the [pinned 0.32 environment default](https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/envs.py). The int32 restriction is visible in [pinned LoRA request state](https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/v1/worker/gpu/lora_utils.py). The current [BitsAndBytes documentation](https://docs.vllm.ai/en/stable/features/quantization/bnb/) requires an external plugin absent from this pinned image, so this probe does not claim an equivalent NF4 vLLM path.

## Conditional full-student-stage estimate

The frozen 424-game design exports three final modules per game: **1,272 generated reports**, including 411 cold-fold reports, and 1,098 accepted training targets if all training games pass review. Upstream module partials are not additional student SFT/inference rows.

For a planning scenario of 1,500–3,000 output tokens per final report, applying the observed 53.84–62.25 tokens/s to all 1,272 reports gives **8.51–19.69 H100 hours, $40.39–$93.39**, before loading, repairs or other pipeline stages. A simple 20% extra-generation allowance gives **$48.47–$112.07**, plus startup overhead. The allocated rate is $4.743648/hour for one H100, six CPU cores and 64 GiB RAM using [Modal pricing](https://modal.com/pricing). The cold base might be faster, but it was not timed separately and receives no speculative discount.

These are conditional projections, not a full-run promise. The actual 424-game final student prompt distribution does not yet exist because the reference run stopped before export. Offline inspection found only two initial direct-final requests, at 2,245 and 9,851 tokens, and 35 upstream partial requests with median 9,408 and range 2,235–9,961. Partials are explicitly not a substitute for the missing final-input distribution. Only one initial direct-final raw response completed, with 794 tokens; it is too small and not a quality-approved corpus sample. The synthetic timing outputs are much shorter than the 1,500–3,000-token scenario. Context growth, more complex real reports, repair tails and differing output lengths can lower throughput. At an unmeasured 40 tokens/s with 20% repairs, the high scenario alone is about $151 before startup; thus this change cannot guarantee the entire pipeline fits $150.

## Minimal production integration proposal

1. Freeze an explicit new `research.inference_backend=vllm_bf16_lora` identity. Keep existing NF4 fitting, epochs, LoRA parameters, data order, checkpoints, fold boundaries and outcome isolation unchanged. Existing NF4 experiment artifacts retain their own identity.
2. Add a fit-only execution boundary around existing expanding-fold validation and `_fit_research`. Return the exact completed `training.json`, adapter file hash, source/target hashes and prediction row IDs. The cold fold returns no adapter. This prevents the old combined fit-and-NF4-generation stage from decoding the same reports first.
3. A separate inference-image worker loads that verified fold adapter from the same run volume and calls `StudentVLLM` / `generate_reports`. Use batch/chunk four initially, preserve per-request attempts, keep complete siblings, and reject full input plus output budgets exceeding context. Fold calls remain sequential around the shared aggregate; verify source input hashes and strict earlier training dates before merging generated modules.
4. Route **ordinary generation and every source-only repair through this same BF16 backend**. Reuse `validate_student_repair` and exact original/parent hashes, but replace its model loading and decoding with the verified backend. The repair worker must require the original inference identity, same model revision, adapter hash, libraries, EOS and settings. Do not send a BF16 report into the existing NF4 repair path while retaining its BF16 provenance. Keep bounded output-budget escalation, raw failed attempts and independent review.
5. Leave forecast/control training and evaluation unchanged. They receive only actual approved student reports, never gold targets. Preserve the manual source-quality gate; this synthetic speed probe grants no factual approval to NBA reports.

The new backend and probe are implemented and isolated, with nine focused CPU tests. Production fit-only routing, BF16 repair routing and driver integration are proposals, not completed changes.

After the successful GPU probe, independent review found that copying a valid cached sibling attempt could bypass the old self-hash-only cache check. The isolated backend now binds each attempt to its request hash, prompt-token hash, runtime hash and output budget, with regressions for sibling substitution and each binding. This CPU-only hardening does not change inference math; the exact GPU-executed source files are retained in the successful run’s `executed_sources/` directory.
