"""Synthetic NF4 + initialized LoRA decode throughput; no NBA data or fitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import bam_training as training


def synthetic_prompt(tokenizer, length, variant):
    start = tokenizer.encode('<|im_start|>system\nUse synthetic evidence only; no game outcomes exist.<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    end = tokenizer.encode('\nWrite a detailed source-only research report with uncertainty.<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
    pattern = tokenizer.encode(f'Synthetic source {variant}: Player availability remains uncertain; no confirmed absence is established. ', add_special_tokens=False)
    remaining = length - len(start) - len(end)
    if remaining < 1 or not pattern: raise ValueError('Invalid synthetic length')
    return start + (pattern * ((remaining + len(pattern) - 1) // len(pattern)))[:remaining] + end


def run(root, config):
    import torch
    import transformers, peft, bitsandbytes
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    result = {'synthetic_only': True, 'actual_sft_performed': False, 'adapter_optimizer_updates': 0,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [],
              'worker_sha256': training.file_hash(training.__file__), 'probe_sha256': training.file_hash(__file__),
              'model': training._model_config(config), 'gpu': torch.cuda.get_device_name(),
              'versions': {'torch': torch.__version__, 'transformers': transformers.__version__,
                           'peft': peft.__version__, 'bitsandbytes': bitsandbytes.__version__},
              'benchmark_only_generation_overrides': {'min_new_tokens': 256, 'eos_token_id': None,
                                                     'forced_eos_token_id': None},
              'production_helper': '_transformer_completion_batch', 'timed_repetitions': 2}
    training.write_json(root / 'result.json', result)
    # Construct the actual configured LoRA modules and save a valid zero-update
    # adapter. Reload through the unchanged production inference loader.
    model, tokenizer = training._load_model(config, training=True)
    zero_b = all(torch.count_nonzero(p).item() == 0 for name, p in model.named_parameters() if 'lora_B' in name)
    if not zero_b: raise ValueError('Synthetic adapter must have zero initial LoRA B matrices')
    adapter = root / 'synthetic_adapter'; model.save_pretrained(adapter)
    result['adapter_files_sha256'] = {p.name: training.file_hash(p) for p in adapter.iterdir() if p.is_file()}
    result['adapter_initialization_seconds'] = time.monotonic() - began
    del model; training._release(None)
    before = time.monotonic()
    model, tokenizer = training._load_model(config, adapter=adapter)
    result.update(inference_load_seconds=time.monotonic() - before,
                  attention=model.config._attn_implementation, use_cache=model.config.use_cache,
                  gradient_checkpointing=model.is_gradient_checkpointing,
                  nf4=model.config.quantization_config.to_dict(), lora=str(model.peft_config))
    model.generation_config.eos_token_id = None
    model.generation_config.forced_eos_token_id = None
    model.generation_config.min_new_tokens = 256
    for length in (4096, 8192):
        for batch_size in (2, 4):
            case = {'prompt_tokens_each': length, 'batch_size': batch_size,
                    'generated_tokens_each': 256, 'measurements': [], 'status': 'running'}
            result['cases'].append(case)
            items = [{'input_ids': synthetic_prompt(tokenizer, length, i)} for i in range(batch_size)]
            case['input_ids_sha256'] = training.value_hash([x['input_ids'] for x in items])
            training.write_json(root / 'result.json', result)
            try:
                # One short warmup per shape; no warmup timing enters the estimate.
                model.generation_config.min_new_tokens = 16
                training._transformer_completion_batch(model, tokenizer, items, 16)
                model.generation_config.min_new_tokens = 256
                torch.cuda.synchronize()
                for repeat in range(2):
                    if time.monotonic() - began > 440:
                        raise TimeoutError('Probe time budget reached; preserve completed shapes')
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    tick = time.perf_counter()
                    output = training._transformer_completion_batch(model, tokenizer, items, 256)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - tick
                    if len(output) != batch_size or any(len(ids) != 256 for ids in output):
                        raise ValueError('Fixed 256-token benchmark contract failed')
                    measured = {'repeat': repeat, 'seconds': elapsed,
                                'aggregate_output_tokens_per_second': batch_size * 256 / elapsed,
                                'per_sequence_output_tokens_per_second': 256 / elapsed,
                                'output_ids_sha256': training.value_hash(output),
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                    case['measurements'].append(measured)
                    training.write_json(root / 'result.json', result)
                    print(json.dumps({'batch_complete': 'student_decode_probe', **case, 'latest': measured}), flush=True)
                case.update(status='complete', median_seconds=statistics.median(x['seconds'] for x in case['measurements']),
                            median_aggregate_output_tokens_per_second=statistics.median(x['aggregate_output_tokens_per_second'] for x in case['measurements']))
            except torch.cuda.OutOfMemoryError as error:
                case.update(status='oom', error=str(error)); torch.cuda.empty_cache()
            except TimeoutError as error:
                case.update(status='time_budget', error=str(error))
                training.write_json(root / 'result.json', result)
                break
            training.write_json(root / 'result.json', result)
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - began)
    training.write_json(root / 'result.json', result)
    print(json.dumps({'batch_complete': 'probe_finished', 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--config', required=True)
    args = parser.parse_args(); run(args.root, json.loads(Path(args.config).read_text()))
