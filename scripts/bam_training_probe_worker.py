"""Isolated synthetic throughput measurement; never an NBA training artifact."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import bam_training as training


def synthetic_ids(tokenizer, prompt_length, target_length=512):
    """Construct exact token counts from repeated, outcome-free synthetic prose."""
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)
    start = encode('<|im_start|>system\nUse only these synthetic source facts; no outcomes are supplied.<|im_end|>\n<|im_start|>user\n')
    end = encode('<|im_end|>\n<|im_start|>assistant\n')
    source = encode('Synthetic source: a player may be available; the report does not establish certainty. ')
    target = encode('The synthetic source leaves availability uncertain. No prediction or result is established. ')
    eos = encode('<|im_end|>\n')
    def fill(pattern, length):
        if length <= 0:
            raise ValueError('Synthetic token budget too small')
        return (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
    prefix = start + fill(source, prompt_length - len(start) - len(end)) + end
    answer = fill(target, target_length - len(eos)) + eos
    return prefix + answer, [-100] * len(prefix) + answer


def run(root, config):
    import torch
    import transformers, peft, bitsandbytes
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, tokenizer = training._load_model(config, training=True)
    model.train()
    load_seconds = time.perf_counter() - started
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config['research']['learning_rate'], weight_decay=.01)
    result = {'synthetic_only': True, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'model': training._model_config(config), 'worker_sha256': training.file_hash(training.__file__),
              'probe_sha256': training.file_hash(__file__), 'load_seconds': load_seconds,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'peft': peft.__version__,
              'bitsandbytes': bitsandbytes.__version__,
              'attention': model.config._attn_implementation,
              'gradient_checkpointing': model.is_gradient_checkpointing,
              'gradient_checkpointing_use_reentrant': False,
              'nf4': model.config.quantization_config.to_dict(),
              'lora': str(model.peft_config), 'trainable_parameters': sum(p.numel() for p in trainable),
              'microbatch_size': 1, 'optimizer_every_microbatch': True,
              'production_gradient_accumulation_steps': config['research']['gradient_accumulation_steps'],
              'cases': []}
    probe = config.get('throughput_probe', {})
    warmups = probe.get('warmup_steps', 2)
    timed_steps = probe.get('timed_steps', 3)
    for total_length in probe.get('total_lengths', [4608, 8704]):
        prompt_length = total_length - 512
        ids, labels = synthetic_ids(tokenizer, prompt_length)
        item = {'prompt_tokens': prompt_length, 'target_tokens': 512, 'total_tokens': len(ids),
                'input_ids_sha256': training.value_hash(ids), 'label_sha256': training.value_hash(labels),
                'warmup_steps': warmups, 'timed_steps': timed_steps, 'measurements': [], 'status': 'running'}
        result['cases'].append(item)
        training.write_json(root / 'result.json', result)
        tensor = torch.tensor([ids], device='cuda')
        mask = torch.ones_like(tensor)
        targets = torch.tensor([labels], device='cuda')
        try:
            for step in range(warmups + timed_steps):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=tensor, attention_mask=mask, labels=targets, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite synthetic completion loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                measured = {'step': step, 'warmup': step < warmups, 'seconds': elapsed,
                            'sequence_tokens_per_second': len(ids) / elapsed,
                            'completion_ce': loss.detach().float().item(),
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                item['measurements'].append(measured)
                training.write_json(root / 'result.json', result)
                print(json.dumps({'prompt_tokens': prompt_length, **measured}), flush=True)
            timed = item['measurements'][warmups:]
            item.update(status='complete', median_seconds=statistics.median(x['seconds'] for x in timed),
                        mean_seconds=statistics.mean(x['seconds'] for x in timed),
                        peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in timed))
        except torch.cuda.OutOfMemoryError as exc:
            item.update(status='oom', error=str(exc))
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        training.write_json(root / 'result.json', result)
        print(json.dumps({'case_summary': {k: v for k, v in item.items() if k != 'measurements'}}), flush=True)
        del tensor, mask, targets
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    training.write_json(root / 'result.json', result)
    print(json.dumps({'probe_complete': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.root, json.loads(Path(args.config).read_text()))
