"""Isolated BF16-base LoRA research inference; NF4 SFT provenance stays explicit.

No model merging, downloads, outcome inputs, text truncation, or base substitution.
Production integration is separate: repairs must use this same backend identity.
"""
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import bam_training as training


def adapter_handle(adapter_hash):
    # This engine contains exactly one adapter. The content identity remains
    # the full SHA256, while vLLM V2 stores the process-local handle as int32.
    if adapter_hash is None:
        return None
    if len(adapter_hash) != 64 or any(c not in '0123456789abcdef' for c in adapter_hash):
        raise ValueError('Adapter requires a complete SHA256 identity')
    return 1


def validate_adapter_bindings(hashes):
    handles = {}
    for digest in hashes:
        handle = adapter_handle(digest)
        if handle in handles and handles[handle] != digest:
            raise ValueError('Different full adapter hashes collide on the batch handle')
        handles[handle] = digest
    if len(handles) > 1:
        raise ValueError('This backend accepts only one adapter identity per batch')
    return handles


def adapter_identity(path, expected_hash, config):
    path = Path(path)
    if any(p.is_symlink() or p.is_dir() for p in path.iterdir()):
        raise ValueError('Adapter must contain ordinary flat files only')
    files = {p.name: training.file_hash(p) for p in sorted(path.iterdir()) if p.is_file()}
    if training.value_hash(files) != expected_hash:
        raise ValueError('Exact adapter files changed')
    if not {'adapter_config.json', 'adapter_model.safetensors'} <= files.keys():
        raise ValueError('A saved safetensors LoRA adapter is required')
    spec = json.loads((path / 'adapter_config.json').read_text())
    options = config['research']
    if (spec.get('peft_type') != 'LORA' or spec.get('task_type') != 'CAUSAL_LM'
            or spec.get('base_model_name_or_path') != config['model_id']
            or spec.get('r') != options['lora_rank'] or spec.get('lora_alpha') != options['lora_alpha']
            or spec.get('bias') != 'none' or spec.get('modules_to_save') or spec.get('use_dora')
            or set(spec.get('target_modules', [])) != {'q_proj', 'k_proj', 'v_proj', 'o_proj'}):
        raise ValueError('Adapter architecture differs from the trained research contract')
    return {'sha256': expected_hash, 'files': files, 'rank': spec['r'], 'config': spec}


def validate_lineage(rows, provenance, config, adapter, *, training_manifest=None, diagnostic=False):
    training.validate_research_rows(rows, targets=False)
    if provenance.get('base_model') != training._model_config(config):
        raise ValueError('Research base checkpoint differs')
    pool = provenance.get('training_game_ids'); latest = provenance.get('training_max_date')
    if not isinstance(pool, list) or len(set(pool)) != len(pool): raise ValueError('Training pool required')
    if diagnostic:
        if (provenance.get('mode') not in ('synthetic_zero_adapter', 'synthetic_trained_adapter') or pool or latest is not None
                or not all(r.get('synthetic_only') is True for r in rows)):
            raise ValueError('Diagnostic outputs cannot masquerade as fitted NBA research')
    elif provenance.get('mode') == 'cold_prompted':
        if adapter is not None or pool or latest is not None: raise ValueError('Invalid cold-fold lineage')
    elif provenance.get('mode') in ('expanding_fold', 'final_train'):
        if not adapter or not training_manifest or not pool or not latest:
            raise ValueError('Fitted adapter and completed training manifest required')
        for field in ('training_game_ids', 'training_max_date', 'base_model'):
            if training_manifest.get(field) != provenance.get(field): raise ValueError('Training lineage differs')
        if training_manifest.get('adapter_files_sha256') != adapter['sha256']:
            raise ValueError('Adapter differs from completed fit')
    else: raise ValueError('Unknown researcher mode')
    if provenance.get('adapter_sha256') != (adapter['sha256'] if adapter else None):
        raise ValueError('Adapter lineage hash differs')
    for row in rows:
        if row['game_id'] in pool or (latest is not None and latest >= row['game_date_et']):
            raise ValueError('Researcher trained on target or same/future game date')
        if row['split'] != 'train' and row['game_date_et'] < config['minimum_eval_date']:
            raise ValueError('Heldout game predates checkpoint cutoff')


def normalize_output(output, expected_ids, eos_ids, budget, adapter_id):
    """Accept only documented token EOS, not a guessed completion or stop string."""
    if list(output.prompt_token_ids) != expected_ids: raise ValueError('Backend changed input token IDs')
    used = getattr(output, 'lora_request', None)
    if (getattr(used, 'lora_int_id', None) if used else None) != adapter_id:
        raise ValueError('Backend used a different adapter or silently omitted it')
    if len(output.outputs) != 1: raise ValueError('Exactly one greedy completion required')
    completion = output.outputs[0]; tokens = list(completion.token_ids)
    if any(type(t) is not int for t in tokens) or len(tokens) > budget:
        raise ValueError('Invalid backend completion token count')
    raw = {'raw_token_ids': tokens, 'raw_text': completion.text, 'output_tokens': len(tokens),
           'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason}
    eos = (tokens[-1] if tokens and tokens[-1] in eos_ids else
           completion.stop_reason if type(completion.stop_reason) is int and completion.stop_reason in eos_ids else None)
    if output.finished and completion.finish_reason == 'stop' and eos is not None:
        return {**raw, 'status': 'complete', 'verified_eos_token_id': eos}
    return {**raw, 'status': 'incomplete', 'error': 'No verified natural EOS; raw incomplete output retained'}


class StudentVLLM:
    def __init__(self, config, model_manifest, *, adapter_path=None, expected_adapter_hash=None):
        # CUDA/library imports and caller-side CPU tensor inspection must not
        # leak initialized thread pools into a forked engine process.
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        from transformers import AutoTokenizer, GenerationConfig
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self.config = config; self.sampling_class = SamplingParams
        spec = model_manifest['model_spec']
        if (spec.get('model_id') != config['model_id'] or spec.get('model_revision') != config['model_revision']
                or spec.get('quantization')):
            raise ValueError('Use the same pinned unquantized HF base checkpoint')
        model_path = Path(model_manifest['snapshot_path'])
        for name, expected in model_manifest['files'].items():
            if training.file_hash(model_path / name) != expected: raise ValueError('Pinned model/tokenizer file changed')
        self.adapter = adapter_identity(adapter_path, expected_adapter_hash, config) if adapter_path else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        generation = GenerationConfig.from_pretrained(model_path, local_files_only=True)
        self.eos_ids = generation.eos_token_id if isinstance(generation.eos_token_id, list) else [generation.eos_token_id]
        if not self.eos_ids or any(type(t) is not int for t in self.eos_ids): raise ValueError('Explicit pinned EOS IDs required')
        options = config.get('student_vllm', {})
        self.max_context = options.get('max_model_len', 32768)
        settings = {'dtype': 'bfloat16', 'quantization': None, 'tensor_parallel_size': 1,
                    'max_model_len': self.max_context, 'max_num_seqs': options.get('max_num_seqs', 12),
                    'max_num_batched_tokens': options.get('max_num_batched_tokens', 8192),
                    'gpu_memory_utilization': options.get('gpu_memory_utilization', .85),
                    'enable_lora': True, 'max_loras': 1, 'max_lora_rank': config['research']['lora_rank'],
                    'lora_dtype': 'bfloat16', 'generation_config': 'vllm', 'seed': config.get('seed', 230),
                    'enable_chunked_prefill': True, 'enable_prefix_caching': True,
                    'enforce_eager': True, 'trust_remote_code': False}
        self.identity = {'backend': 'vllm_student_lora', 'inference_base_precision': 'bfloat16',
                         'runtime_environment': {'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'},
                         'training_base_precision': 'bitsandbytes_nf4_double_quant_bfloat16_compute',
                         'model': model_manifest, 'adapter': self.adapter, 'settings': settings,
                         'eos_token_ids': self.eos_ids, 'worker_sha256': training.file_hash(__file__),
                         'training_helper_sha256': training.file_hash(training.__file__),
                         'versions': {p: version(p) for p in ('vllm', 'torch', 'transformers', 'tokenizers')},
                         'sampling': {'temperature': 0., 'top_p': 1., 'top_k': -1, 'ignore_eos': False},
                         'precision_change': 'NF4 SFT adapter applied to original BF16 base; no token-equivalence claim'}
        self.adapter_id = adapter_handle(expected_adapter_hash) if self.adapter else None
        self.lora = LoRARequest('research-' + expected_adapter_hash, self.adapter_id, str(adapter_path)) if self.adapter else None
        before = time.monotonic()
        self.llm = LLM(model=str(model_path), tokenizer=str(model_path), **settings)
        self.load_seconds = time.monotonic() - before

    def generate(self, items, budget, *, use_adapter=True):
        expected = self.adapter['sha256'] if use_adapter and self.adapter else None
        validate_adapter_bindings([expected] + [i.get('adapter_sha256', expected) for i in items])
        if type(budget) is not int or not 1 <= budget <= 8192: raise ValueError('Bounded output required')
        if any(not i['input_ids'] or len(i['input_ids']) + budget > self.max_context for i in items):
            raise ValueError('Full prompt plus output exceeds context; truncation prohibited')
        params = self.sampling_class(**self.identity['sampling'], max_tokens=budget,
                                     stop_token_ids=self.eos_ids, seed=self.config.get('seed', 230))
        before = time.monotonic()
        output = self.llm.generate([{'prompt_token_ids': i['input_ids']} for i in items], params,
                                   lora_request=self.lora if use_adapter else None, use_tqdm=False)
        if len(output) != len(items): raise ValueError('Backend omitted a request')
        elapsed = time.monotonic() - before
        records = []
        for item, result in zip(items, output):
            record = normalize_output(result, item['input_ids'], self.eos_ids, budget,
                                      self.adapter_id if use_adapter else None)
            record.update(request_sha256=item['request_sha256'], prompt_tokens=len(item['input_ids']),
                          output_budget=budget,
                          prompt_token_ids_sha256=training.value_hash(item['input_ids']),
                          batch_seconds=elapsed, batch_size=len(items), execution_sha256=training.value_hash(self.identity))
            # Decode actual returned tokens. Never manufacture an EOS token.
            record['text'] = self.tokenizer.decode(record['raw_token_ids'], skip_special_tokens=True) if record['status'] == 'complete' else ''
            records.append(record)
        return records


def validate_attempt_binding(record, item, backend, budget):
    expected = {'request_sha256': item['request_sha256'],
                'prompt_token_ids_sha256': training.value_hash(item['input_ids']),
                'execution_sha256': training.value_hash(backend.identity), 'output_budget': budget}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Generation attempt belongs to a different request, prompt, runtime or output budget')
    if record.get('status') not in ('complete', 'incomplete', 'context_limit'):
        raise ValueError('Unknown generation attempt status')


def generate_reports(root, rows, backend, provenance, config, *, training_manifest=None, diagnostic=False):
    """Per-fold durable outputs; current driver integration must remain explicit."""
    validate_lineage(rows, provenance, config, backend.adapter, training_manifest=training_manifest, diagnostic=diagnostic)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    budgets = training.research_output_budgets(config)
    inference = {'identity_sha256': training.value_hash(backend.identity),
                 'backend': 'vllm_student_lora', 'base_precision': 'bfloat16',
                 'training_base_precision': backend.identity['training_base_precision']}
    output_provenance = {**provenance, 'inference': inference}
    training._lock(root, {'rows_sha256': training.value_hash(rows), 'provenance': output_provenance,
                          'runtime_identity': backend.identity, 'output_budgets': budgets})
    items = []
    for row in rows:
        ids = training.prompt_tokens(backend.tokenizer, row['messages'], config['research'].get('max_seq_length', 16384))
        if row.get('expected_prompt_token_ids_sha256') not in (None, training.value_hash(ids)):
            raise ValueError('Pinned prompt token IDs differ from the reference contract')
        items.append({'input_ids': ids, 'request_sha256': training.value_hash(row)})
    completed, attempts = {}, {i['request_sha256']: [] for i in items}
    for budget in budgets:
        pending = []
        for item in items:
            key = item['request_sha256']
            if key in completed: continue
            path = root / 'attempts' / key / f'{budget}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['sha256'] != training.value_hash(saved['record']): raise ValueError('Cached generation attempt changed')
                record = saved['record']
                validate_attempt_binding(record, item, backend, budget)
                attempts[key].append({'path': str(path), 'sha256': saved['sha256']})
                if record['status'] == 'complete': completed[key] = record
            elif len(item['input_ids']) + budget > backend.max_context:
                record = {'status': 'context_limit', 'request_sha256': key, 'output_budget': budget,
                          'prompt_token_ids_sha256': training.value_hash(item['input_ids']),
                          'execution_sha256': training.value_hash(backend.identity),
                          'error': 'Full input retained; requested output cannot fit'}
                training.write_json(path, {'record': record, 'sha256': training.value_hash(record)})
                attempts[key].append({'path': str(path), 'sha256': training.value_hash(record)})
            else: pending.append(item)
        chunk_size = config.get('student_vllm', {}).get('durable_chunk_size', 4)
        if type(chunk_size) is not int or not 1 <= chunk_size <= 16:
            raise ValueError('Durable generation chunks must contain 1..16 requests')
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            records = backend.generate(chunk, budget)
            if len(records) != len(chunk): raise ValueError('Backend omitted generation requests')
            for item, record in zip(chunk, records):
                validate_attempt_binding(record, item, backend, budget)
                key = item['request_sha256']; path = root / 'attempts' / key / f'{budget}.json'
                training.write_json(path, {'record': record, 'sha256': training.value_hash(record)})
                attempts[key].append({'path': str(path), 'sha256': training.value_hash(record)})
                if record['status'] == 'complete': completed[key] = record
            print(json.dumps({'batch_complete': len(chunk), 'backend': 'vllm_student_lora'}), flush=True)
    outputs = [{**{k: v for k, v in row.items() if k != 'messages'},
                'report': completed[training.value_hash(row)]['text'], 'input_sha256': training.value_hash(row['messages']),
                'research_provenance': output_provenance, 'generation_attempts': attempts[training.value_hash(row)]}
               for row in rows if training.value_hash(row) in completed]
    failures = [i['request_sha256'] for i in items if i['request_sha256'] not in completed]
    training.write_rows(root / 'generated_modules.jsonl', outputs)
    manifest = {'sha256': training.file_hash(root / 'generated_modules.jsonl'), 'complete': not failures,
                'failures': failures, 'expected_modules': len(rows), 'modules': len(outputs),
                'synthetic_only': diagnostic, 'identity_sha256': training.value_hash(backend.identity)}
    training.write_json(root / 'generated_modules.manifest.json', manifest)
    return manifest
