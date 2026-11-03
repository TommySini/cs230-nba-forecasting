"""BAM research SFT and outcome forecasting workers; no CUDA imports at import time.

See reports/bam_pipeline/training_design.md for the artifact/API contract.
The same immutable HF checkpoint is loaded separately for research and forecasting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import shutil


FORBIDDEN_INPUT_KEYS = {'home_win', 'outcome', 'label', 'target', 'postmortem', 'recap', 'gold_report'}
_WORKER_DEADLINE = None


def checkpoint_helper_hash():
    import bam_checkpoint
    return file_hash(bam_checkpoint.__file__)


def _deadline_near():
    return _WORKER_DEADLINE is not None and time.monotonic() >= _WORKER_DEADLINE - 120


def _adapter_state(model):
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(model)


def _apply_adapter(model, state):
    from peft import set_peft_model_state_dict
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys:
        raise ValueError('Unexpected adapter checkpoint tensors')


def _checkpoint_identity(config, rows, encoded, tokenizer, optimizer, named_parameters, accumulation):
    import bam_checkpoint
    tokenizer_spec = {'vocab': tokenizer.get_vocab(), 'chat_template': tokenizer.chat_template,
                      'special_tokens': tokenizer.special_tokens_map}
    return {'model': _model_config(config), 'config_sha256': value_hash(config),
            'rows_sha256': value_hash(rows), 'tokenized_sha256': value_hash(encoded),
            'worker_sha256': file_hash(__file__), 'tokenizer_sha256': value_hash(tokenizer_spec),
            'row_count': len(rows), 'gradient_accumulation_steps': accumulation,
            'parameter_names': bam_checkpoint.optimizer_parameter_names(optimizer, named_parameters)}


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    temporary.replace(path)


def _identity(row):
    if not isinstance(row.get('game_id'), str) or not row['game_id']:
        raise ValueError('Missing game_id')
    if date.fromisoformat(row['game_date_et']).isoformat() != row['game_date_et']:
        raise ValueError('Noncanonical date')
    if row['split'] not in ('train', 'dev', 'test'):
        raise ValueError('Unknown split')


def _messages(messages, completion=False):
    roles = ['system', 'user', 'assistant'] if completion else ['system', 'user']
    if [m['role'] for m in messages] != roles:
        raise ValueError('Expected exactly ' + ','.join(roles))
    if any(not isinstance(m['content'], str) or not m['content'].strip() for m in messages):
        raise ValueError('Empty/nontext message')


def validate_groups(rows):
    """All modules/augmentations stay with their game; date blocks never cross splits."""
    games = {}
    dates = defaultdict(set)
    for row in rows:
        _identity(row)
        identity = (row['game_date_et'], row['split'])
        if games.setdefault(row['game_id'], identity) != identity:
            raise ValueError('Game crossed dates/splits')
        dates[row['split']].add(row['game_date_et'])
    ordered = [dates[s] for s in ('train', 'dev', 'test') if dates[s]]
    if any(max(left) >= min(right) for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Splits are not strictly chronological date groups')
    return games


def validate_research_rows(rows, *, targets):
    validate_groups(rows)
    keys = set()
    for row in rows:
        key = (row['game_id'], row['module'], row.get('unit_id', ''))
        if key in keys:
            raise ValueError('Duplicate research module/unit')
        keys.add(key)
        _messages(row['messages'], completion=targets)
        if not row.get('source_hashes'):
            raise ValueError('Missing source provenance')
        if FORBIDDEN_INPUT_KEYS & row.keys():
            raise ValueError('Postgame fields in research record')
        if targets and (row['split'] != 'train' or row.get('review_status') != 'accepted'
                        or not row.get('teacher_hash')):
            raise ValueError('Research fitting requires accepted, train-only gold with teacher hash')
    return rows


def expanding_folds(inputs, targets, folds=3):
    """Contiguous date blocks; include research-only earlier games in each fitting pool."""
    validate_research_rows(inputs, targets=False)
    validate_research_rows(targets, targets=True)
    validate_groups(inputs + targets)
    train = [r for r in inputs if r['split'] == 'train']
    dates = sorted({r['game_date_et'] for r in train})
    if not dates or folds < 1:
        raise ValueError('Need train inputs and positive fold count')
    blocks = min(folds, len(dates))
    plans = []
    for i in range(blocks):
        block = dates[i * len(dates) // blocks:(i + 1) * len(dates) // blocks]
        fitting = [r for r in targets if r['game_date_et'] < block[0]]
        prediction = [r for r in train if r['game_date_et'] in block]
        plans.append({'fold': i, 'fit': fitting, 'predict': prediction,
                      'mode': 'expanding_fold' if fitting else 'cold_prompted'})
    heldout = [r for r in inputs if r['split'] != 'train']
    if heldout:
        plans.append({'fold': 'final', 'fit': targets, 'predict': heldout, 'mode': 'final_train'})
    return plans


def validate_prediction_rows(rows, minimum_eval_date=None):
    validate_groups(rows)
    if len({r['game_id'] for r in rows}) != len(rows):
        raise ValueError('Forecasting must have one row per independent game')
    for row in rows:
        _messages(row['messages'])
        if FORBIDDEN_INPUT_KEYS & row.keys():
            raise ValueError('Outcome/gold fields in predictor inputs')
        if row.get('forecast_eligible', True) is not True:
            raise ValueError('Research-only game entered forecast cohort')
        if not 0 < row['p_boost'] < 1:
            raise ValueError('Invalid original booster probability')
        booster = row['booster_provenance']
        if not booster.get('training_max_date') or booster['training_max_date'] >= row['game_date_et']:
            raise ValueError('Booster was not fit strictly before target date')
        provenance = row['research_provenance']
        components = provenance if isinstance(provenance, list) else [provenance]
        if not components:
            raise ValueError('Missing research provenance')
        for item in components:
            if row['game_id'] in item['training_game_ids']:
                raise ValueError('Student saw target-game gold')
            latest = item['training_max_date']
            if latest is not None and latest >= row['game_date_et']:
                raise ValueError('Student saw future or same-date gold')
            if item['mode'] == 'cold_prompted':
                if item['training_game_ids'] or latest is not None:
                    raise ValueError('Cold student contains training games')
            elif (item['mode'] not in ('expanding_fold', 'final_train')
                  or not item['training_game_ids'] or latest is None or not item.get('adapter_sha256')):
                raise ValueError('Invalid student training provenance')
            if row['split'] == 'train' and item['mode'] == 'final_train':
                raise ValueError('Final researcher used to predict training game')
        if minimum_eval_date and row['split'] != 'train' and row['game_date_et'] < minimum_eval_date:
            raise ValueError('Heldout game predates prediction-model eligibility cutoff')
    return rows


def chat_token_ids(tokenizer, messages, *, generation_prompt):
    """An explicit flat list is required across Transformers 4.x and 5.x."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                       add_generation_prompt=generation_prompt, return_dict=False)
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError('Chat tokenizer must return a nonempty flat list of integer token IDs')
    return ids


def completion_tokens(tokenizer, messages, max_length):
    """Mask every pregame prompt token, preserving all complete source/target tokens."""
    _messages(messages, completion=True)
    prefix = chat_token_ids(tokenizer, messages[:-1], generation_prompt=True)
    full = chat_token_ids(tokenizer, messages, generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template is not prefix-stable; cannot safely mask completion')
    if len(full) > max_length:
        raise ValueError(f'Complete example has {len(full)} tokens, limit {max_length}; chunk upstream')
    if len(full) <= len(prefix):
        raise ValueError('Empty completion')
    return {'input_ids': full, 'labels': [-100] * len(prefix) + full[len(prefix):],
            'prefix_index': len(prefix) - 1}


def prompt_tokens(tokenizer, messages, max_length):
    _messages(messages)
    ids = chat_token_ids(tokenizer, messages, generation_prompt=True)
    if not ids or len(ids) > max_length:
        raise ValueError('Complete input exceeds token budget; chunk upstream, never silently truncate')
    return ids


def _model_config(config):
    if not re.fullmatch('[0-9a-f]{40}', config.get('model_revision', '')):
        raise ValueError('An immutable 40-character HF revision is required')
    if not config.get('model_id') or not config.get('minimum_eval_date'):
        raise ValueError('Require model_id and minimum_eval_date')
    return {k: config[k] for k in ('model_id', 'model_revision', 'minimum_eval_date')}


def _load_model(config, *, training=False, adapter=None, stage='research'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    _model_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError('These workers require CUDA; use pure validation functions locally')
    torch.manual_seed(config.get('seed', 230))
    torch.cuda.manual_seed_all(config.get('seed', 230))
    kwargs = {'revision': config['model_revision'], 'cache_dir': config.get('cache_dir'),
              'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], **kwargs, torch_dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = not training
    if training:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={'use_reentrant': False})
        stage_options = config.get(stage, {})
        options = config.get('lora', {}) | {
            'rank': stage_options.get('lora_rank', config.get('lora', {}).get('rank', 16)),
            'alpha': stage_options.get('lora_alpha', config.get('lora', {}).get('alpha', 32)),
            'dropout': stage_options.get('lora_dropout', config.get('lora', {}).get('dropout', .05)),
        }
        model = get_peft_model(model, LoraConfig(
            r=options.get('rank', 16), lora_alpha=options.get('alpha', 32),
            lora_dropout=options.get('dropout', .05), bias='none', task_type='CAUSAL_LM',
            target_modules=options.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])))
    elif adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    if not training:
        model.eval()
    return model, tokenizer


def _release(model):
    # Caller must drop its own reference before empty_cache can reclaim model storage.
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _lock(directory, specification):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'request.json'
    if path.exists() and json.loads(path.read_text()) != specification:
        raise ValueError('Existing run belongs to different config/data; choose a new run directory')
    write_json(path, specification)


def _adapter_hash(directory):
    return value_hash({p.name: file_hash(p) for p in sorted(Path(directory).glob('*')) if p.is_file()})


def _research_steps(model, tokenizer, rows, encoded, destination, config, *, stop_after_steps=None):
    """Unchanged research objective with resumable optimizer-group boundaries."""
    import torch
    import bam_checkpoint
    options = config.get('research', {})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=options.get('learning_rate', 1e-4), weight_decay=.01)
    accumulation = int(options.get('gradient_accumulation_steps', 8))
    epochs = int(options.get('epochs', 1))
    if accumulation < 1 or epochs < 1:
        raise ValueError('Positive accumulation/epochs required')
    identity = _checkpoint_identity(config, rows, encoded, tokenizer, optimizer, model.named_parameters(), accumulation)
    checkpoint_root = Path(destination) / 'checkpoints'
    rng = random.Random(config.get('seed', 230))
    resumed = bam_checkpoint.restore_checkpoint(checkpoint_root, identity=identity, adapter_state=_adapter_state(model),
        optimizer=optimizer, named_parameters=model.named_parameters(), shuffle_rng=rng,
        apply_adapter=lambda state: _apply_adapter(model, state))
    cursor = resumed['cursor'] if resumed else None
    history = list(cursor['history']) if cursor else []
    global_step = cursor['global_step'] if cursor else 0
    starting_epoch = cursor['epoch'] if cursor else 0
    last_saved = time.monotonic()
    device = trainable[0].device
    model.train()
    for epoch in range(starting_epoch, epochs):
        continuing = cursor is not None and epoch == starting_epoch
        order = list(cursor['order']) if continuing else list(range(len(encoded)))
        if not continuing: rng.shuffle(order)
        total = cursor['epoch_loss_sum'] if continuing else 0.
        offset = cursor['next_group_start'] if continuing else 0
        for start in range(offset, len(order), accumulation):
            group = order[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                item = encoded[index]
                ids = torch.tensor([item['input_ids']], device=device)
                labels = torch.tensor([item['labels']], device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
                if not torch.isfinite(loss): raise ValueError('Nonfinite research loss')
                (loss / len(group)).backward()
                total += loss.detach().float().item()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            global_step += 1
            following = start + len(group)
            interrupted = stop_after_steps is not None and global_step >= stop_after_steps
            budget_stop = _deadline_near()
            if following == len(order) or interrupted or budget_stop or bam_checkpoint.checkpoint_due(last_saved):
                state = {'epoch': epoch, 'next_group_start': following, 'global_step': global_step,
                         'order': order, 'epoch_seen': following, 'epoch_loss_sum': total, 'history': history}
                bam_checkpoint.save_checkpoint(checkpoint_root, identity=identity, cursor=state,
                    adapter_state=_adapter_state(model), optimizer=optimizer,
                    named_parameters=model.named_parameters(), shuffle_rng=rng)
                last_saved = time.monotonic()
            if interrupted or budget_stop:
                raise RuntimeError('Research optimizer checkpointed; resume the same immutable fold')
        history.append({'epoch': epoch + 1, 'mean_completion_ce': total / len(encoded)})
        print(json.dumps(history[-1]), flush=True)
    return {'history': history, 'optimizer_steps': global_step,
            'trainable_parameters': sum(p.numel() for p in trainable), 'checkpoint_identity': identity}


def _fit_research(rows, destination, config):
    """Small explicit optimizer loop avoids Trainer/API coupling and prompt loss."""
    import torch
    validate_research_rows(rows, targets=True)
    if not rows:
        raise ValueError('No accepted research targets')
    options = config.get('research', {})
    specification = {'model': _model_config(config), 'config': config, 'rows_sha256': value_hash(rows),
                     'worker_sha256': file_hash(__file__), 'checkpoint_helper_sha256': checkpoint_helper_hash()}
    _lock(destination, specification)
    done = Path(destination) / 'training.json'
    if done.exists():
        result = json.loads(done.read_text())
        if result['adapter_files_sha256'] != _adapter_hash(Path(destination) / 'adapter'):
            raise ValueError('Saved research adapter changed')
        return result
    model, tokenizer = _load_model(config, training=True)
    encoded = [completion_tokens(tokenizer, r['messages'], options.get('max_seq_length', 8192)) for r in rows]
    fitted = _research_steps(model, tokenizer, rows, encoded, destination, config)
    adapter_path = Path(destination) / 'adapter'
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    result = {'rows': len(rows), 'training_game_ids': sorted({r['game_id'] for r in rows}),
              'training_max_date': max(r['game_date_et'] for r in rows), 'history': fitted['history'],
              'adapter_files_sha256': _adapter_hash(adapter_path), 'base_model': _model_config(config),
              'trainable_parameters': fitted['trainable_parameters'], 'optimizer_steps': fitted['optimizer_steps'],
              'checkpoint_helper_sha256': checkpoint_helper_hash(),
              'loss': 'completion-only token cross entropy; no outcome labels'}
    write_json(done, result)
    print(json.dumps({'research_fit_complete': str(done), 'adapter_sha256': result['adapter_files_sha256']}), flush=True)
    del model
    _release(None)
    return result


def train_research(run_dir, config):
    """Fit final researcher only on accepted train targets, including research-only games."""
    root = Path(run_dir)
    return _fit_research(read_rows(root / 'research_examples.jsonl'), root / 'research/final', config)


def research_output_budgets(config):
    options = config.get('research', {})
    first = options.get('max_new_tokens', 4096)
    budgets = options.get('output_token_budgets', [first])
    if (not isinstance(budgets, list) or not 1 <= len(budgets) <= 2 or budgets[0] != first
            or any(type(n) is not int or not 1 <= n <= 8192 for n in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError('Declare one or two increasing research output budgets, starting with max_new_tokens, at most8192')
    return budgets


def bounded_research_completions(items, provenance, config, directory, *, max_context, eos_ids, generate_batch, decode):
    """Preserve all raw attempts; retry length only with the identical source/adapter.

    No GPU dependency here: the caller supplies one already-loaded adapter's
    generation function. Successful siblings survive incomplete/context failures.
    """
    budgets = research_output_budgets(config)
    if not eos_ids or any(type(token) is not int for token in eos_ids):
        raise ValueError('Explicit EOS token IDs required')
    if len({item['request_sha256'] for item in items}) != len(items):
        raise ValueError('Duplicate bounded generation request')
    completed, attempts, failures, terminal = {}, {item['request_sha256']: [] for item in items}, [], set()
    directory = Path(directory)
    for item in items:
        _lock(directory / item['request_sha256'], {
            'input_sha256': item['input_sha256'], 'input_token_ids_sha256': value_hash(item['input_ids']),
            'research_provenance': provenance, 'output_token_budgets': budgets, 'max_context': max_context})
    for attempt, budget in enumerate(budgets):
        pending = []
        for item in items:
            key = item['request_sha256']
            if key in completed or key in terminal: continue
            path = directory / key / f'attempt_{attempt + 1}_{budget}.json'
            if path.exists():
                saved = json.loads(path.read_text()); record = saved['record']
                if saved['sha256'] != value_hash(record) or record['request_sha256'] != key or record['max_new_tokens'] != budget:
                    raise ValueError('Saved research generation attempt changed')
                attempts[key].append({'path': str(path), 'sha256': saved['sha256'], 'status': record['status']})
                if record['status'] == 'complete': completed[key] = {'text': record['text'], 'generation_attempts': attempts[key]}
                if record['status'] == 'runtime_error': terminal.add(key)
                continue
            record = {'request_sha256': key, 'input_sha256': item['input_sha256'],
                      'prompt_token_ids_sha256': value_hash(item['input_ids']), 'prompt_tokens': len(item['input_ids']),
                      'research_provenance': provenance, 'attempt': attempt + 1, 'max_new_tokens': budget,
                      'raw_token_ids': [], 'raw_text': '', 'text': '', 'finish_reason': None}
            if len(item['input_ids']) + budget > max_context:
                record.update(status='context_limit', error='Full prompt plus declared output budget exceeds context; input unchanged')
                write_json(path, {'record': record, 'sha256': value_hash(record)})
                attempts[key].append({'path': str(path), 'sha256': value_hash(record), 'status': record['status']})
                print(json.dumps({'generation_attempt_complete': key, 'status': record['status'], 'budget': budget}), flush=True)
            else:
                pending.append((item, path, record))
        if not pending: continue
        try:
            outputs = generate_batch([item for item, _, _ in pending], budget)
            if len(outputs) != len(pending): raise ValueError('Generation batch returned wrong output count')
            error = None
        except Exception as exc:
            outputs = [[] for _ in pending]; error = f'{type(exc).__name__}: {exc}'
        for (item, path, record), tokens in zip(pending, outputs):
            key = item['request_sha256']
            if not isinstance(tokens, list) or any(type(token) is not int for token in tokens) or len(tokens) > budget:
                raise ValueError('Invalid generated token list')
            stop = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
            record.update(raw_token_ids=tokens, raw_text=decode(tokens), output_tokens=len(tokens))
            if error:
                record.update(status='runtime_error', error=error)
                terminal.add(key)
            elif stop is None:
                record.update(status='token_limit', finish_reason='length', error='No EOS; incomplete report retained and rejected')
            else:
                record.update(status='complete', finish_reason='stop', text=decode(tokens[:stop + 1]))
            write_json(path, {'record': record, 'sha256': value_hash(record)})
            attempts[key].append({'path': str(path), 'sha256': value_hash(record), 'status': record['status']})
            if record['status'] == 'complete': completed[key] = {'text': record['text'], 'generation_attempts': attempts[key]}
        print(json.dumps({'generation_attempt_complete': len(pending), 'budget': budget}), flush=True)
    for item in items:
        key = item['request_sha256']
        if key not in completed: failures.append({'request_sha256': key, 'attempts': attempts[key]})
    return {'completed': completed, 'failures': failures}


def _transformer_completion_batch(model, tokenizer, items, budget):
    import torch
    width = max(len(item['input_ids']) for item in items)
    device = next(model.parameters()).device
    tensor = torch.tensor([[tokenizer.pad_token_id] * (width-len(item['input_ids'])) + item['input_ids'] for item in items], device=device)
    attention = torch.tensor([[0] * (width-len(item['input_ids'])) + [1] * len(item['input_ids']) for item in items], device=device)
    with torch.inference_mode():
        full = model.generate(input_ids=tensor, attention_mask=attention, max_new_tokens=budget,
                              do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return [row[width:].tolist() for row in full]


def generate_research(run_dir, config):
    """Generate every training report strictly out of time, then final-fit dev/test reports."""
    import torch
    # Fold selection controls bounded execution, not the scientific configuration.
    config = dict(config)
    config['research'] = dict(config.get('research', {}))
    requested_folds = config['research'].pop('fold_ids', None)
    root = Path(run_dir)
    inputs = read_rows(root / 'research_inputs.jsonl')
    targets = read_rows(root / 'research_examples.jsonl')
    plans = expanding_folds(inputs, targets, config.get('research', {}).get('folds', 3))
    if any(r['split'] != 'train' and r['game_date_et'] < config['minimum_eval_date'] for r in inputs):
        raise ValueError('Prediction-time research game predates checkpoint eligibility cutoff')
    gold_by_key = {(r['game_id'], r['module'], r.get('unit_id', '')): r for r in targets}
    for row in inputs:
        gold = gold_by_key.get((row['game_id'], row['module'], row.get('unit_id', '')))
        if gold and (gold['messages'][:-1] != row['messages'] or gold['source_hashes'] != row['source_hashes']):
            raise ValueError('Gold supervision and generation source packets differ')
    mode = config.get('research', {}).get('mode', 'trained')
    if mode not in ('trained', 'prompted'):
        raise ValueError('Unknown research mode')
    if mode == 'prompted':
        for plan in plans:
            plan.update(fit=[], mode='cold_prompted')
    all_fold_ids = [plan['fold'] for plan in plans]
    if requested_folds is not None:
        if not requested_folds or len(set(map(str, requested_folds))) != len(requested_folds):
            raise ValueError('fold_ids must be a nonempty unique list')
        if not set(map(str, requested_folds)) <= set(map(str, all_fold_ids)):
            raise ValueError('Unknown requested research fold')
        plans = [p for p in plans if str(p['fold']) in set(map(str, requested_folds))]
    _lock(root / 'research/generation', {'config': config, 'inputs': value_hash(inputs),
          'targets': value_hash(targets), 'worker_sha256': file_hash(__file__),
          'checkpoint_helper_sha256': checkpoint_helper_hash()})
    aggregate_path = root / 'generated_modules.jsonl'
    if aggregate_path.exists():
        if json.loads(aggregate_path.with_suffix('.manifest.json').read_text())['sha256'] != file_hash(aggregate_path):
            raise ValueError('Accumulated research output changed')
        generated = read_rows(aggregate_path)
    else:
        generated = []
    previous = {(r['game_id'], r['module'], r.get('unit_id', '')): r for r in generated}
    generated = []
    options = config.get('research', {})
    for plan in plans:
        directory = root / 'research' / ('final' if plan['fold'] == 'final' else f"fold_{plan['fold']}")
        if plan['fit']:
            trained = _fit_research(plan['fit'], directory, config)
            adapter = directory / 'adapter'
            provenance = {k: trained[k] for k in ('training_game_ids', 'training_max_date')}
            provenance['adapter_sha256'] = trained['adapter_files_sha256']
        else:
            adapter = None
            provenance = {'training_game_ids': [], 'training_max_date': None, 'adapter_sha256': None}
        provenance.update(mode=plan['mode'], base_model=_model_config(config))
        output_path = directory / 'generated.jsonl'
        if output_path.exists():
            if json.loads(output_path.with_suffix('.manifest.json').read_text())['sha256'] != file_hash(output_path):
                raise ValueError('Cached generated report changed')
            cached = read_rows(output_path)
            expected = [(r['game_id'], r['module'], r.get('unit_id', '')) for r in plan['predict']]
            actual = [(r['game_id'], r['module'], r.get('unit_id', '')) for r in cached]
            if expected != actual or any(r['research_provenance'] != provenance for r in cached):
                raise ValueError('Stale fold output')
            generated.extend(cached)
            continue
        output_by_key, pending = {}, []
        for row in plan['predict']:
            request_hash = value_hash(row)
            item_path = directory / 'module_cache' / (request_hash + '.json')
            if item_path.exists():
                saved = json.loads(item_path.read_text())
                if (saved['request_sha256'] != request_hash or saved['output_sha256'] != value_hash(saved['output'])
                        or saved['output']['research_provenance'] != provenance):
                    raise ValueError('Cached student module content/provenance changed')
                output_by_key[request_hash] = saved['output']
            else:
                pending.append(row)
        model = tokenizer = None
        if pending:
            model, tokenizer = _load_model(config, adapter=adapter)
        batch_size = int(options.get('generation_batch_size', 2))
        if batch_size < 1:
            raise ValueError('Positive generation batch size required')
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            encoded = [prompt_tokens(tokenizer, r['messages'], options.get('max_seq_length', 8192)) for r in batch]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            items = [{'request_sha256': value_hash(row), 'input_ids': ids,
                      'input_sha256': value_hash(row['messages'])} for row, ids in zip(batch, encoded)]
            result = bounded_research_completions(items, provenance, config, directory / 'generation_attempts',
                max_context=model.config.max_position_embeddings, eos_ids=eos,
                generate_batch=lambda requests, budget: _transformer_completion_batch(model, tokenizer, requests, budget),
                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
            for row in batch:
                completion = result['completed'].get(value_hash(row))
                if completion is None: continue
                generated_row = {k: v for k, v in row.items() if k != 'messages'} | {
                    'report': completion['text'], 'generation_attempts': completion['generation_attempts'],
                    'input_sha256': value_hash(row['messages']), 'research_provenance': provenance}
                request_hash = value_hash(row)
                output_by_key[request_hash] = generated_row
                write_json(directory / 'module_cache' / (request_hash + '.json'), {
                    'request_sha256': request_hash, 'output': generated_row, 'output_sha256': value_hash(generated_row)})
            print(json.dumps({'batch_complete': len(result['completed']), 'failures': result['failures']}), flush=True)
            if result['failures']:
                raise ValueError('Student generation exhausted declared budgets; durable raw attempts retained')
            if _deadline_near():
                raise RuntimeError('Generation progress saved; resume the same immutable fold')
        output = [output_by_key[value_hash(row)] for row in plan['predict']]
        write_rows(output_path, output)
        write_json(output_path.with_suffix('.manifest.json'), {'sha256': file_hash(output_path)})
        generated.extend(output)
        del model
        _release(None)
    for row in generated:
        key = (row['game_id'], row['module'], row.get('unit_id', ''))
        if key in previous and previous[key] != row:
            raise ValueError('Earlier generated module changed across fold invocations')
        previous[key] = row
    ordered_keys = [(r['game_id'], r['module'], r.get('unit_id', '')) for r in inputs]
    if not set(previous) <= set(ordered_keys):
        raise ValueError('Accumulated generation includes a foreign request')
    generated = [previous[key] for key in ordered_keys if key in previous]
    write_rows(root / 'generated_modules.jsonl', generated)
    write_json(aggregate_path.with_suffix('.manifest.json'), {'sha256': file_hash(aggregate_path)})
    return {'modules': len(generated), 'executed_folds': [p['fold'] for p in plans],
            'complete': len(generated) == len(inputs), 'expected_modules': len(inputs),
            'sha256': file_hash(root / 'generated_modules.jsonl')}


def _forecast_rows(root, config):
    return validate_prediction_rows(read_rows(Path(root) / 'prediction_inputs.jsonl'), config['minimum_eval_date'])


def validate_student_repair(request, original_input, initial_output, previous_output):
    """Repair may change wording, never the source packet, adapter, or temporal lineage."""
    attempt=request.get('attempt')
    if type(attempt) is not int or not 1<=attempt<=2:
        raise ValueError('Student repair is limited to two attempts')
    key=lambda row:(row['game_id'],row['module'])
    if not key(request)==key(original_input)==key(initial_output)==key(previous_output):
        raise ValueError('Repair crosses game/module boundaries')
    if any(request[field]!=original_input[field] or initial_output[field]!=original_input[field]
           or previous_output[field]!=original_input[field] for field in ('game_date_et','split')):
        raise ValueError('Repair changes the original game date or split')
    if (request['messages']!=original_input['messages'] or request['source_hashes']!=original_input['source_hashes']
            or request['original_input_sha256']!=value_hash(original_input['messages'])
            or initial_output['input_sha256']!=request['original_input_sha256']):
        raise ValueError('Repair source prompt/provenance changed')
    provenance=initial_output['research_provenance']
    if request['research_provenance']!=provenance or previous_output['research_provenance']!=provenance:
        raise ValueError('Repair would switch research adapters or training lineage')
    if (request['parent_output_sha256']!=value_hash(previous_output)
            or request['previous_response']!=previous_output['report']
            or request.get('repair_history',[])!=previous_output.get('repair_history',[])
            or len(previous_output.get('repair_history',[]))!=attempt-1):
        raise ValueError('Repair does not extend the preserved preceding student output')
    if request['game_id'] in provenance['training_game_ids']:
        raise ValueError('Student repair researcher saw target-game gold')
    if provenance['training_max_date'] is not None and provenance['training_max_date']>=original_input['game_date_et']:
        raise ValueError('Student repair researcher saw future gold')
    if not isinstance(request.get('feedback'),dict) or not request['feedback'].get('errors'):
        raise ValueError('Source-based rejection feedback required')
    if FORBIDDEN_INPUT_KEYS & request.keys():
        raise ValueError('Outcome/gold fields are forbidden in student repair requests')
    _messages(request['messages'])
    return request


def repair_research(run_dir,config):
    """Bounded source-only regeneration using each original fold's exact saved adapter."""
    import torch
    root=Path(run_dir)
    config=dict(config);config['research']=dict(config.get('research',{}))
    batch_id=config['research'].pop('repair_batch_id',None)
    if not isinstance(batch_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',batch_id):
        raise ValueError('An explicit safe research.repair_batch_id is required')
    requests=read_rows(root/'student_repair_requests'/(batch_id+'.jsonl'))
    if not requests or len({(r['game_id'],r['module']) for r in requests})!=len(requests):
        raise ValueError('Repair batch must contain unique game/modules')
    inputs={(r['game_id'],r['module']):r for r in read_rows(root/'research_inputs.jsonl')}
    aggregate=root/'generated_modules.jsonl'
    if json.loads(aggregate.with_suffix('.manifest.json').read_text())['sha256']!=file_hash(aggregate):
        raise ValueError('Initial student output changed')
    initial={(r['game_id'],r['module']):r for r in read_rows(aggregate)}
    known={value_hash(row):row for row in initial.values()}
    for path in sorted((root/'research/repairs').glob('*/outputs.jsonl')):
        if json.loads(path.with_suffix('.manifest.json').read_text())['sha256']!=file_hash(path):
            raise ValueError('Earlier student repair output changed')
        known.update({value_hash(row):row for row in read_rows(path)})
    for request in requests:
        key=(request['game_id'],request['module'])
        if request['parent_output_sha256'] not in known:
            raise ValueError('Repair parent is not a preserved student output')
        validate_student_repair(request,inputs[key],initial[key],known[request['parent_output_sha256']])
    directory=root/'research/repairs'/batch_id
    _lock(directory,{'config':config,'requests_sha256':value_hash(requests),'worker_sha256':file_hash(__file__)})
    output_path=directory/'outputs.jsonl'
    if output_path.exists():
        if json.loads(output_path.with_suffix('.manifest.json').read_text())['sha256']!=file_hash(output_path):
            raise ValueError('Completed repair batch changed')
        return {'repair_batch_id':batch_id,'modules':len(requests),'sha256':file_hash(output_path)}
    groups=defaultdict(list)
    for request in requests:
        groups[value_hash(request['research_provenance'])].append(request)
    output_map={}
    for group in groups.values():
        provenance=group[0]['research_provenance'];adapter=None
        if provenance.get('base_model')!=_model_config(config):
            raise ValueError('Repair base checkpoint differs from the original researcher')
        if provenance['mode']!='cold_prompted':
            matches=[]
            for path in (root/'research').glob('*/training.json'):
                manifest=json.loads(path.read_text())
                if (manifest['adapter_files_sha256']==provenance['adapter_sha256']
                        and manifest['training_game_ids']==provenance['training_game_ids']
                        and manifest['training_max_date']==provenance['training_max_date']):
                    matches.append(path.parent/'adapter')
            if not matches or any(_adapter_hash(path)!=provenance['adapter_sha256'] for path in matches):
                raise ValueError('Exact original fold adapter unavailable or modified')
            adapter=matches[0]
        elif provenance['training_game_ids'] or provenance['adapter_sha256'] is not None:
            raise ValueError('Cold-prompted repair has incompatible adapter lineage')
        pending=[]
        for request in group:
            cache=directory/'modules'/(value_hash(request)+'.json')
            if cache.exists():
                saved=json.loads(cache.read_text())
                if saved['sha256']!=value_hash(saved['output']): raise ValueError('Partial repair cache changed')
                output_map[(request['game_id'],request['module'])]=saved['output']
            else: pending.append(request)
        if not pending: continue
        model,tokenizer=_load_model(config,adapter=adapter)
        for request in pending:
            feedback={'instruction':'Repair your preceding report using only the original pregame sources. Correct rejected claims/citations, preserve opposing evidence and uncertainty. Do not use outside knowledge, target-game results, gold reports or a different researcher. Return the original requested complete JSON schema.',
                      'previous_student_report':request['previous_response'],'source_review_feedback':request['feedback']}
            messages=request['messages']+[{'role':'user','content':json.dumps(feedback,ensure_ascii=False)}]
            ids=chat_token_ids(tokenizer,messages,generation_prompt=True)
            if len(ids)>config['research'].get('max_seq_length',16384):
                raise ValueError('Complete student repair exceeds context; no source/feedback truncation permitted')
            eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            item={'request_sha256':value_hash(request),'input_sha256':value_hash(messages),'input_ids':ids}
            result=bounded_research_completions([item],provenance,config,directory/'generation_attempts',
                max_context=model.config.max_position_embeddings,eos_ids=eos,
                generate_batch=lambda requests,budget:_transformer_completion_batch(model,tokenizer,requests,budget),
                decode=lambda tokens:tokenizer.decode(tokens,skip_special_tokens=True))
            if result['failures']:
                raise ValueError('Student repair exhausted declared budgets; durable raw attempts retained')
            completion=result['completed'][value_hash(request)]
            parent=known[request['parent_output_sha256']]
            output={**parent,'report':completion['text'],'generation_attempts':completion['generation_attempts'],
                'repair_history':request.get('repair_history',[])+[{'attempt':request['attempt'],
                    'repair_batch_id':batch_id,'request_sha256':value_hash(request),
                    'parent_output_sha256':request['parent_output_sha256'],'feedback_sha256':value_hash(request['feedback']),
                    'repair_messages_sha256':value_hash(messages),'prompt_token_ids_sha256':value_hash(ids)}]}
            write_json(directory/'modules'/(value_hash(request)+'.json'),{'output':output,'sha256':value_hash(output)})
            output_map[(request['game_id'],request['module'])]=output
            print(json.dumps({'batch_complete':1,'student_repair':request['game_id'],'attempt':request['attempt']}),flush=True)
            if _deadline_near(): raise RuntimeError('Repair progress saved; resume the same immutable batch')
        del model
        _release(None)
    outputs=[output_map[(r['game_id'],r['module'])] for r in requests]
    write_rows(output_path,outputs);write_json(output_path.with_suffix('.manifest.json'),{'sha256':file_hash(output_path)})
    return {'repair_batch_id':batch_id,'modules':len(outputs),'sha256':file_hash(output_path),'same_adapter_only':True}


def encode_reports(run_dir, config):
    """Frozen base reader; research LoRA is never accidentally applied to the reader."""
    import torch
    import numpy as np
    root = Path(run_dir)
    rows = _forecast_rows(root, config)
    _lock(root / 'reader', {'config': config, 'rows_sha256': value_hash(rows),
                          'worker_sha256': file_hash(__file__)})
    model, tokenizer = _load_model(config)
    features = []
    for row in rows:
        ids = prompt_tokens(tokenizer, row['messages'], config.get('forecast', {}).get('max_seq_length', 8192))
        with torch.inference_mode():
            # Base transformer avoids allocating the vocabulary logits and all-layer hidden states.
            hidden = model.base_model(input_ids=torch.tensor([ids], device='cuda'),
                                      use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
            features.append(torch.nn.functional.normalize(hidden, dim=0).cpu().numpy())
    np.savez_compressed(root / 'reader/features.npz', game_ids=np.array([r['game_id'] for r in rows]),
                        embeddings=np.stack(features))
    write_json(root / 'reader/features.json', {'inputs_sha256': value_hash(rows),
               'features_sha256': file_hash(root / 'reader/features.npz'), 'model': _model_config(config)})
    del model
    _release(None)
    return {'games': len(rows), 'dimension': len(features[0])}


def _labels(root, split, expected):
    rows = read_rows(Path(root) / f'labels_{split}.jsonl')
    if len({r['game_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate labels')
    labels = {r['game_id']: r['home_win'] for r in rows}
    if set(labels) != set(expected) or any(v not in (0, 1) for v in labels.values()):
        raise ValueError('Labels do not match the exact forecast cohort')
    return labels


def metrics(probabilities, outcomes):
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError('Empty or unpaired evaluation')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probabilities')
    if any(y not in (0, 1) for y in outcomes):
        raise ValueError('Expected actual binary outcomes')
    n = len(outcomes)
    clipped = [min(1 - 1e-12, max(1e-12, p)) for p in probabilities]
    return {'games': n, 'brier': sum((p-y)**2 for p,y in zip(probabilities,outcomes))/n,
            'log_loss': -sum(y*math.log(p)+(1-y)*math.log1p(-p) for p,y in zip(clipped,outcomes))/n,
            'accuracy': sum((p >= .5) == y for p,y in zip(probabilities,outcomes))/n,
            'mean_probability': sum(probabilities)/n, 'home_win_rate': sum(outcomes)/n}


def paired_date_bootstrap(rows, probabilities, labels, *, samples=5000, seed=230):
    """Resample dates jointly for both methods, retaining every game on sampled dates."""
    if len(rows) != len(probabilities) or not rows or samples < 1:
        raise ValueError('Invalid paired bootstrap inputs')
    groups = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        y = labels[row['game_id']]
        groups[row['game_date_et']].append((probability-y)**2-(row['p_boost']-y)**2)
    dates = sorted(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        differences = [d for key in rng.choices(dates, k=len(dates)) for d in groups[key]]
        draws.append(sum(differences)/len(differences))
    draws.sort()
    differences = [d for key in dates for d in groups[key]]
    return {'hybrid_minus_boost_brier': sum(differences)/len(differences),
            'ci95': [draws[int(.025*(samples-1))], draws[int(.975*(samples-1))]],
            'date_clusters': len(dates), 'samples': samples,
            'scope': 'Conditional on fitted models and dev selection; excludes training uncertainty'}


def _sigmoid(values):
    import numpy as np
    from scipy.special import expit
    return expit(np.asarray(values))


def _offset(rows):
    import numpy as np
    return np.array([math.log(r['p_boost'])-math.log1p(-r['p_boost']) for r in rows])


def fit_frozen_control(features, rows, train_labels, dev_labels, penalties):
    """Original control recipe: log loss + L2, dev Brier; zero correction always competes."""
    import numpy as np
    from scipy.optimize import minimize
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    dev = [i for i,r in enumerate(rows) if r['split']=='dev']
    if not train or not dev or features.shape[0] != len(rows):
        raise ValueError('Control needs aligned train/dev features')
    x = np.c_[features, np.ones(len(rows))]
    y = np.array([train_labels[rows[i]['game_id']] for i in train])
    offset = _offset(rows)
    candidates = [{'name': 'zero_correction', 'theta': np.zeros(x.shape[1]), 'penalty': None}]
    for penalty in penalties:
        if penalty <= 0:
            raise ValueError('Positive L2 regularization required')
        def objective(theta):
            z = offset[train] + x[train] @ theta
            error = _sigmoid(z)-y
            return (float(np.mean(np.logaddexp(0,z)-y*z)+.5*penalty*np.dot(theta,theta)),
                    x[train].T @ error / len(y)+penalty*theta)
        result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                          options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('Frozen head fit failed: '+result.message)
        candidates.append({'name': f'l2_{penalty}', 'theta': result.x, 'penalty':penalty})
    yd = [dev_labels[rows[i]['game_id']] for i in dev]
    for candidate in candidates:
        candidate['dev'] = metrics(_sigmoid(offset[dev]+x[dev]@candidate['theta']).tolist(),yd)
    # Stable min gives zero correction tie priority.
    selected = min(candidates,key=lambda c:c['dev']['brier'])
    return selected, [{k:v for k,v in c.items() if k!='theta'} for c in candidates]


def train_forecaster(run_dir, config):
    """Fit frozen control and optional separate forecasting LoRA; test outcomes stay closed."""
    import numpy as np
    root = Path(run_dir)
    rows = _forecast_rows(root,config)
    train_labels = _labels(root,'train',[r['game_id'] for r in rows if r['split']=='train'])
    dev_labels = _labels(root,'dev',[r['game_id'] for r in rows if r['split']=='dev'])
    specification = {'config': config, 'inputs_sha256': value_hash(rows),
                     'train_labels': value_hash(train_labels), 'dev_labels': value_hash(dev_labels),
                     'worker_sha256': file_hash(__file__), 'checkpoint_helper_sha256': checkpoint_helper_hash()}
    options=config.get('forecast',{})
    uses_explanations=(options.get('train_lora',True) and
                      float(options.get('explanation_loss_weight',options.get('explanation_weight',0.)))>0)
    specification['forecast_explanations_sha256']=(file_hash(root/'forecast_explanations_train.jsonl') if uses_explanations else None)
    _lock(root/'forecast',specification)
    if (root/'evaluation_started.json').exists():
        raise ValueError('Test evaluation started; no fitting or reselection permitted')
    manifest = json.loads((root/'reader/features.json').read_text())
    if manifest['inputs_sha256'] != value_hash(rows) or manifest['features_sha256'] != file_hash(root/'reader/features.npz'):
        raise ValueError('Frozen reader features changed')
    with np.load(root/'reader/features.npz',allow_pickle=False) as saved:
        if saved['game_ids'].tolist() != [r['game_id'] for r in rows]:
            raise ValueError('Reader features are misaligned')
        features = saved['embeddings']
    options = config.get('forecast',{})
    control,candidates = fit_frozen_control(features,rows,train_labels,dev_labels,options.get('l2_candidates',[.001,.01,.1]))
    np.save(root/'forecast/frozen_theta.npy',control['theta'])
    selection = {'frozen':{k:v for k,v in control.items() if k!='theta'}, 'frozen_candidates':candidates,
                 'inputs_sha256':value_hash(rows), 'model':_model_config(config), 'refit_on_dev':False}
    if options.get('train_lora',True):
        selection['adapted'] = _fit_forecast_lora(root,config,rows,train_labels,dev_labels)
    selection['artifacts'] = {str(p.relative_to(root)):file_hash(p) for p in sorted((root/'forecast').rglob('*'))
                              if p.is_file() and p.name!='selection.json'}
    write_json(root/'forecast/selection.json',selection)
    return selection


def _prefix_hidden(model, ids):
    """Use only pregame input; optional teacher-forced explanation is a separate pass."""
    import torch
    # PeftModel.base_model is the LoRA wrapper; .model is its CausalLM, whose
    # base_model property reaches the transformer while retaining injected LoRA layers.
    backbone = model.get_base_model().base_model if hasattr(model,'get_base_model') else model.base_model
    tensor = torch.tensor([ids],device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        hidden = backbone(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                          use_cache=False,return_dict=True).last_hidden_state[0,-1].float()
    return torch.nn.functional.normalize(hidden,dim=0)


def _joint_state(model, head):
    return {'model.' + key: value for key, value in _adapter_state(model).items()} | {
            'head.' + key: value for key, value in head.state_dict().items()}


def _apply_joint_state(model, head, state):
    _apply_adapter(model, {key[6:]: value for key, value in state.items() if key.startswith('model.')})
    head.load_state_dict({key[5:]: value for key, value in state.items() if key.startswith('head.')}, strict=True)


def _state_digest(state):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode()); digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_selected_candidate(root, candidate):
    for relative, expected in candidate.get('candidate_files', {}).items():
        path = Path(root) / relative
        if not path.resolve().is_relative_to(Path(root).resolve()) or file_hash(path) != expected:
            raise ValueError('Saved dev-selected candidate changed')


def _forecast_steps(root, config, rows, train_labels, dev_labels, model, tokenizer, head, *, stop_after_steps=None):
    """Same weighted objective/optimizer; free the prefix graph before CE forward."""
    import torch
    import bam_checkpoint
    root = Path(root); options = config.get('forecast', {})
    tokens = [prompt_tokens(tokenizer, r['messages'], options.get('max_seq_length', 8192)) for r in rows]
    training = [i for i, r in enumerate(rows) if r['split'] == 'train']
    development = [i for i, r in enumerate(rows) if r['split'] == 'dev']
    explanation_weight = float(options.get('explanation_loss_weight', options.get('explanation_weight', 0.)))
    brier_weight = float(options.get('brier_loss_weight', options.get('brier_weight', 1.)))
    if brier_weight <= 0 or explanation_weight < 0: raise ValueError('Invalid joint objective weights')
    explanations = {}
    if explanation_weight:
        for item in read_rows(root / 'forecast_explanations_train.jsonl'):
            if item['game_id'] not in train_labels or item.get('review_status') != 'accepted' or item['game_id'] in explanations:
                raise ValueError('Explanation must be reviewed, unique and train-only')
            explanations[item['game_id']] = item['explanation']
        if set(explanations) != set(train_labels): raise ValueError('Missing training explanations')
    encoded = {i: completion_tokens(tokenizer, rows[i]['messages'] + [{'role': 'assistant', 'content': explanations[rows[i]['game_id']]}],
                                   options.get('max_seq_length', 8192)) for i in training} if explanation_weight else {}
    trainable = [p for p in model.parameters() if p.requires_grad]
    named = [('model.' + name, p) for name, p in model.named_parameters()] + [('head.' + name, p) for name, p in head.named_parameters()]
    optimizer = torch.optim.AdamW([{'params': trainable, 'lr': options.get('learning_rate', options.get('lora_learning_rate', 2e-5))},
        {'params': head.parameters(), 'lr': options.get('head_learning_rate', 1e-3)}], weight_decay=.01)
    accumulation = int(options.get('gradient_accumulation_steps', 8)); epochs = int(options.get('epochs', 2))
    if accumulation < 1 or epochs < 1: raise ValueError('Invalid accumulation/epochs')
    identity = _checkpoint_identity(config, [rows[i] for i in training], {'tokens': tokens, 'explanations': encoded},
                                    tokenizer, optimizer, named, accumulation)
    identity.update(train_labels_sha256=value_hash(train_labels), dev_labels_sha256=value_hash(dev_labels),
                    validation_rows_sha256=value_hash([rows[i] for i in development]),
                    explanations_sha256=value_hash(explanations))
    zero = {'name': 'zero_correction', 'epoch': 0,
            'dev': metrics([rows[i]['p_boost'] for i in development], [dev_labels[rows[i]['game_id']] for i in development])}
    rng = random.Random(config.get('seed', 230)); checkpoint_root = root / 'forecast_resume/checkpoints'
    resumed = bam_checkpoint.restore_checkpoint(checkpoint_root, identity=identity, adapter_state=_joint_state(model, head),
        optimizer=optimizer, named_parameters=named, shuffle_rng=rng, apply_adapter=lambda state: _apply_joint_state(model, head, state))
    cursor = resumed['cursor'] if resumed else None
    best = cursor['best'] if cursor else dict(zero); _verify_selected_candidate(root, best)
    history = list(cursor['history']) if cursor else []; global_step = cursor['global_step'] if cursor else 0
    starting_epoch = cursor['epoch'] if cursor else 0; saved_step = global_step if cursor else -1
    last_saved = time.monotonic(); device = trainable[0].device
    for epoch in range(starting_epoch, epochs):
        model.train(); head.train()
        continuing = cursor is not None and epoch == starting_epoch
        order = list(cursor['order']) if continuing else list(range(len(training)))
        if not continuing: rng.shuffle(order)
        offset = cursor['next_group_start'] if continuing else 0
        total = cursor['epoch_loss_sum'] if continuing else 0.
        def save_state(following):
            return bam_checkpoint.save_checkpoint(checkpoint_root, identity=identity,
                cursor={'epoch': epoch, 'next_group_start': following, 'global_step': global_step, 'order': order,
                        'epoch_seen': following, 'epoch_loss_sum': total, 'history': history, 'best': best},
                adapter_state=_joint_state(model, head), optimizer=optimizer, named_parameters=named, shuffle_rng=rng)
        for start in range(offset, len(order), accumulation):
            group = order[start:start + accumulation]; optimizer.zero_grad(set_to_none=True)
            for position in group:
                index = training[position]; row = rows[index]
                probability = torch.sigmoid(head(_prefix_hidden(model, tokens[index])).squeeze() + math.log(row['p_boost']) - math.log1p(-row['p_boost']))
                brier = brier_weight * (probability-train_labels[row['game_id']])**2
                if not torch.isfinite(brier): raise ValueError('Nonfinite forecast Brier loss')
                (brier / len(group)).backward()
                total += brier.detach().float().item()
                del brier, probability
                if explanation_weight:
                    item = encoded[index]; ids = torch.tensor([item['input_ids']], device=device)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        ce = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.tensor([item['labels']], device=device), use_cache=False).loss
                    if not torch.isfinite(ce): raise ValueError('Nonfinite forecast CE loss')
                    (explanation_weight * ce / len(group)).backward()
                    total += explanation_weight * ce.detach().float().item()
                    del ce, ids
            torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            interrupted = stop_after_steps is not None and global_step >= stop_after_steps
            budget_stop = _deadline_near()
            if interrupted or budget_stop or bam_checkpoint.checkpoint_due(last_saved):
                save_state(start + len(group)); saved_step = global_step; last_saved = time.monotonic()
            if interrupted or budget_stop: raise RuntimeError('Forecast optimizer checkpointed; resume identical training/dev inputs')
        model.eval(); head.eval()
        binding = {'epoch': epoch + 1, 'state_sha256': _state_digest(_joint_state(model, head)), 'identity_sha256': value_hash(identity)}
        candidate_dir = root / 'forecast_candidates' / f'epoch_{epoch + 1:03d}'
        record_path = candidate_dir / 'candidate.json'
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            if saved['sha256'] != value_hash(saved['record']) or saved['record']['binding'] != binding:
                raise ValueError('Saved validation candidate differs from resumed state/data')
            candidate = saved['record']['candidate']; _verify_selected_candidate(root, candidate)
        else:
            with torch.inference_mode():
                probabilities = [torch.sigmoid(head(_prefix_hidden(model, tokens[i])).squeeze() + math.log(rows[i]['p_boost']) - math.log1p(-rows[i]['p_boost'])).item() for i in development]
            candidate = {'name': 'forecast_lora', 'epoch': epoch + 1,
                         'dev': metrics(probabilities, [dev_labels[rows[i]['game_id']] for i in development])}
            if candidate['dev']['brier'] < best['dev']['brier']:
                candidate_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(candidate_dir / 'adapter', safe_serialization=True)
                torch.save(head.state_dict(), candidate_dir / 'head.pt')
                candidate['candidate_path'] = str(candidate_dir.relative_to(root))
                candidate['candidate_files'] = {str(p.relative_to(root)): file_hash(p) for p in candidate_dir.rglob('*') if p.is_file()}
            record = {'binding': binding, 'candidate': candidate}
            write_json(record_path, {'record': record, 'sha256': value_hash(record)})
        if candidate['dev']['brier'] < best['dev']['brier']: best = candidate
        if saved_step != global_step:
            save_state(len(order)); saved_step = global_step; last_saved = time.monotonic()
        history.append(candidate)
        print(json.dumps({'batch_complete': 'forecast_epoch', 'candidate': candidate}), flush=True)
    _verify_selected_candidate(root, best)
    if best['name'] == 'forecast_lora':
        selected = root / best['candidate_path']
        shutil.copytree(selected / 'adapter', root / 'forecast/adapter', dirs_exist_ok=True)
        shutil.copyfile(selected / 'head.pt', root / 'forecast/head.pt')
    return {**best, 'candidates': [zero] + history, 'optimizer_steps': global_step,
            'loss': 'actual-outcome Brier plus optional completion-only CE; separate backward passes before unchanged clip/step',
            'head_input': 'last pregame prefix token; explanation evaluated in separate pass'}


def _fit_forecast_lora(root,config,rows,train_labels,dev_labels):
    import torch
    model,tokenizer = _load_model(config,training=True,stage='forecast')
    head = torch.nn.Linear(model.config.hidden_size,1,device='cuda',dtype=torch.float32)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    result = _forecast_steps(root,config,rows,train_labels,dev_labels,model,tokenizer,head)
    del model,head
    _release(None)
    return result


def evaluate(run_dir,config):
    """One locked heldout evaluation, no model updates and no dev+train refit."""
    import numpy as np
    root = Path(run_dir)
    selection_path = root/'forecast/selection.json'
    selection = json.loads(selection_path.read_text())
    rows = _forecast_rows(root,config)
    if json.loads((root/'forecast/request.json').read_text())['config'] != config:
        raise ValueError('Evaluation config differs from locked training config')
    if selection['inputs_sha256']!=value_hash(rows) or selection['model']!=_model_config(config):
        raise ValueError('Selection/input mismatch')
    for relative,digest in selection['artifacts'].items():
        if file_hash(root/relative)!=digest:
            raise ValueError('Selected weights or fit provenance changed')
    lock = {'selection_sha256':file_hash(selection_path),'inputs_sha256':value_hash(rows),
            'config_sha256':value_hash(config),'worker_sha256':file_hash(__file__)}
    started = root/'evaluation_started.json'
    if started.exists():
        if json.loads(started.read_text()) != lock:
            raise ValueError('Heldout evaluation already started with different protocol')
        if (root/'evaluation.json').exists():
            manifest=json.loads((root/'evaluation_manifest.json').read_text())
            for relative,digest in manifest['files'].items():
                if file_hash(root/relative)!=digest:
                    raise ValueError('Completed heldout evaluation artifact changed: '+relative)
            if set(manifest['files'])!={'evaluation.json','test_predictions.jsonl','labels_test.jsonl'}:
                raise ValueError('Incomplete heldout evaluation completion manifest')
            return json.loads((root/'evaluation.json').read_text())
        raise ValueError('Interrupted test evaluation: inspect retained outputs; do not automatically repeat')
    write_json(started,lock)
    test = [r for r in rows if r['split']=='test']
    # Predictions are fixed before opening the test outcome file.
    with np.load(root/'reader/features.npz',allow_pickle=False) as saved:
        if saved['game_ids'].tolist()!=[r['game_id'] for r in rows]:
            raise ValueError('Feature ordering changed')
        features = saved['embeddings']
    manifest = json.loads((root/'reader/features.json').read_text())
    if manifest['features_sha256']!=file_hash(root/'reader/features.npz'):
        raise ValueError('Features changed')
    theta = np.load(root/'forecast/frozen_theta.npy',allow_pickle=False)
    indices = [i for i,r in enumerate(rows) if r['split']=='test']
    frozen = _sigmoid(_offset(test)+np.c_[features[indices],np.ones(len(test))]@theta).tolist()
    forecasts = {'boosting':[r['p_boost'] for r in test],'frozen_reader':frozen}
    if 'adapted' in selection:
        if selection['adapted']['epoch']==0:
            forecasts['forecast_lora'] = forecasts['boosting'].copy()
        else:
            import torch
            model,tokenizer = _load_model(config,adapter=root/'forecast/adapter')
            head = torch.nn.Linear(model.config.hidden_size,1,device='cuda',dtype=torch.float32)
            head.load_state_dict(torch.load(root/'forecast/head.pt',map_location='cuda',weights_only=True))
            head.eval()
            predictions = []
            with torch.inference_mode():
                for row in test:
                    ids = prompt_tokens(tokenizer,row['messages'],config.get('forecast',{}).get('max_seq_length',8192))
                    predictions.append(torch.sigmoid(head(_prefix_hidden(model,ids)).squeeze()+
                         math.log(row['p_boost'])-math.log1p(-row['p_boost'])).item())
            forecasts['forecast_lora'] = predictions
            del model,head
            _release(None)
    prediction_rows = [{'game_id':r['game_id'],'game_date_et':r['game_date_et'],
                        **{name:values[i] for name,values in forecasts.items()}} for i,r in enumerate(test)]
    write_rows(root/'test_predictions.jsonl',prediction_rows)
    labels = _labels(root,'test',[r['game_id'] for r in test])
    outcomes = [labels[r['game_id']] for r in test]
    evaluation_options = config.get('evaluation', {})
    result = {'evaluation_status':evaluation_options.get('status', config.get('evaluation_status','exploratory; freshness not established')),
              'selection_sha256':file_hash(selection_path), 'test_predictions_sha256':file_hash(root/'test_predictions.jsonl'),
              'metrics':{name:metrics(values,outcomes) for name,values in forecasts.items()},
              'comparisons':{name:paired_date_bootstrap(test,values,labels,
                              samples=evaluation_options.get('bootstrap_replicates', 5000),
                              seed=evaluation_options.get('bootstrap_seed', config.get('seed',230)))
                             for name,values in forecasts.items() if name!='boosting'},
              'same_games':True,'test_labels_sha256':file_hash(root/'labels_test.jsonl')}
    write_json(root/'evaluation.json',result)
    write_json(root/'evaluation_manifest.json',{'files':{name:file_hash(root/name) for name in
               ('evaluation.json','test_predictions.jsonl','labels_test.jsonl')}})
    return result


def smoke_training(run_dir, config):
    """Synthetic CUDA integration test, never an NBA result or research-quality score."""
    import torch
    root = Path(run_dir)
    _lock(root/'synthetic_smoke', {'config':config, 'worker_sha256':file_hash(__file__)})
    messages = [
        [{'role':'system','content':'Use only the supplied synthetic pregame fact.'},
         {'role':'user','content':f'Synthetic example {i}: HOME player {i} is listed as doubtful.'},
         {'role':'assistant','content':f'HOME player {i} is doubtful. Availability and impact remain uncertain.'}]
        for i in range(3)]
    write_rows(root/'synthetic_smoke/examples.jsonl', [{'synthetic':True,'messages':row} for row in messages])
    model,tokenizer = _load_model(config,training=True)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0.)
    losses=[]
    for row in messages:
        encoded=completion_tokens(tokenizer,row,512)
        tensor=torch.tensor([encoded['input_ids']],device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                       labels=torch.tensor([encoded['labels']],device='cuda'),use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError('Synthetic SFT loss is not finite')
        loss.backward(); optimizer.step(); losses.append(loss.detach().float().item())
    changed=sum(not torch.equal(old,new.detach()) for old,new in zip(before,parameters))
    if not changed:
        raise ValueError('Synthetic SFT did not update any LoRA parameters')
    del before,optimizer
    head=torch.nn.Linear(model.config.hidden_size,1,device='cuda',dtype=torch.float32)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    ids=prompt_tokens(tokenizer,messages[0][:-1],512)
    altered=messages[0][:-1]+[{'role':'assistant','content':'A completely different synthetic teacher explanation.'}]
    a=completion_tokens(tokenizer,messages[0],512)
    b=completion_tokens(tokenizer,altered,512)
    if a['input_ids'][:a['prefix_index']+1]!=b['input_ids'][:b['prefix_index']+1]:
        raise ValueError('Teacher explanation changed the pregame prefix')
    joint=torch.optim.AdamW([{'params':parameters,'lr':1e-4},{'params':head.parameters(),'lr':1e-3}],weight_decay=0.)
    gradient_checks=[]
    for step in range(2):
        joint.zero_grad(set_to_none=True)
        probability=torch.sigmoid(head(_prefix_hidden(model,ids)).squeeze()+math.log(.6/.4))
        probability.square().backward()  # Synthetic binary outcome zero, never a real game label.
        adapter_norm=sum(p.grad.detach().float().abs().sum().item() for p in parameters if p.grad is not None)
        head_norm=sum(p.grad.detach().float().abs().sum().item() for p in head.parameters() if p.grad is not None)
        gradient_checks.append({'step':step,'probability':probability.item(),
                                'adapter_gradient_l1':adapter_norm,'head_gradient_l1':head_norm})
        joint.step()
    if (gradient_checks[0]['adapter_gradient_l1']!=0 or gradient_checks[0]['head_gradient_l1']<=0
            or gradient_checks[1]['adapter_gradient_l1']<=0):
        raise ValueError('Residual-head/LoRA gradient boundary failed')
    if abs(gradient_checks[0]['probability']-.6)>1e-6:
        raise ValueError('Zero head did not exactly reproduce booster probability')
    result={'synthetic_only':True,'nba_performance_claim':False,'sft_examples':3,'completion_ce':losses,
            'changed_lora_tensors':changed,'prefix_unchanged_by_teacher_explanation':True,
            'gradient_checks':gradient_checks,'model':_model_config(config),
            'prompt_token_ids_sha256':value_hash(ids)}
    write_json(root/'synthetic_smoke/result.json',result)
    del joint,model,head,parameters
    _release(None)
    return result


def _compare_replay_states(expected, actual):
    import torch
    summary = {'tensor_count': 0, 'bitwise_equal_tensors': 0, 'max_absolute_difference': 0.}
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('Replay tensor structure changed')
            summary['tensor_count'] += 1
            summary['bitwise_equal_tensors'] += int(torch.equal(a, b))
            if a.numel():
                summary['max_absolute_difference'] = max(summary['max_absolute_difference'], (a.float()-b.float()).abs().max().item())
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
        elif isinstance(a, dict):
            if a.keys() != b.keys(): raise ValueError('Replay state keys changed')
            for key in a: compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b): raise ValueError('Replay state sequence changed')
            for first, second in zip(a, b): compare(first, second)
        elif a != b:
            raise ValueError('Replay non-tensor state changed')
    compare(expected, actual)
    return summary


def smoke_resume(run_dir, config):
    """Actual CUDA NF4 research and joint-head interrupted/reloaded replay."""
    import torch
    root = Path(run_dir) / 'synthetic_resume_smoke'
    _lock(root, {'config': config, 'worker_sha256': file_hash(__file__), 'checkpoint_helper_sha256': checkpoint_helper_hash()})
    small = json.loads(json.dumps(config))
    for stage in ('research', 'forecast'):
        small[stage] = dict(small.get(stage, {}), epochs=2, gradient_accumulation_steps=1, max_seq_length=512)
    small['forecast']['explanation_loss_weight'] = 1.
    rows = [{'game_id': f'synthetic:{i}', 'game_date_et': '2024-01-01', 'split': 'train',
             'module': 'availability', 'source_hashes': ['synthetic-source'], 'teacher_hash': 'synthetic-target',
             'review_status': 'accepted', 'messages': [
                 {'role': 'system', 'content': 'Only use the synthetic source. This is not a real game.'},
                 {'role': 'user', 'content': f'Synthetic player {i} is listed as doubtful; impact is uncertain.'},
                 {'role': 'assistant', 'content': f'Synthetic player {i} is doubtful, with uncertain availability.'}]} for i in range(3)]
    result = {'synthetic_only': True, 'nba_performance_claim': False, 'checkpoint_helper_sha256': checkpoint_helper_hash(), 'checks': {}}
    def read_final(checkpoint_root):
        pointer = json.loads((checkpoint_root / 'latest.json').read_text())
        path = checkpoint_root / pointer['checkpoint']
        return {name: torch.load(path / (name + '.pt'), map_location='cpu', weights_only=True) for name in ('adapter', 'optimizer', 'rng')}
    for stage in ('research', 'forecast'):
        reference = None
        for mode in ('uninterrupted', 'interrupted', 'resumed'):
            destination = root / stage / ('full' if mode == 'uninterrupted' else 'resume')
            model, tokenizer = _load_model(small, training=True, stage=stage)
            head = None
            try:
                if stage == 'research':
                    encoded = [completion_tokens(tokenizer, row['messages'], 512) for row in rows]
                    _research_steps(model, tokenizer, rows, encoded, destination, small, stop_after_steps=4 if mode == 'interrupted' else None)
                    checkpoint_root = destination / 'checkpoints'
                else:
                    head = torch.nn.Linear(model.config.hidden_size, 1, device='cuda', dtype=torch.float32)
                    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
                    forecast_rows = [{k: row[k] for k in ('game_id', 'game_date_et', 'split')} | {
                        'messages': row['messages'][:-1], 'p_boost': .6} for row in rows]
                    forecast_rows.append({**forecast_rows[0], 'game_id': 'synthetic:dev', 'split': 'dev', 'game_date_et': '2024-02-01'})
                    labels = {row['game_id']: 0 for row in forecast_rows if row['split'] == 'train'}
                    write_rows(destination / 'forecast_explanations_train.jsonl', [
                        {'game_id': row['game_id'], 'review_status': 'accepted', 'explanation': row['messages'][-1]['content']} for row in rows])
                    _forecast_steps(destination, small, forecast_rows, labels, {'synthetic:dev': 0}, model, tokenizer, head,
                                    stop_after_steps=4 if mode == 'interrupted' else None)
                    checkpoint_root = destination / 'forecast_resume/checkpoints'
                if mode == 'interrupted': raise AssertionError('Expected optimizer-boundary interruption')
                state = read_final(checkpoint_root)
                if mode == 'uninterrupted': reference = state
                else:
                    check = _compare_replay_states(reference, state)
                    # RNG tensors must be exact even when CUDA arithmetic allows tolerance.
                    rng_check = _compare_replay_states(reference['rng'], state['rng'])
                    if rng_check['tensor_count'] != rng_check['bitwise_equal_tensors']:
                        raise ValueError('CUDA/CPU RNG state did not restore exactly')
                    result['checks'][stage] = {**check, 'rng_exact': True, 'optimizer_steps': 6}
            except RuntimeError as exc:
                if mode != 'interrupted' or 'checkpointed' not in str(exc): raise
                print(json.dumps({'batch_complete': 'synthetic_interruption', 'stage': stage}), flush=True)
            finally:
                del model, head
                _release(None)
        write_json(root / 'result.json', result)
    print(json.dumps({'batch_complete': 'synthetic_resume_smoke', 'result': result}), flush=True)
    return result


def main():
    global _WORKER_DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['research-train','research-generate','research-repair','encode','forecast-train','evaluate','smoke','smoke-resume'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.budget_seconds is not None:
        if not 180 <= args.budget_seconds <= 21300: raise ValueError('Invalid bounded worker runtime')
        _WORKER_DEADLINE = time.monotonic() + args.budget_seconds
    functions = {'research-train':train_research,'research-generate':generate_research,'encode':encode_reports,
                 'forecast-train':train_forecaster,'evaluate':evaluate,'smoke':smoke_training,'smoke-resume':smoke_resume,'research-repair':repair_research}
    print(json.dumps(functions[args.stage](args.run_dir,config),indent=2),flush=True)


if __name__ == '__main__':
    main()
