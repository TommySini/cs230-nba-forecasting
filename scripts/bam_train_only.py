"""Bounded TRAIN-only joint optimization; no held-out loader or evaluator.

The public entry point requires a hash-pinned, source-audited TRAIN bundle.
Experiment execution belongs on the admitted remote GPU worker. Local tests use
only tiny synthetic tensors and never load the forecasting checkpoint or data.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import bam_checkpoint as checkpoint
import bam_training as common
from bam_aws_preflight import is_commercial_region


MODEL_ID = 'Qwen/Qwen2.5-14B-Instruct'
MODEL_REVISION = 'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'


class RuntimeLimit(RuntimeError):
    """The admitted wall-clock budget was reached; no partial update was applied."""


def validate_config(config):
    common._model_config(config)
    if config['model_id'] != MODEL_ID or config['model_revision'] != MODEL_REVISION:
        raise ValueError('The pilot requires the approved pinned Qwen2.5-14B forecaster')
    options = config.get('train_only', {})
    for key in ('max_optimizer_steps', 'max_runtime_seconds', 'max_seq_length', 'evaluate_every_steps'):
        if type(options.get(key)) is not int or options[key] < 1:
            raise ValueError('Explicit positive TRAIN-only limit required: ' + key)
    for key in ('learning_rate', 'head_learning_rate', 'explanation_weight', 'gradient_clip'):
        value = options.get(key)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Explicit finite positive TRAIN-only setting required: ' + key)
    for key in ('minimum_relative_brier_reduction', 'minimum_relative_ce_reduction'):
        value = options.get(key)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 < value < 1:
            raise ValueError('Pin the TRAIN learning criterion before fitting: ' + key)
    if config.get('forecast', {}).get('lora_rank') != 8:
        raise ValueError('TRAIN pilot requires the approved rank-8 forecaster adapters')
    return options


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise RuntimeLimit('TRAIN runtime limit reached; uncommitted gradients discarded')


def _autocast(device):
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def prefix_hidden(model, token_ids, device):
    """Read the last pregame token; this call has no gold-token argument."""
    import torch
    backbone = model.get_base_model().base_model if hasattr(model, 'get_base_model') else model.base_model
    tensor = torch.tensor([token_ids], device=device)
    with _autocast(device):
        hidden = backbone(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                          use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
    return torch.nn.functional.normalize(hidden, dim=0)


def _probability(model, head, tokens, p_boost):
    import torch
    return torch.sigmoid(head(prefix_hidden(model, tokens, head.weight.device)).squeeze()
                         + math.log(p_boost) - math.log1p(-p_boost))


def _completion_loss(model, item, device):
    import torch
    ids = torch.tensor([item['input_ids']], device=device)
    with _autocast(device):
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                     labels=torch.tensor([item['labels']], device=device), use_cache=False).loss
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite TRAIN completion loss')
    return loss


def _gradient_norm(parameters):
    import torch
    total = 0.
    for parameter in parameters:
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise ValueError('Nonfinite TRAIN gradient')
            total += parameter.grad.detach().float().square().sum().item()
    return math.sqrt(total)


def train_metrics(model, head, rows, labels, tokens, encoded, weight, deadline):
    """Inference-mode metrics on exactly the supplied TRAIN cohort."""
    import torch
    model.eval(); head.eval()
    probabilities, brier, ce = [], 0., 0.
    with torch.inference_mode():
        for i, row in enumerate(rows):
            _check_deadline(deadline)
            value = _probability(model, head, tokens[i], row['p_boost']).item()
            if not math.isfinite(value):
                raise ValueError('Nonfinite TRAIN probability')
            probabilities.append(value)
            brier += (value - labels[row['game_id']]) ** 2 / len(rows)
        for item in encoded.values():
            _check_deadline(deadline)
            ce += _completion_loss(model, item, head.weight.device).float().item() / len(encoded)
    return {'brier': brier, 'gold_completion_ce': ce if encoded else None,
            'weighted_total': brier + weight * ce, 'outcome_examples': len(rows),
            'gold_examples': len(encoded), 'probabilities': probabilities}


def joint_update(model, head, optimizer, rows, labels, tokens, encoded, order, *, weight, gradient_clip, deadline):
    """One exact mean_N(Brier) + lambda mean_M(per-example CE) update.

    A complete cohort is the accumulation group. Sparse gold never changes the
    outcome denominator, and outcomes without gold never dilute the gold term.
    Separate backward passes release the prefix graph before gold is supplied.
    """
    import torch
    if sorted(order) != list(range(len(rows))) or not rows:
        raise ValueError('TRAIN accumulation order must contain every game exactly once')
    adapters = [p for p in model.parameters() if p.requires_grad]
    head_parameters = list(head.parameters())
    before_adapters = [p.detach().clone() for p in adapters]
    before_head = [p.detach().clone() for p in head_parameters]
    model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
    brier_sum, ce_sum = 0., 0.
    try:
        # Complete the probability objective before any teacher-forced forward.
        for index in order:
            _check_deadline(deadline)
            row = rows[index]
            probability = _probability(model, head, tokens[index], row['p_boost'])
            brier = (probability - labels[row['game_id']]).square()
            if not torch.isfinite(brier):
                raise ValueError('Nonfinite TRAIN Brier loss')
            (brier / len(rows)).backward()
            brier_sum += brier.detach().float().item() / len(rows)
            del probability, brier
        brier_adapter_norm = _gradient_norm(adapters)
        brier_head_norm = _gradient_norm(head_parameters)
        for index in order:
            if index not in encoded:
                continue
            _check_deadline(deadline)
            ce = _completion_loss(model, encoded[index], head.weight.device)
            (weight * ce / len(encoded)).backward()
            ce_sum += ce.detach().float().item() / len(encoded)
            del ce
        joint_adapter_norm = _gradient_norm(adapters)
        joint_head_norm = _gradient_norm(head_parameters)
        _check_deadline(deadline)
        torch.nn.utils.clip_grad_norm_(adapters + head_parameters, gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in adapters + head_parameters):
            raise ValueError('Nonfinite TRAIN parameter after optimizer update')
    finally:
        optimizer.zero_grad(set_to_none=True)
    return {'brier': brier_sum, 'gold_completion_ce': ce_sum if encoded else None,
            'weighted_total': brier_sum + weight * ce_sum,
            'brier_adapter_gradient_norm': brier_adapter_norm, 'brier_head_gradient_norm': brier_head_norm,
            'joint_adapter_gradient_norm': joint_adapter_norm, 'joint_head_gradient_norm': joint_head_norm,
            'adapter_updated': any(not torch.equal(p.detach(), old) for p, old in zip(adapters, before_adapters)),
            'head_updated': any(not torch.equal(p.detach(), old) for p, old in zip(head_parameters, before_head))}


def _validate_core_inputs(rows, labels, gold):
    if not rows or any(row.get('split') != 'train' for row in rows):
        raise ValueError('TRAIN-only optimizer rejects every non-TRAIN row')
    ids = [row['game_id'] for row in rows]
    if len(set(ids)) != len(ids) or set(labels) != set(ids) or not set(gold) <= set(ids):
        raise ValueError('TRAIN input/outcome/gold alignment mismatch')
    if any(type(value) is not int or value not in (0, 1) for value in labels.values()):
        raise ValueError('TRAIN outcomes must be binary integers')
    for row in rows:
        common._messages(row['messages'])
        if common.FORBIDDEN_INPUT_KEYS & row.keys():
            raise ValueError('Outcome or gold in pregame record')
        if isinstance(row['p_boost'], bool) or not 0 < row['p_boost'] < 1:
            raise ValueError('Invalid prior booster probability')
    for game_id, item in gold.items():
        if (item.get('game_id') != game_id or item.get('split') != 'train'
                or item.get('review_status') != 'accepted' or not item.get('explanation', '').strip()):
            raise ValueError('TRAIN explanations require accepted, aligned gold')


def _learning_progress(baseline, last_metrics, history, options):
    step = len(history)
    latest_measured_step = max((item['step'] for item in history if 'inference_train' in item), default=0)
    baseline_brier = baseline['brier']; baseline_ce = baseline['gold_completion_ce']
    brier_reduction = 1 - last_metrics['brier'] / baseline_brier if baseline_brier else 0.
    ce_reduction = 1 - last_metrics['gold_completion_ce'] / baseline_ce if baseline_ce else 0.
    updates_verified = bool(history) and any(x['optimization']['head_updated'] for x in history) and any(
        x['optimization']['adapter_updated'] for x in history)
    brier_adapter_verified = any(x['optimization']['brier_adapter_gradient_norm'] > 0 for x in history[1:])
    proved = (latest_measured_step == step and step > 1 and updates_verified and brier_adapter_verified
              and brier_reduction >= options['minimum_relative_brier_reduction']
              and ce_reduction >= options['minimum_relative_ce_reduction'])
    return {'last_measured_step': latest_measured_step, 'relative_brier_reduction': brier_reduction,
            'relative_ce_reduction': ce_reduction, 'head_and_adapters_updated': updates_verified,
            'brier_gradient_reached_adapters_after_zero_head': brier_adapter_verified,
            'learning_criterion_met': proved}


def train_steps(root, config, rows, labels, gold, model, tokenizer, head, *,
                manifest_sha256, deadline=None, stop_after_steps=None):
    """Joint fitting core; public train_bundle additionally verifies source receipts.

    The synthetic tests inject a tiny model. Production only constructs the
    pinned QLoRA model after immutable source validation and launch admission.
    """
    import torch
    options = validate_config(config)
    _validate_core_inputs(rows, labels, gold)
    deadline = deadline if deadline is not None else time.monotonic() + options['max_runtime_seconds']
    root = Path(root)
    tokens = [common.prompt_tokens(tokenizer, row['messages'], options['max_seq_length']) for row in rows]
    encoded = {i: common.completion_tokens(tokenizer, row['messages'] + [
        {'role': 'assistant', 'content': gold[row['game_id']]['explanation']}], options['max_seq_length'])
        for i, row in enumerate(rows) if row['game_id'] in gold}
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in name for name, _ in parameters):
        raise ValueError('Only LoRA adapters may be trainable in the forecaster backbone')
    named = [('model.' + name, p) for name, p in model.named_parameters()] + [
        ('head.' + name, p) for name, p in head.named_parameters()]
    optimizer = torch.optim.AdamW([
        {'params': [p for _, p in parameters], 'lr': options['learning_rate']},
        {'params': list(head.parameters()), 'lr': options['head_learning_rate']}], weight_decay=0.01)
    identity = common._checkpoint_identity(config, rows, {'tokens': tokens, 'gold': encoded},
                                          tokenizer, optimizer, named, len(rows))
    identity.update(worker_sha256=common.file_hash(__file__), helper_sha256=common.file_hash(common.__file__),
                    manifest_sha256=manifest_sha256, outcomes_sha256=common.value_hash(labels),
                    gold_sha256=common.value_hash(gold), phase='train_only')
    rng = random.Random(config.get('seed', 230))
    checkpoint_root = root / 'checkpoints'
    resumed = checkpoint.restore_checkpoint(checkpoint_root, identity=identity,
        adapter_state=common._joint_state(model, head), optimizer=optimizer, named_parameters=named,
        shuffle_rng=rng, apply_adapter=lambda state: common._apply_joint_state(model, head, state))
    cursor = resumed['cursor'] if resumed else None
    step = cursor['global_step'] if cursor else 0
    history = list(cursor['history']) if cursor else []
    baseline = cursor['baseline'] if cursor else None
    if baseline is None:
        if any(torch.count_nonzero(p.detach()).item() for p in head.parameters()):
            raise ValueError('The initial residual probability head must be exactly zero')
        baseline = train_metrics(model, head, rows, labels, tokens, encoded, options['explanation_weight'], deadline)
        if any(abs(value - row['p_boost']) > 1e-6 for value, row in zip(baseline['probabilities'], rows)):
            raise ValueError('Zero head failed to reproduce the prior boosting forecast')
        baseline['zero_head_equivalence_verified'] = True
        common.write_json(root / 'baseline_train.json', baseline)
    last_metrics = cursor['last_metrics'] if cursor else baseline
    max_steps = options['max_optimizer_steps']
    status = 'maximum_steps'
    while step < max_steps:
        if _learning_progress(baseline, last_metrics, history, options)['learning_criterion_met']:
            status = 'learning_criterion_met'
            break
        order = list(range(len(rows))); rng.shuffle(order)
        try:
            update = joint_update(model, head, optimizer, rows, labels, tokens, encoded, order,
                                  weight=options['explanation_weight'], gradient_clip=options['gradient_clip'], deadline=deadline)
        except RuntimeLimit:
            status = 'runtime_limit'
            break
        step += 1
        record = {'step': step, 'optimization': update}
        if step % options['evaluate_every_steps'] == 0 or step == max_steps:
            try:
                last_metrics = train_metrics(model, head, rows, labels, tokens, encoded, options['explanation_weight'], deadline)
                record['inference_train'] = last_metrics
            except RuntimeLimit:
                status = 'runtime_limit'
        history.append(record)
        # Cursor points to the next complete-cohort update. No partial gradient
        # group or held-out selection state is ever serialized.
        cursor = {'epoch': step, 'next_group_start': 0, 'global_step': step, 'order': order,
                  'epoch_seen': 0, 'epoch_loss_sum': 0., 'history': history,
                  'baseline': baseline, 'last_metrics': last_metrics}
        checkpoint.save_checkpoint(checkpoint_root, identity=identity, cursor=cursor,
            adapter_state=common._joint_state(model, head), optimizer=optimizer,
            named_parameters=named, shuffle_rng=rng)
        common.write_json(root / 'learning_curve_train.json', {'baseline': baseline, 'steps': history})
        if stop_after_steps is not None and step >= stop_after_steps:
            raise RuntimeError('TRAIN optimizer checkpointed for controlled resume test')
        if status == 'runtime_limit':
            break
    progress = _learning_progress(baseline, last_metrics, history, options)
    if progress['learning_criterion_met']:
        status = 'learning_criterion_met'
    result = {'phase': 'train_only', 'status': status, 'optimizer_steps': step,
              'manifest_sha256': manifest_sha256, 'identity_sha256': common.value_hash(identity),
              'baseline': baseline, 'last_inference_train': last_metrics,
              **progress, 'generalization_claim': False,
              'objective': 'mean_N(Brier) + explanation_weight * mean_M(per-example gold CE)',
              'probability_input': 'pregame prefix only', 'checkpoint_root': 'checkpoints'}
    common.write_json(root / 'result_train.json', result)
    return result


def _read_execution_admission(path, preflight_path, manifest_sha256, config, job_id):
    """Validate a launcher receipt; this is a guard, not independent AWS attestation.

    The launcher remains responsible for role identity, live pricing, credit
    eligibility, ledger reservation and external timed instance termination.
    """
    options = validate_config(config)
    receipt = json.loads(Path(path).read_text())
    preflight_bytes = Path(preflight_path).read_bytes()
    preflight = json.loads(preflight_bytes)
    model = {key: config[key] for key in ('model_id', 'model_revision')}
    binding = {'job_id': job_id, 'manifest_sha256': manifest_sha256,
               'config_sha256': common.value_hash(config), 'model': model, 'region': receipt.get('region')}
    if (receipt.get('platform') != 'aws_ec2' or not is_commercial_region(receipt.get('region'))
            or receipt.get('status') != 'admitted'
            or not isinstance(receipt.get('reservation_id'), str) or not receipt['reservation_id'].strip()
            or not isinstance(job_id, str) or not job_id.strip() or receipt.get('job_id') != job_id
            or receipt.get('config_sha256') != common.value_hash(config) or receipt.get('model') != model
            or receipt.get('manifest_sha256') != manifest_sha256
            or receipt.get('max_runtime_seconds') != options['max_runtime_seconds']
            or receipt.get('external_termination_armed') is not True):
        raise ValueError('Require a matching admitted EC2 reservation in the plan region and external termination')
    if (receipt.get('preflight_report_sha256') != hashlib.sha256(preflight_bytes).hexdigest()
            or preflight.get('schema_version') != 1 or preflight.get('binding') != binding
            or preflight.get('prerequisites_satisfied') is not True or preflight.get('blockers') != []
            or preflight.get('reservation_created') is not False or preflight.get('launch_authorized') is not False):
        raise ValueError('Require the matching complete preflight report plus a separate launcher reservation')
    checked = datetime.fromisoformat(preflight['checked_at'].replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    if checked.tzinfo is None or checked > now or now - checked > timedelta(hours=24):
        raise ValueError('Preflight report is future-dated or older than 24 hours')
    financials = preflight.get('financials', {})
    if financials.get('cost_bounds_complete') is not True:
        raise ValueError('Incomplete preflight cost bounds')
    amounts = [receipt.get('reserved_cents')] + [financials.get(key) for key in (
        'bounded_reservation_cents', 'verified_credit_available_cents', 'ledger_available_cents')]
    if any(type(value) is not int or value <= 0 for value in amounts):
        raise ValueError('Launch funding must use positive integer cents')
    reserved, bounded, credit, ledger = amounts
    if not bounded <= reserved <= min(credit, ledger):
        raise ValueError('Reservation must cover the cost bound within verified credits and remaining project cap')
    expiry = datetime.fromisoformat(receipt['expires_at_utc'].replace('Z', '+00:00'))
    if expiry.tzinfo is None or expiry <= now or expiry > checked + timedelta(hours=24):
        raise ValueError('Launch admission expired or lacks timezone')
    return receipt


def train_bundle(bundle, expected_manifest_sha256, config, output, admission, preflight_report, job_id):
    from bam_train_contract import load_bundle
    options = validate_config(config)
    # Receipt is checked before any model/network initialization.
    receipt = _read_execution_admission(admission, preflight_report, expected_manifest_sha256, config, job_id)
    manifest, rows, outcomes, gold = load_bundle(bundle, expected_manifest_sha256, require_all_gold=True)
    if not manifest.get('pilot') or not 16 <= len(rows) <= 32:
        raise ValueError('The first execution must be a 16–32-game reviewed-gold pilot')
    output = Path(output)
    request = {'manifest_sha256': expected_manifest_sha256, 'config': config, 'admission': receipt}
    common._lock(output, request)
    started_path = output / 'started_utc.json'
    if not started_path.exists():
        common.write_json(started_path, {'started_utc': datetime.now(timezone.utc).isoformat()})
    started = datetime.fromisoformat(json.loads(started_path.read_text())['started_utc'])
    remaining = options['max_runtime_seconds'] - (datetime.now(timezone.utc) - started).total_seconds()
    if remaining <= 0:
        raise RuntimeLimit('The original wall-clock admission has expired; do not reset it on resume')
    deadline = time.monotonic() + remaining
    import torch
    model, tokenizer = common._load_model(config, training=True, stage='forecast')
    _check_deadline(deadline)
    head = torch.nn.Linear(model.config.hidden_size, 1, device='cuda', dtype=torch.float32)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    try:
        result = train_steps(output, config, rows, {key: value['home_win'] for key, value in outcomes.items()},
                             gold, model, tokenizer, head, manifest_sha256=expected_manifest_sha256, deadline=deadline)
        model.save_pretrained(output / 'adapter', safe_serialization=True)
        tokenizer.save_pretrained(output / 'adapter')
        torch.save(head.state_dict(), output / 'head.pt')
        common.write_json(output / 'model_files.json', {
            str(path.relative_to(output)): common.file_hash(path)
            for path in sorted((output / 'adapter').rglob('*')) if path.is_file()} | {'head.pt': common.file_hash(output / 'head.pt')})
        return result
    finally:
        del model, head
        common._release(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--preflight-report', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    result = train_bundle(args.bundle, args.manifest_sha256, json.loads(Path(args.config).read_text()),
                          args.output, args.admission, args.preflight_report, args.job_id)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
