"""Atomic optimizer-boundary checkpoints, independent of the production worker.

No torch import until save/restore. Callers supply PEFT adapter tensors and a
strict loader callback; this module never saves or replaces base-model weights.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
import uuid


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _json(path, value):
    with Path(path).open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_identity(identity):
    for field in ('config_sha256', 'rows_sha256', 'tokenized_sha256', 'worker_sha256', 'tokenizer_sha256'):
        if not re.fullmatch(r'[a-f0-9]{64}', identity.get(field, '')):
            raise ValueError('Missing or invalid checkpoint identity: ' + field)
    model = identity.get('model', {})
    if not model.get('model_id') or not re.fullmatch(r'[a-f0-9]{40}', model.get('model_revision', '')):
        raise ValueError('Checkpoint needs exact pinned base-model identity')
    for field in ('row_count', 'gradient_accumulation_steps'):
        if type(identity.get(field)) is not int or identity[field] < 1:
            raise ValueError('Invalid checkpoint identity: ' + field)
    names = identity.get('parameter_names')
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or len(names) != len(set(names))):
        raise ValueError('Optimizer parameter names must be explicit, ordered and unique')
    # Canonical JSON roundtrip disallows hidden nonserializable identity objects.
    if json.loads(json.dumps(identity, allow_nan=False)) != identity:
        raise ValueError('Checkpoint identity must use canonical JSON values')


def _validate_cursor(cursor, identity):
    for key in ('epoch', 'next_group_start', 'global_step', 'epoch_seen'):
        if type(cursor.get(key)) is not int or cursor[key] < 0:
            raise ValueError('Invalid checkpoint cursor: ' + key)
    order = cursor.get('order')
    if (not isinstance(order, list) or any(type(i) is not int for i in order)
            or sorted(order) != list(range(identity['row_count']))):
        raise ValueError('Saved row order is not the exact dataset permutation')
    offset = cursor['next_group_start']
    if (offset > len(order) or cursor['epoch_seen'] != offset
            or offset != len(order) and offset % identity['gradient_accumulation_steps']):
        raise ValueError('Cursor is not at a completed accumulation-group boundary')
    loss = cursor.get('epoch_loss_sum')
    if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError('Invalid saved epoch loss sum')
    history = cursor.get('history')
    if not isinstance(history, list) or len(history) != cursor['epoch']:
        raise ValueError('History must contain only completed previous epochs')
    json.dumps(cursor, allow_nan=False)


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _adapter_schema(state):
    import torch
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Adapter state must be a nonempty tensor dictionary')
    return {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)} for name, tensor in state.items()}


def optimizer_parameter_names(optimizer, named_parameters):
    """Derive names from actual parameter objects, never caller-supplied order."""
    named = list(named_parameters)
    if (any(not isinstance(name, str) or not name for name, _ in named)
            or len({name for name, _ in named}) != len(named)
            or len({id(parameter) for _, parameter in named}) != len(named)):
        raise ValueError('Named model tensors must have unique names and object identities')
    by_object = {id(parameter): name for name, parameter in named}
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Repeated optimizer parameter')
    if any(id(p) not in by_object for p in parameters):
        raise ValueError('Optimizer parameter is not bound to a named model tensor')
    return [by_object[id(p)] for p in parameters]


def _runtime_signature():
    import torch
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    return {'torch_version': str(torch.__version__), 'cuda_available': available,
            'cuda_device_count': count,
            'cuda_device_names': [torch.cuda.get_device_name(i) for i in range(count)]}


def _check_optimizer(optimizer, identity, named_parameters, *, boundary=False):
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    actual_names = optimizer_parameter_names(optimizer, named_parameters)
    if actual_names != identity['parameter_names']:
        raise ValueError('Actual optimizer named-parameter order differs from identity')
    if boundary and any(p.grad is not None for p in parameters):
        raise ValueError('Checkpoint only after optimizer.step and zero_grad(set_to_none=True)')
    return [{'name': name, 'shape': list(p.shape), 'dtype': str(p.dtype), 'device': str(p.device)}
            for name, p in zip(actual_names, parameters)]


def _read_pointer(root, identity):
    root = Path(root)
    pointer_path = root / 'latest.json'
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text())
    name = pointer.get('checkpoint', '')
    match = re.fullmatch(r'step_(\d{12})_[a-f0-9]{32}', name)
    if not match:
        raise ValueError('Unsafe checkpoint pointer')
    directory = root / name
    manifest_path = directory / 'manifest.json'
    if directory.is_symlink() or not directory.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError('Checkpoint pointer references an incomplete or unsafe directory')
    if _digest(manifest_path) != pointer.get('manifest_sha256'):
        raise ValueError('Checkpoint manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('version') != 1 or manifest.get('identity') != identity:
        raise ValueError('Checkpoint scientific identity changed')
    if manifest.get('helper_sha256') != _digest(__file__):
        raise ValueError('Checkpoint helper implementation changed')
    expected = {'adapter.pt', 'optimizer.pt', 'rng.pt', 'cursor.json'}
    if set(manifest.get('files', {})) != expected:
        raise ValueError('Incomplete checkpoint manifest')
    for name, metadata in manifest['files'].items():
        path = directory / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != metadata['bytes']
                or _digest(path) != metadata['sha256']):
            raise ValueError('Checkpoint file checksum mismatch: ' + name)
    cursor = json.loads((directory / 'cursor.json').read_text())
    _validate_cursor(cursor, identity)
    if cursor['global_step'] != pointer.get('global_step') or int(match[1]) != cursor['global_step']:
        raise ValueError('Checkpoint step/pointer mismatch')
    return directory, manifest, cursor, pointer


def checkpoint_due(last_saved_monotonic, *, interval_seconds=180, now=None):
    """Check at optimizer boundaries; callers initialize this after loading."""
    if not 0 < interval_seconds <= 300:
        raise ValueError('Checkpoint interval must be at most five minutes')
    return (time.monotonic() if now is None else now) - last_saved_monotonic >= interval_seconds


def save_checkpoint(root, *, identity, cursor, adapter_state, optimizer, named_parameters, shuffle_rng):
    """Save complete immutable state, then atomically advance latest.json.

    Emit progress only after the durable local commit. A Modal parent must then
    commit its volume; this helper does not pretend local fsync is a cloud commit.
    """
    import torch
    _validate_identity(identity); _validate_cursor(cursor, identity)
    optimizer_schema = _check_optimizer(optimizer, identity, named_parameters, boundary=True)
    adapter_schema = _adapter_schema(adapter_state)
    runtime = _runtime_signature()
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError('Checkpoint root cannot be a symlink')
    previous = _read_pointer(root, identity)
    if previous and cursor['global_step'] <= previous[2]['global_step']:
        raise ValueError('Checkpoint progress must increase monotonically')
    # A crash after directory rename but before pointer replacement leaves an
    # orphan. Unique immutable names permit replay from the preceding pointer
    # without overwriting or colliding with that uncommitted checkpoint.
    destination = root / f"step_{cursor['global_step']:012d}_{uuid.uuid4().hex}"
    temporary = root / ('.pending-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        torch.save(_cpu(adapter_state), temporary / 'adapter.pt')
        torch.save(_cpu(optimizer.state_dict()), temporary / 'optimizer.pt')
        torch.save({'shuffle': shuffle_rng.getstate(), 'python': random.getstate(),
                    'torch_cpu': torch.get_rng_state(),
                    'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary / 'rng.pt')
        _json(temporary / 'cursor.json', cursor)
        files = {}
        for path in sorted(temporary.iterdir()):
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            files[path.name] = {'sha256': _digest(path), 'bytes': path.stat().st_size}
        _json(temporary / 'manifest.json', {'version': 1, 'identity': identity, 'helper_sha256': _digest(__file__),
              'adapter_schema': adapter_schema, 'optimizer_schema': optimizer_schema,
              'optimizer_class': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
              'runtime': runtime, 'files': files})
        _sync_directory(temporary)
        temporary.rename(destination)
        _sync_directory(root)
        pointer = {'checkpoint': destination.name, 'manifest_sha256': _digest(destination / 'manifest.json'),
                   'global_step': cursor['global_step']}
        pending_pointer = root / ('.latest-' + uuid.uuid4().hex + '.json')
        _json(pending_pointer, pointer)
        pending_pointer.replace(root / 'latest.json')
        _sync_directory(root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps({'checkpoint_complete': pointer, 'epoch': cursor['epoch'],
                      'next_group_start': cursor['next_group_start']}), flush=True)
    return pointer


def restore_checkpoint(root, *, identity, adapter_state, optimizer, named_parameters, shuffle_rng, apply_adapter):
    """Validate all identities/hashes before mutation; restore RNG last.

    `adapter_state` describes current expected adapter names/shapes/dtypes.
    `apply_adapter` must strictly load the supplied saved adapter tensors.
    `named_parameters` must enumerate the live model (and, if present, named
    forecast head) tensors, not names zipped against an optimizer parameter list.
    Model/optimizer construction must happen before calling this function.
    """
    import torch
    _validate_identity(identity)
    saved = _read_pointer(root, identity)
    if saved is None:
        return None
    directory, manifest, cursor, pointer = saved
    if _runtime_signature() != manifest['runtime']:
        raise ValueError('Checkpoint CPU/CUDA runtime or device topology changed')
    if _adapter_schema(adapter_state) != manifest['adapter_schema']:
        raise ValueError('Adapter tensor identity/shape/dtype changed')
    if _check_optimizer(optimizer, identity, named_parameters, boundary=True) != manifest['optimizer_schema']:
        raise ValueError('Optimizer tensor identity/shape/dtype changed')
    if manifest['optimizer_class'] != type(optimizer).__module__ + '.' + type(optimizer).__qualname__:
        raise ValueError('Optimizer algorithm changed')
    state = torch.load(directory / 'optimizer.pt', map_location='cpu', weights_only=True)
    expected_groups = optimizer.state_dict()['param_groups']
    if state.get('param_groups') != expected_groups:
        raise ValueError('Optimizer groups/order/hyperparameters changed')
    rng = torch.load(directory / 'rng.pt', map_location='cpu', weights_only=True)
    if len(rng['torch_cuda']) != manifest['runtime']['cuda_device_count']:
        raise ValueError('CUDA RNG device topology changed')
    adapter = torch.load(directory / 'adapter.pt', map_location='cpu', weights_only=True)
    if _adapter_schema(adapter) != manifest['adapter_schema']:
        raise ValueError('Saved adapter tensor schema disagrees with manifest')
    apply_adapter(adapter)
    optimizer.load_state_dict(state)
    shuffle_rng.setstate(rng['shuffle']); random.setstate(rng['python'])
    torch.set_rng_state(rng['torch_cpu'])
    if rng['torch_cuda']:
        torch.cuda.set_rng_state_all(rng['torch_cuda'])
    return {'cursor': cursor, 'pointer': pointer}
