"""Bounded two-H200 teacher comparison, isolated from the frozen BAM app."""
import json
import hashlib
from pathlib import Path
import re
import subprocess
import threading
import time

import modal

APP_NAME = 'cs230-nba-bam-teacher-probe'
app = modal.App(APP_NAME)
models = modal.Volume.from_name('cs230-nba-research-models', create_if_missing=True, version=2)
runs = modal.Volume.from_name('cs230-nba-research-runs', create_if_missing=True, version=2)
# Build from the cached inference image locally. The remote service import must
# not depend on another deployment module that Modal has not mounted.
if modal.is_local():
    from bam_cloud import ROOT, inference_image
    image = inference_image.add_local_file(ROOT / 'scripts/bam_teacher_probe_worker.py',
                                          '/opt/bam_teacher_probe_worker.py', copy=True)
else:
    image = modal.Image.debian_slim(python_version='3.14')
PINNED_MODEL = {'model_id': 'Qwen/Qwen3-235B-A22B-Thinking-2507-FP8',
                'model_revision': 'f07f63f2bbd7540917118ebdf3812696ef303b03'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def lock_request(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Existing teacher probe request differs')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def _run_logged(command, cwd, path, *, seconds, commit):
    with path.open('a') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        timer = threading.Timer(seconds, process.kill)
        timer.start()
        committed = time.monotonic()
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
                if time.monotonic() - committed >= 20:
                    commit()
                    committed = time.monotonic()
            code = process.wait()
            if code:
                raise RuntimeError(f'Teacher probe exited {code}; log retained at {path}')
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
            commit()


@app.function(image=image, gpu='H200:2', cpu=12, memory=131072,
              volumes={'/models': models, '/runs': runs}, timeout=1800,
              min_containers=0, max_containers=1, single_use_containers=True, retries=0)
def compare(requests, model_spec, run_id, config, expected_worker_hash):
    """One fixed checkpoint, at most 40 TRAIN-only source-reading requests."""
    spec = dict(model_spec)
    if spec != PINNED_MODEL:
        raise ValueError('This diagnostic is pinned to the July 2025 teacher checkpoint')
    if not 1 <= len(requests) <= 40 or any(r.get('split') != 'train' for r in requests):
        raise ValueError('Bounded TRAIN-only diagnostic requests required')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', run_id):
        raise ValueError('Invalid teacher probe run ID')
    if digest('/opt/bam_teacher_probe_worker.py') != expected_worker_hash:
        raise ValueError('Teacher probe worker hash changed')
    models.reload()
    runs.reload()
    key = spec['model_id'].replace('/', '--') + '--' + spec['model_revision']
    manifest = json.loads((Path('/models/bam/manifests') / (key + '.json')).read_text())
    if manifest['model_spec'] != spec:
        raise ValueError('Pinned teacher manifest differs')
    root = Path('/runs/bam/generation') / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock_request(root / 'request.json', {'requests': requests, 'model_manifest': manifest,
        'config': config, 'worker_sha256': expected_worker_hash,
        'scope': 'Actual-claim source audit only; never forecasting or training inputs'})
    runs.commit()
    try:
        _run_logged(['/opt/nba-venv/bin/python', '-u', '/opt/bam_teacher_probe_worker.py',
                     '--root', str(root), '--request', str(root / 'request.json')],
                    root, root / 'execution.log', seconds=1740, commit=runs.commit)
        return {'run_id': run_id, 'summary': json.loads((root / 'summary.json').read_text()),
                'identity': json.loads((root / 'identity.json').read_text()),
                'responses': [json.loads(line) for line in (root / 'responses.jsonl').read_text().splitlines()]}
    finally:
        runs.commit()
