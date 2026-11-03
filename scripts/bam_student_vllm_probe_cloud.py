"""Isolated, single H100 student decoding probe; ten-minute maximum."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('cs230-nba-bam-student-vllm-probe')
models = modal.Volume.from_name('cs230-nba-research-models', version=2)
runs = modal.Volume.from_name('cs230-nba-research-runs', version=2)
if modal.is_local():
    from bam_cloud import inference_image
    image = (inference_image.add_local_file(ROOT / 'scripts/bam_student_vllm.py', '/opt/bam_student_vllm.py', copy=True)
             .add_local_file(ROOT / 'scripts/bam_student_vllm_probe_worker.py', '/opt/bam_student_vllm_probe_worker.py', copy=True))
else:
    image = modal.Image.debian_slim(python_version='3.14')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n'); temporary.replace(path)


@app.function(image=image, gpu='H100', volumes={'/models': models, '/runs': runs},
              cpu=6, memory=65536, timeout=600, min_containers=0, max_containers=1,
              single_use_containers=True, retries=0)
def measure(run_id, config, expected_worker, expected_probe, expected_backend):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', run_id): raise ValueError('Invalid probe ID')
    if (config['model_id'] != 'Qwen/Qwen2.5-14B-Instruct'
            or config['model_revision'] != 'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
            or config['cache_dir'] != '/models/bam/hf_cache'):
        raise ValueError('Probe must use the already cached pinned 14B checkpoint')
    if digest('/opt/bam_training.py') != expected_worker or digest('/opt/bam_student_vllm_probe_worker.py') != expected_probe:
        raise ValueError('Probe/production worker hash differs')
    if digest('/opt/bam_student_vllm.py') != expected_backend:
        raise ValueError('Backend hash differs')
    models.reload(); runs.reload()
    root = Path('/runs/bam/probes') / run_id; root.mkdir(parents=True, exist_ok=True)
    request = {'config': config, 'worker_sha256': expected_worker, 'probe_sha256': expected_probe, 'backend_sha256': expected_backend}
    if (root / 'request.json').exists() and json.loads((root / 'request.json').read_text()) != request:
        raise ValueError('Immutable probe request differs')
    write_json(root / 'request.json', request); write_json(root / 'config.json', config)
    started = datetime.now(timezone.utc).isoformat(); runs.commit()
    error = None; process = None
    try:
        with (root / 'execution.log').open('a') as stream:
            process = subprocess.Popen(['/opt/nba-venv/bin/python', '-u', '/opt/bam_student_vllm_probe_worker.py',
                                        '--root', str(root), '--config', str(root / 'config.json')],
                                       cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            timer = threading.Timer(540, process.kill); timer.start(); last_commit = time.monotonic()
            try:
                for line in process.stdout:
                    stream.write(line); stream.flush(); print(line, end='', flush=True)
                    if '"batch_complete"' in line or time.monotonic() - last_commit > 20:
                        runs.commit(); last_commit = time.monotonic()
                if process.wait(): raise RuntimeError('Probe child failed; see retained execution log')
            finally:
                timer.cancel()
                if process.poll() is None: process.kill(); process.wait()
    except Exception as exc:
        error = str(exc)
    finally:
        runs.commit()
    return {'run_id': run_id, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'error': error,
            'result': json.loads((root / 'result.json').read_text()) if (root / 'result.json').exists() else None,
            'raw_log': (root / 'execution.log').read_text() if (root / 'execution.log').exists() else ''}
