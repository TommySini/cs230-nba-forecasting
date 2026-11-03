"""Run the pinned research benchmark on one bounded, automatically released Modal GPU."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('cs230-nba-full-note-research')
models = modal.Volume.from_name('cs230-nba-research-models', create_if_missing=True, version=2)
runs = modal.Volume.from_name('cs230-nba-research-runs', create_if_missing=True, version=2)
image = (
    modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04', add_python='3.14')
    .entrypoint([])
    .add_local_file(ROOT / 'requirements-research-cuda.txt', '/tmp/research-requirements.txt', copy=True)
    .run_commands('python -m venv /opt/nba-venv',
                  '/opt/nba-venv/bin/pip install --no-cache-dir -r /tmp/research-requirements.txt')
    .env({'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false'})
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(payload, expected_sha, destination, prefix):
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError('Transferred archive checksum mismatch')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if not member.isfile() or not parts or parts[0] != prefix or '..' in parts or member.name.startswith('/'):
                raise ValueError('Unexpected archive member: ' + member.name)
        archive.extractall(destination, filter='data')
    root = destination / prefix
    (root / '.venv').symlink_to('/opt/nba-venv', target_is_directory=True)
    (root / 'data/models').mkdir(parents=True, exist_ok=True)
    config = json.loads((root / 'config/game_note_research.json').read_text())
    name = Path(config['model_path']).name
    (root / config['model_path']).symlink_to(Path('/models') / name, target_is_directory=True)
    return root


def run_logged(command, cwd, log, seconds):
    # MLX's CUDA wheel supplies newer matched CUDA libraries than the base
    # image. Resolve those together rather than mixing its cuSOLVER with the
    # base image's older cuBLAS, which lacks cublasSetEnvironmentMode.
    env = os.environ.copy()
    libraries = sorted(Path('/opt/nba-venv/lib/python3.14/site-packages/nvidia').glob('*/lib'))
    env['LD_LIBRARY_PATH'] = ':'.join([*(str(p) for p in libraries), env.get('LD_LIBRARY_PATH', '')])
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env)
        timer = threading.Timer(seconds, process.kill)
        timer.start()
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            result = process.wait()
            if result:
                raise RuntimeError(f'Research subprocess failed with code {result}; log: {log}')
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()


@app.function(image=image, volumes={'/models': models}, cpu=2, memory=16384,
              timeout=1800, max_containers=1, single_use_containers=True, retries=0)
def fetch_model(config, expected):
    name = Path(config['model_path']).name
    target = Path('/models') / name
    spec = {'id': config['model_id'], 'revision': config['model_revision'], 'path': str(target)}
    program = """import json,sys
from huggingface_hub import snapshot_download
s=json.loads(sys.argv[1])
snapshot_download(s['id'],revision=s['revision'],local_dir=s['path'],token=False,
allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE','README.md'])
"""
    subprocess.run(['/opt/nba-venv/bin/python', '-c', program, json.dumps(spec)], check=True, timeout=1500)
    for relative, wanted in expected.items():
        if digest(target / Path(relative).name) != wanted:
            raise ValueError('Cloud checkpoint differs from frozen model: ' + relative)
    models.commit()
    return {'revision': config['model_revision'], 'model_hashes': 'passed',
            'bytes': sum(p.stat().st_size for p in target.iterdir() if p.is_file())}


@app.function(image=image, gpu='H100', volumes={'/models': models, '/runs': runs},
              cpu=4, memory=32768, timeout=1200, max_containers=1,
              single_use_containers=True, retries=0)
def benchmark(payload, expected_sha, run_id):
    root = unpack(payload, expected_sha, Path('/runs/benchmarks') / run_id, 'research_compute_bundle')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                         '--format=csv,noheader'], text=True).strip()
    (root / 'modal_hardware.json').write_text(json.dumps({'gpu': hardware, 'run_id': run_id}) + '\n')
    try:
        run_logged([str(root / '.venv/bin/python'), str(root / 'scripts/benchmark_research_device.py'), '--stage', 'run'],
                   root, root / 'benchmark.log', 1050)
        results = list((root / 'probe/results').glob('*/result.json'))
        if len(results) != 1:
            raise ValueError('Benchmark did not produce exactly one complete result')
        result = json.loads(results[0].read_text())
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode='w:gz') as archive:
            for path in sorted((root / 'probe/results').rglob('*')):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
            for name in ('benchmark.log', 'modal_hardware.json'):
                archive.add(root / name, arcname=name, recursive=False)
        return {'run_id': run_id, 'hardware': hardware, 'result': result, 'archive': content.getvalue()}
    finally:
        runs.commit()


@app.local_entrypoint()
def main():
    bundle = ROOT / 'data/probes/research_compute_bundle'
    payload = (ROOT / 'data/probes/research_compute_bundle.tar.gz').read_bytes()
    config = json.loads((bundle / 'config/game_note_research.json').read_text())
    manifest = json.loads((bundle / 'probe/manifest.json').read_text())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'data/probes/modal_benchmark' / run_id
    output.mkdir(parents=True, exist_ok=False)
    status = {'run_id': run_id, 'stage': 'model_download', 'gpu': 'H100', 'gpu_timeout_seconds': 1200,
              'payload_sha256': hashlib.sha256(payload).hexdigest()}
    (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    try:
        fetched = fetch_model.remote(config, manifest['model_files'])
        print(json.dumps(fetched), flush=True)
        status['stage'] = 'gpu_benchmark'
        (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
        response = benchmark.remote(payload, status['payload_sha256'], run_id)
    except Exception as error:
        status.update(stage='failed', error=str(error))
        (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
        raise
    (output / 'results.tar.gz').write_bytes(response.pop('archive'))
    (output / 'result.json').write_text(json.dumps(response, indent=2)+'\n')
    status['stage'] = 'complete'
    (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps({'local_results': str(output), 'hardware': response['hardware'],
                      'extraction_seconds': response['result']['extraction_seconds'],
                      'verification_seconds': response['result']['verification_seconds'],
                      'semantic_checker_all_passed': response['result']['semantic_checker_all_passed']}), flush=True)
