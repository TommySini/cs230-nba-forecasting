"""One isolated, bounded H100 throughput probe; no production app deployment."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import threading

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('cs230-nba-bam-training-probe')
models = modal.Volume.from_name('cs230-nba-research-models', version=2)
runs = modal.Volume.from_name('cs230-nba-research-runs', version=2)
if modal.is_local():
    from bam_cloud import training_image
    image = training_image.add_local_file(ROOT / 'scripts/bam_training_probe_worker.py', '/opt/bam_training_probe_worker.py', copy=True)
else:
    image = modal.Image.debian_slim(python_version='3.12')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def _run_logged(command, cwd, path, seconds, commit):
    with path.open('a') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        timer = threading.Timer(seconds, process.kill); timer.start()
        try:
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True); commit()
            if process.wait():
                raise RuntimeError('Probe child failed; inspect preserved execution.log')
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill(); process.wait()


@app.function(image=image, gpu='H100', volumes={'/models': models, '/runs': runs},
              cpu=6, memory=65536, timeout=1200, min_containers=0, max_containers=1,
              single_use_containers=True, retries=0)
def measure(run_id, config, expected_worker, expected_probe):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', run_id):
        raise ValueError('Invalid probe ID')
    models.reload(); runs.reload()
    if digest('/opt/bam_training.py') != expected_worker or digest('/opt/bam_training_probe_worker.py') != expected_probe:
        raise ValueError('Isolated worker identity changed')
    root = Path('/runs/bam/probes') / run_id
    root.mkdir(parents=True, exist_ok=True)
    request = {'config': config, 'worker_sha256': expected_worker, 'probe_sha256': expected_probe}
    prior = root / 'request.json'
    if prior.exists() and json.loads(prior.read_text()) != request:
        raise ValueError('Probe ID reused with different request')
    write_json(prior, request); write_json(root / 'config.json', config); runs.commit()
    error = None
    try:
        _run_logged(['/opt/bam-venv/bin/python', '-u', '/opt/bam_training_probe_worker.py',
                     '--root', str(root), '--config', str(root / 'config.json')], root,
                    root / 'execution.log', seconds=1050, commit=runs.commit)
    except Exception as exc:
        error = str(exc)
    finally:
        runs.commit()
    return {'run_id': run_id, 'error': error,
            'result': json.loads((root / 'result.json').read_text()) if (root / 'result.json').exists() else None,
            'raw_log': (root / 'execution.log').read_text()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/bam_pipeline_v1.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--total-lengths', nargs='+', type=int, default=[4608, 8704])
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--timed-steps', type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    if (not all(1024 <= value <= 16384 for value in args.total_lengths)
            or not 1 <= args.warmup_steps <= 2 or not 1 <= args.timed_steps <= 3):
        raise ValueError('Probe lengths or repetitions exceed bounded scope')
    config['throughput_probe'] = {'total_lengths': args.total_lengths,
                                  'warmup_steps': args.warmup_steps, 'timed_steps': args.timed_steps}
    request = {'run_id': args.run_id, 'config': config,
               'expected_worker': digest(ROOT / 'scripts/bam_training.py'),
               'expected_probe': digest(ROOT / 'scripts/bam_training_probe_worker.py')}
    write_json(output / 'request.json', request)
    with modal.enable_output():
        with app.run(detach=True):
            call = measure.spawn(**request)
            write_json(output / 'call.json', {'call_id': call.object_id, 'started_utc': datetime.now(timezone.utc).isoformat()})
            print(json.dumps({'call_id': call.object_id}), flush=True)
            while True:
                try:
                    result = call.get(timeout=45)
                    break
                except Exception as exc:
                    if isinstance(exc, TimeoutError) or type(exc).__name__ == 'TimeoutError':
                        print('Waiting for isolated throughput probe', flush=True)
                        continue
                    write_json(output / 'error.json', {'type': type(exc).__name__, 'error': str(exc)})
                    raise
            write_json(output / 'response.json', result)
            (output / 'execution.log').write_text(result['raw_log'])
            if result['result']:
                write_json(output / 'result.json', result['result'])
            print(json.dumps({'run_id': args.run_id, 'error': result['error'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
