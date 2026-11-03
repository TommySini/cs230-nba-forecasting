"""Resume the research-only snapshot on one Modal H100 and return audited outputs."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import modal
from modal_research import image, models, runs, unpack, run_logged, digest

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('cs230-nba-full-note-research-batch')


@app.function(image=image, gpu='H100', volumes={'/models': models, '/runs': runs},
              cpu=4, memory=32768, timeout=7200, max_containers=1,
              single_use_containers=True, retries=0)
def research_batch(payload, expected_sha, run_id):
    root = unpack(payload, expected_sha, Path('/runs/research') / run_id, 'nba_research')
    source = json.loads((root / 'transfer_manifest.json').read_text())
    for name, expected in source['files'].items():
        if digest(root / name) != expected:
            raise ValueError('Research transfer file changed: ' + name)
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                         '--format=csv,noheader'], text=True).strip()
    version_code = "import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ('mlx','mlx-lm','transformers','numpy')}))"
    versions = json.loads(subprocess.check_output([str(root / '.venv/bin/python'), '-c', version_code], text=True))
    if versions != source['runtime_versions']:
        raise ValueError('Cloud research runtime versions differ from the frozen protocol')
    execution = {'run_id': run_id, 'gpu': hardware, 'versions': versions,
                 'input_archive_sha256': expected_sha, 'model_revision': source['model_revision'],
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                 'gpu_timeout_seconds': 7200, 'labels_or_booster_inputs_uploaded': False}
    log = root / 'research.log'
    try:
        run_logged([str(root / '.venv/bin/python'), str(root / 'scripts/run_research_experiment.py'),
                    '--stage', 'research'], root, log, 6900)
        execution['status'] = 'complete'
    except Exception as error:
        execution.update(status='failed', error=str(error))
    execution['finished_utc'] = datetime.now(timezone.utc).isoformat()
    # Return only research artifacts and the log. Weight/env symlinks, raw feeds,
    # account details and unrelated cloud files cannot enter this archive.
    paths = []
    for folder in ('extract', 'verify', 'attempts', 'reports'):
        paths.extend(p for p in (root / 'data/processed/game_note_research' / folder).rglob('*') if p.is_file())
    for name in ('progress.json', 'research_manifest.json', 'coverage.json'):
        path = root / 'reports/game_note_research' / name
        if path.is_file():
            paths.append(path)
    execution['output_files'] = {str(p.relative_to(root)): digest(p) for p in sorted(paths)}
    execution['new_or_changed_response_files'] = {
        name: value for name, value in execution['output_files'].items()
        if '/reports/' not in name and source['files'].get(name) != value}
    (root / 'modal_execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode='w:gz') as archive:
        for path in sorted(paths) + [log, root / 'modal_execution.json']:
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    runs.commit()
    return {'execution': execution, 'archive': content.getvalue()}


@app.local_entrypoint()
def main(snapshot: str):
    path = Path(snapshot).resolve()
    if not path.is_relative_to(ROOT / 'data/probes/cloud_research_transfer') or path.suffixes[-2:] != ['.tar', '.gz']:
        raise ValueError('Use an explicitly exported research-only snapshot')
    completed = [p for p in (ROOT / 'data/probes/modal_benchmark').glob('*/result.json')
                 if json.loads(p.read_text())['result'].get('finite_5120_number_encoder_probe')]
    if not completed:
        raise ValueError('A successful bounded Modal compute probe is required first')
    benchmark = json.loads(sorted(completed)[-1].read_text())['result']
    if benchmark['schema_and_source_checks'] != 'passed' or sum(not c['passed'] for c in benchmark['factual_challenges']) > 2:
        raise ValueError('Compute probe failed or factual checks regressed; inspect before full execution')
    config = json.loads((ROOT / 'config/game_note_research.json').read_text())
    if benchmark['model_revision'] != config['model_revision'] or not benchmark['model_hashes_verified']:
        raise ValueError('Benchmark did not verify the current pinned checkpoint')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'data/probes/modal_research' / run_id
    output.mkdir(parents=True, exist_ok=False)
    payload = path.read_bytes()
    status = {'run_id': run_id, 'stage': 'cloud_research', 'snapshot': str(path.relative_to(ROOT)),
              'snapshot_sha256': hashlib.sha256(payload).hexdigest(), 'gpu': 'H100', 'timeout_seconds': 7200}
    (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    try:
        response = research_batch.remote(payload, status['snapshot_sha256'], run_id)
    except Exception as error:
        status.update(stage='failed', error=str(error))
        (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
        raise
    archive = response.pop('archive')
    (output / 'results.tar.gz').write_bytes(archive)
    (output / 'execution.json').write_text(json.dumps(response['execution'], indent=2)+'\n')
    status.update(stage=response['execution']['status'], results_sha256=hashlib.sha256(archive).hexdigest())
    (output / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps({'local_results': str(output), 'status': status['stage'],
                      'returned_files': len(response['execution']['output_files'])}), flush=True)
    if status['stage'] != 'complete':
        raise RuntimeError('Remote research stopped; partial responses are preserved for inspection and resumption')
