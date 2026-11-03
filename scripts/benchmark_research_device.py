"""Prepare a label-free portable GPU probe, or run it in an isolated checkout."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import run_game_note_research as research
from check_research_quality import GID, challenge_cases

ROOT = research.ROOT
BUNDLE = ROOT / 'data/probes/research_compute_bundle'


def prepare():
    config = research.read_json(research.CONFIG)
    protocol = research.verify_protocol(config)
    units = [u for u in research.read_rows(research.DATA / 'units.jsonl') if u['game']['game_id'] == GID]
    if len(units) != 15 or any(u['game']['split'] != 'train' for u in units):
        raise ValueError('Compute probe requires the complete reviewed 15-page training packet')
    if BUNDLE.exists():
        raise ValueError('Probe bundle already exists; preserve it or choose a fresh snapshot explicitly')
    files = ['scripts/benchmark_research_device.py', 'scripts/run_game_note_research.py',
             'scripts/game_note_tables.py', 'scripts/check_research_quality.py',
             'config/game_note_research.json', 'reports/game_note_research/protocol.json',
             'requirements-research-cuda.txt']
    for relative in files:
        dest = BUNDLE / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    research.write_rows(BUNDLE / 'probe/units.jsonl', units)
    files.append('probe/units.jsonl')
    reference = {}
    for unit in units:
        response = research.read_json(research.DATA / 'extract' / (unit['id'] + '.json'))
        if response['status'] != 'complete' or response['request_sha256'] != unit['request_sha256']:
            raise ValueError('Missing completed local probe reference')
        reference[unit['id']] = {'text_sha256': research.value_hash(response['text']),
                                 'response_protocol_sha256': response['protocol_sha256']}
    research.write_json(BUNDLE / 'probe/local_reference.json', reference)
    files.append('probe/local_reference.json')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'game_id': GID,
                'split': 'train', 'units': len(units), 'input_tokens': sum(u['input_tokens'] for u in units),
                'files': {p: research.sha(BUNDLE / p) for p in files},
                'model_files': protocol['model_sha256'], 'versions': protocol['versions'],
                'scope': 'Compute and factual-error probe only. No target labels, booster probabilities, credentials, provisioning or main-run output files.'}
    research.write_json(BUNDLE / 'probe/manifest.json', manifest)
    print(json.dumps({'bundle': str(BUNDLE), 'files': len(files) + 1,
                      'bytes_without_model': sum(p.stat().st_size for p in BUNDLE.rglob('*') if p.is_file()),
                      'model_download_gib': 7.75, 'remote_compute_tested': False}, indent=2))


def check_bundle():
    manifest = research.read_json(ROOT / 'probe/manifest.json')
    for path, digest in manifest['files'].items():
        if research.sha(ROOT / path) != digest:
            raise ValueError('Compute-probe file changed: ' + path)
    units = research.read_rows(ROOT / 'probe/units.jsonl')
    if len(units) != 15 or any(u['game']['split'] != 'train' or u['game']['game_id'] != GID for u in units):
        raise ValueError('Compute probe contains a different game or a nontraining example')
    for unit in units:
        if research.value_hash(research.extraction_messages(unit)) != unit['request_sha256']:
            raise ValueError('Probe prompt differs from the current frozen research request')
    return manifest, units


def run(download_model=False):
    manifest, units = check_bundle()
    config = research.read_json(research.CONFIG)
    # Remote images may mount the checkout's .venv as a directory symlink.
    # Verify the active environment itself, not the spelling of its executable.
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        raise ValueError('Use the probe checkout .venv/bin/python')
    for package, expected in manifest['versions'].items():
        if version(package) != expected:
            raise ValueError('Different runtime version: ' + package)
    if download_model:
        from huggingface_hub import snapshot_download
        snapshot_download(config['model_id'], revision=config['model_revision'],
                          local_dir=ROOT / config['model_path'], token=False,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'LICENSE', 'README.md'])
    for path, digest in manifest['model_files'].items():
        if research.sha(ROOT / path) != digest:
            raise ValueError('Probe checkpoint differs from the frozen model: ' + path)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'probe/results' / run_id
    if output.exists():
        raise ValueError('Refusing to overwrite a previous device probe')
    output.mkdir(parents=True)
    started = time.monotonic()
    mx, model, tokenizer = research.load_runtime(config)
    if mx.default_device() != mx.gpu:
        raise ValueError('GPU required: refusing an accidental CPU-only benchmark')
    identity = {'platform': platform.platform(), 'python': platform.python_version(),
                'device': str(mx.default_device()), 'device_info': mx.device_info(),
                'versions': {p: version(p) for p in manifest['versions']},
                'model_revision': config['model_revision'], 'model_hashes_verified': True,
                'probe_manifest_sha256': research.sha(ROOT / 'probe/manifest.json')}
    research.write_json(output / 'runtime.json', identity)
    research.DATA = output
    extraction_started = time.monotonic()
    research.execute_tasks(units, 'extract', config, model, tokenizer)
    extraction_seconds = time.monotonic() - extraction_started
    tasks = research.verification_tasks(units, config)
    verification_started = time.monotonic()
    research.execute_tasks(tasks, 'verify', config, model, tokenizer)
    verification_seconds = time.monotonic() - verification_started
    challenge, expected = challenge_cases(units)
    research.DATA = output / 'challenges'
    research.execute_tasks(challenge, 'verify', config, model, tokenizer)
    cases = []
    for task in challenge:
        decision = research.read_json(research.DATA / 'verify' / (task['id'] + '.json'))['parsed']['decisions'][0]
        cases.append({**decision, 'expected_supported': expected[decision['id']],
                      'passed': (decision['verdict'] == 'supported') == expected[decision['id']]})
    # Exercise the exact hidden-state/cache operations needed by the encoder.
    # This is a kernel check using a research prompt, not a forecast or head fit.
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    ids = research.encode_messages(tokenizer, research.extraction_messages(units[0]))
    for start in range(0, len(ids), config['prefill_step_size']):
        hidden = model.model(mx.array(ids[start:start + config['prefill_step_size']])[None], cache=cache)
        mx.eval(hidden)
    vector = hidden[0, -1]
    if vector.shape != (5120,) or not bool(mx.all(mx.isfinite(vector)).item()):
        raise ValueError('Encoder kernel probe returned an invalid hidden-state vector')
    reference = research.read_json(ROOT / 'probe/local_reference.json')
    exact = sum(research.value_hash(research.read_json(output / 'extract' / (u['id']+'.json'))['text'])
                == reference[u['id']]['text_sha256'] for u in units)
    result = {**identity, 'extraction_seconds': extraction_seconds,
              'verification_seconds': verification_seconds, 'total_seconds_including_challenges': time.monotonic()-started,
              'units': len(units), 'source_pages': 15, 'verification_requests': len(tasks),
              'peak_memory_gib': mx.get_peak_memory()/1024**3, 'finite_5120_number_encoder_probe': True,
              'exact_local_extraction_matches': exact, 'factual_challenges': cases,
              'semantic_checker_all_passed': all(c['passed'] for c in cases),
              'schema_and_source_checks': 'passed', 'main_experiment_modified': False,
              'scope': 'Benchmark only. Inspect source quality and hardware differences before adopting this backend. A completed benchmark is not factual certification or a trained forecaster.'}
    research.write_json(output / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'check', 'run'))
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'check':
        manifest, units = check_bundle()
        print(json.dumps({'files_checked': len(manifest['files']), 'training_units_checked': len(units)}))
    else:
        run(args.download_model)
