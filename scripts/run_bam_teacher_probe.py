"""Resume the fixed, bounded actual-claim teacher diagnostic with full provenance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import modal

from bam_training import file_hash, value_hash, write_json
from run_bam_pipeline import DurableCloud, immutable_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    payload = json.loads((root / 'launch_payload.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    model_manifest = json.loads((root / 'model_manifest.json').read_text())
    assert value_hash(payload['requests']) == manifest['requests_sha256']
    assert value_hash(payload['config']) == manifest['config_sha256']
    assert payload['model_spec'] == manifest['model'] == model_manifest['model_spec']
    assert file_hash(Path(__file__).with_name('bam_teacher_probe_worker.py')) == manifest['worker_sha256']
    assert file_hash(Path(__file__).with_name('bam_training.py')) == manifest['training_tokenizer_helper_sha256']
    assert payload['expected_worker_hash'] == manifest['worker_sha256']
    deployment = {'cloud_sha256': file_hash(Path(__file__).with_name('bam_teacher_probe_cloud.py')),
                  'controller_sha256': file_hash(__file__), 'at_utc': datetime.now(timezone.utc).isoformat(),
                  'scientific_manifest_sha256': file_hash(root / 'manifest.json')}
    deployment_path = root / 'deployments' / (deployment['cloud_sha256'] + '.json')
    if not deployment_path.exists():
        immutable_json(deployment_path, deployment)
    cloud = DurableCloud(root / 'calls', retry_failed=args.retry_failed,
        functions={'compare': modal.Function.from_name('cs230-nba-bam-teacher-probe', 'compare', environment_name='main')})
    result = cloud.invoke('compare', [payload[k] for k in
        ('requests', 'model_spec', 'run_id', 'config', 'expected_worker_hash')],
        {'launch_payload_sha256': file_hash(root / 'launch_payload.json'),
         'manifest_sha256': file_hash(root / 'manifest.json')})
    identity = result['identity']
    assert identity['model'] == model_manifest
    assert identity['worker_sha256'] == manifest['worker_sha256']
    assert identity['tokenizer_helper_sha256'] == manifest['training_tokenizer_helper_sha256']
    assert identity['settings']['tensor_parallel_size'] == 2
    expected = {r['request_id']: r for r in payload['requests']}
    tokens = {r['request_id']: r for r in manifest['tokenization']}
    assert len(result['responses']) == len(expected)
    assert {r['request_id'] for r in result['responses']} == set(expected)
    for response in result['responses']:
        rid = response['request_id']
        assert response['request_sha256'] == value_hash(expected[rid])
        assert response['response_sha256'] == value_hash({k: v for k, v in response.items() if k != 'response_sha256'})
        assert response['execution_sha256'] == value_hash(identity)
        assert response['prompt_token_ids_sha256'] == tokens[rid]['prompt_token_ids_sha256']
        assert response['prompt_tokens'] == tokens[rid]['prompt_tokens']
    immutable_json(root / 'result.json', result)
    write_json(root / 'verification.json', {'status': 'passed', 'result_sha256': file_hash(root / 'result.json'),
        'scientific_manifest_sha256': file_hash(root / 'manifest.json'), 'responses': len(expected),
        'checks': ['checkpoint', 'worker', 'tokenizer_helper', 'tensor_parallelism',
                   'complete_request_set', 'request_and_response_hashes', 'runtime_hash', 'local_prompt_tokenization']})
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
