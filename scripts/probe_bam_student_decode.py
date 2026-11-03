"""Run/reconnect the separately authorized <=10-minute student decode probe."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import modal
from bam_student_decode_probe_cloud import app, measure, digest, write_json, ROOT


def collect(call, output):
    while True:
        try: response = call.get(timeout=45); break
        except Exception as error:
            if isinstance(error, TimeoutError) or type(error).__name__ == 'TimeoutError':
                print('Waiting for isolated student decode probe', flush=True); continue
            write_json(output / 'error.json', {'type': type(error).__name__, 'error': str(error)}); raise
    write_json(output / 'response.json', response)
    (output / 'execution.log').write_text(response['raw_log'])
    if response['result'] is not None: write_json(output / 'result.json', response['result'])
    print(json.dumps({'run_id': response['run_id'], 'error': response['error'], 'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config/bam_pipeline_v4_reference.json')
    parser.add_argument('--run-id', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    request = {'run_id': args.run_id, 'config': json.loads(Path(args.config).read_text()),
               'expected_worker': digest(ROOT / 'scripts/bam_training.py'),
               'expected_probe': digest(ROOT / 'scripts/bam_student_decode_probe_worker.py')}
    if (output / 'request.json').exists() and json.loads((output / 'request.json').read_text()) != request:
        raise ValueError('Local immutable probe request differs')
    write_json(output / 'request.json', request)
    if (output / 'response.json').exists():
        print('Probe response already retained; no additional compute.'); return
    if (output / 'call.json').exists():
        collect(modal.FunctionCall.from_id(json.loads((output / 'call.json').read_text())['call_id']), output); return
    if (output / 'spawn_intent.json').exists():
        raise RuntimeError('Prior spawn intent lacks call ID; inspect orphan call before any retry')
    with modal.enable_output(), app.run(detach=True):
        write_json(output / 'spawn_intent.json', {'utc': datetime.now(timezone.utc).isoformat()})
        call = measure.spawn(**request)
        write_json(output / 'call.json', {'call_id': call.object_id, 'started_utc': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'call_id': call.object_id}), flush=True)
        collect(call, output)


if __name__ == '__main__': main()
