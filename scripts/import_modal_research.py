"""Check a complete private cloud return before promoting research artifacts."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import run_game_note_research as r


def validate_execution(response, relative, registry=None, protocol=None):
    registry = registry if registry is not None else r.read_json(r.DATA/'remote_import.json')
    expected = registry['response_files'].get(relative)
    if expected is None or r.sha(r.ROOT/relative) != expected:
        raise ValueError('Imported research response changed: '+relative)
    digest = response.get('execution_sha256')
    if not digest:
        raise ValueError('Missing cloud execution identity')
    identity = r.read_json(r.DATA/'executions'/(digest+'.json'))
    if r.value_hash(identity) != digest or identity != registry['executions'].get(digest):
        raise ValueError('Cloud execution metadata changed')
    protocol = protocol if protocol is not None else r.verify_protocol(r.read_json(r.CONFIG))
    if identity['conversion']['source_files'] != protocol['model_sha256']:
        raise ValueError('Cloud run used a different original checkpoint')


def main(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(r.ROOT/'data/probes/modal_vllm'):
        raise ValueError('Use a saved Modal run directory')
    execution = r.read_json(directory/'execution.json')
    if execution['benchmark'] or execution['status'] != 'complete':
        raise ValueError('Only complete full-research runs can enter the experiment')
    plan = r.read_json(r.REPORT/'cloud_execution_plan.json')
    protocol = r.verify_protocol(r.read_json(r.CONFIG),check_model=True)
    members = {}
    with tarfile.open(fileobj=io.BytesIO((directory/'results.tar.gz').read_bytes()),mode='r:gz') as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or member.name in members:
                raise ValueError('Unsafe or duplicate return path')
            members[member.name] = archive.extractfile(member).read()
    for name, expected in execution['output_files'].items():
        import hashlib
        if name not in members or hashlib.sha256(members[name]).hexdigest() != expected:
            raise ValueError('Cloud return checksum differs: '+name)
    # The cloud must not rewrite the frozen sources, protocol or gate.
    for name in ('reports/game_note_research/protocol.json','reports/game_note_research/quality_gate.json',
                 'data/processed/game_note_research/units.jsonl','data/processed/game_note_research/coverage.json'):
        if members[name] != (r.ROOT/name).read_bytes():
            raise ValueError('Cloud changed frozen experiment input: '+name)
    identities = {}
    for name, content in members.items():
        if name.startswith('data/processed/game_note_research/executions/'):
            identity = json.loads(content)
            digest = r.value_hash(identity)
            allowed_workers = {plan['worker_sha256'], *plan.get('compatible_worker_sha256', [])}
            if Path(name).stem != digest or identity['worker_sha256'] not in allowed_workers:
                raise ValueError('Unapproved execution implementation')
            if identity['conversion'] != plan['conversion'] or identity['settings'] != plan['settings'] or identity['versions'] != plan['versions']:
                raise ValueError('Cloud backend differs from reviewed training-only benchmark')
            if identity['conversion']['source_files'] != protocol['model_sha256']:
                raise ValueError('Different source checkpoint')
            identities[digest] = identity
    responses = {}
    for name, content in members.items():
        parts = Path(name).parts
        if len(parts)!=5 or parts[:3]!=('data','processed','game_note_research') or parts[3] not in ('extract','verify'):
            continue
        response = json.loads(content)
        if response['status']!='complete' or response['finish_reason']!='stop' or response['protocol_sha256'] not in r.response_protocols():
            raise ValueError('Incomplete or foreign cloud response')
        if response.get('execution_sha256'):
            if response['execution_sha256'] not in identities:
                raise ValueError('Unknown response backend')
            responses[name] = execution['output_files'][name]
        else:
            # Legacy results must be exactly the already completed local cache.
            if not (r.ROOT/name).exists() or content != (r.ROOT/name).read_bytes():
                raise ValueError('Unattributed new response or changed local result')
        if (r.ROOT/name).exists():
            previous = r.read_json(r.ROOT/name)
            if previous.get('status')=='complete' and previous != response:
                raise ValueError('Refusing to replace a completed local result')
    registry = {'run_id':execution['run_id'],'plan_sha256':r.sha(r.REPORT/'cloud_execution_plan.json'),
                'archive_sha256':r.sha(directory/'results.tar.gz'),'executions':identities,'response_files':responses}
    # All file/checkpoint/backend checks above pass before the first promotion.
    allowed = ('data/processed/game_note_research/extract/', 'data/processed/game_note_research/verify/',
               'data/processed/game_note_research/attempts/', 'data/processed/game_note_research/executions/',
               'data/processed/game_note_research/reports/')
    for name, content in members.items():
        if name.startswith(allowed) or name in ('reports/game_note_research/research_manifest.json',
                'reports/game_note_research/coverage.json','reports/game_note_research/progress.json'):
            target = r.ROOT/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
    r.write_json(r.DATA/'remote_import.json',registry)
    print(json.dumps({'imported_cloud_responses':len(responses),'executions':len(identities),
                      'next':'Validate every response and source against the original local PDFs before forecasting.'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    main(parser.parse_args().directory)
