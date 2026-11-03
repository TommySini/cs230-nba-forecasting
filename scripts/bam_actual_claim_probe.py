"""Frozen actual-output claim isolation diagnostic; never launches inference."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import bam_data
import bam_research as research
import bam_training as training

ROOT = Path(__file__).resolve().parents[1]
RUN = 'bam_reference_v4_20261102'
SCHEMA = {key: value for key, value in research.REVIEW_SCHEMA['decisions'][0].items()}


def immutable(path, raw):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Frozen diagnostic differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()


def isolated_request(game, artifact, finding, *, extra_pages=()):
    if game['split'] != 'train':
        raise ValueError('Actual-output diagnostic is training-only')
    wanted = {(c['source_id'], c['page']) for c in finding['citations']} | set(extra_pages)
    context = [{**research._source_metadata(source), 'pages': [
        {**research._page_metadata(source, page), 'lines': deepcopy(page['lines'])}
        for page in source['pages'] if (source['source_id'], page['page']) in wanted]}
        for source in game['sources'] if any(sid == source['source_id'] for sid, _ in wanted)]
    actual = {(source['source_id'], page['page']) for source in context for page in source['pages']}
    if wanted != actual:
        raise ValueError('Requested source page missing')
    # Keep the production compact-review rubric, except its whole-report task.
    instruction = research.REVIEW_INSTRUCTION
    start = instruction.index('Assess the entire report too:')
    end = instruction.index('Every supplied ID', start)
    instruction = instruction[:start] + instruction[end:]
    instruction += (' This is an isolated finding check. Assess this original finding including fact_date, '
                    'interpretation, counterevidence and uncertainty. No other generated findings are supplied. '
                    'Do not make a whole-report coverage judgment. Return one decision object with the exact original id.')
    payload = {'game': research.metadata(game), 'scope': 'one actual finding; no whole-report coverage claim',
        'candidate': deepcopy(finding), 'pregame_source_context': context,
        'original_provenance': {'run_id': RUN, 'artifact_hash': artifact['artifact_hash'],
            'request_id': artifact['request_id'], 'stage': artifact['stage'], 'model': artifact['model']},
        'output_schema': SCHEMA}
    request = {'stage': 'actual_claim_review', 'game_id': game['game_id'], 'split': 'train',
        'parent_hashes': [artifact['artifact_hash']], 'original_finding_id': finding['id'],
        'messages': [{'role': 'system', 'content': instruction},
                     {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
        'max_tokens': 4500, 'output_schema': SCHEMA}
    request['request_id'] = research.value_hash(request)
    return request


def check_decision(request, text):
    decision = json.loads(text) if isinstance(text, str) else deepcopy(text)
    if not isinstance(decision, dict) or set(decision) != set(SCHEMA):
        raise ValueError('Incomplete or unexpected decision schema')
    if decision['id'] != request['original_finding_id']:
        raise ValueError('Foreign candidate ID')
    if decision['verdict'] not in ('supported', 'unsupported', 'unclear'):
        raise ValueError('Missing verdict')
    if any(not isinstance(decision[k], str) or not decision[k].strip() for k in ('source_reading', 'reason')):
        raise ValueError('Missing source assessment')
    payload = json.loads(request['messages'][1]['content'])
    visible = {(s['source_id'], p['page'], line['line']): line['text']
               for s in payload['pregame_source_context'] for p in s['pages'] for line in p['lines']}
    citations = decision['required_citations']
    if not isinstance(citations, list) or not citations:
        raise ValueError('Missing required source references')
    required = set()
    for cite in citations:
        if not isinstance(cite, dict) or set(cite) != {'source_id', 'page', 'lines'} or type(cite['page']) is not int:
            raise ValueError('Malformed source reference')
        if not isinstance(cite['lines'], list) or not cite['lines'] or any(type(n) is not int for n in cite['lines']):
            raise ValueError('Malformed source lines')
        keys = {(cite['source_id'], cite['page'], n) for n in cite['lines']}
        if not keys <= visible.keys():
            raise ValueError('Source reference absent from visible canonical evidence')
        required.update(keys)
    original = {(c['source_id'], c['page'], n) for c in payload['candidate']['citations'] for n in c['lines']}
    missing = required - original
    decision['model_verdict'] = decision['verdict']
    decision['missing_original_citations'] = sorted(missing)
    if missing and decision['verdict'] == 'supported':
        decision['verdict'] = 'unclear'
    decision['canonical_evidence'] = [{'source_id': s, 'page': p, 'line': n, 'text': visible[s, p, n]} for s, p, n in sorted(required)]
    return decision


def prepare(output):
    run = ROOT / 'data/processed/bam_runs' / RUN
    artifacts = [json.loads(p.read_text()) for p in (run / 'artifacts').glob('*.json')]
    def find(prefix, fid):
        matches = [a for a in artifacts if a['artifact_hash'].startswith(prefix)]
        if len(matches) != 1: raise ValueError('Ambiguous original artifact')
        artifact = matches[0]
        return artifact, next(f for f in artifact['output']['findings'] if f['id'] == fid)
    # Source judgments fixed before inference. Candidates are byte-for-byte
    # copies of real findings, never fabricated or rewritten benchmark claims.
    selected = [
        ('0a2ced4c15a6', 'matchup-f1', 'unsupported', 'Source p38L36 is TOTALS; PITP means paint, not transition.', ()),
        ('0a2ced4c15a6', 'matchup-f2', 'unsupported', 'Source p38L36 gives total fast-break points, not an average.', ()),
        ('0a2ced4c15a6', 'matchup-f3', 'unsupported', 'Source p38L36 gives total bench points, not an average.', ()),
        ('0a2ced4c15a6', 'matchup-f4', 'unsupported', 'Source p38L36 gives total points off turnovers, not an average.', ()),
        ('2c6549f9c12d', 'recent_changes_rest_workload-f4', 'unsupported', 'p12L10–11 says at least20/6/8; p12L54 and p1 give25.7PPG.', (1,)),
        ('b9ce3ab9b36b', 'recent_changes_rest_workload-f2', 'unsupported', 'fact_date2023 unsupported; p37L10 explicitly dates Lakers game10/30/2024.', (37,)),
        ('b9ce3ab9b36b', 'recent_changes_rest_workload-f3', 'unsupported', 'fact_date2023 unsupported; p37L12 explicitly dates Bucks game11/02/2024.', (37,)),
        ('8f771a22289a', 'matchup-f1', 'unsupported', 'p7L59–60 says defensive rating7TH, not2ND.', ()),
        ('8cb15b5f1b1d', 'recent_changes_rest_workload-f1', 'supported', 'Named Adebayo table p10L20–21/26 gives38:36,13PTS,20REB,7AST; prior-game caveat preserved.', ()),
        ('917353aaeace', 'recent_changes_rest_workload-f3', 'supported', 'Named Herro table p11L67–68/73 gives36:49,31PTS,4REB,3AST; prior-game caveat preserved.', ()),
        ('df48308dbb', 'matchup-f1', 'supported', 'Explicit76ers current-season team table/header/row gives all stated published averages and fractions.', ()),
        ('bc3c2f6aa4c6', 'availability_rotation-f6', 'supported', 'Explicit Cunningham current-season card/header/row gives25.7/6.3/9.5 and remaining stated values.', ()),
        ('8cb15b5f1b1d', 'recent_changes_rest_workload-f2', 'supported', 'p4L9–10 explicitly states current road back-to-back and season counts; interpretation remains uncertain.', ()),
        ('8f771a22289a', 'matchup-f5', 'supported', 'p11L2–4 prose explicitly names Mobley and gives all stated season averages, shooting fractions and ranks.', ()),
        ('b0a5fd030946', 'matchup-f7', 'supported', 'p3L32 explicitly says Nuggets won nine of last ten against Pistons; interpretation remains historical and uncertain.', ()),
        ('ed02ab2e3f12', 'sha256:5cb46d15c6de10c09984146227e6e3b076e544e87dabf0a9686a8ac0e0ad3a75:p4:c0-f3', 'supported', 'p4L22–30 explicitly separates Herro57combined and52regular-season streaks; future continuation is uncertain.', ()),
    ]
    config = json.loads((ROOT / 'config/bam_pipeline_v4_reference.json').read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/models/bam_tokenizer' / config['teacher']['model_revision'], local_files_only=True, trust_remote_code=False)
    requests, expected = [], []
    snapshot = ROOT / 'data/processed/bam_pipeline/bam_v1_20261101_initial'
    for prefix, fid, verdict, rationale, extras in selected:
        artifact, finding = find(prefix, fid)
        game = bam_data.load_snapshot_game(snapshot, artifact['game_id'])
        request = isolated_request(game, artifact, finding, extra_pages=[(game['sources'][0]['source_id'], p) for p in extras])
        tokens = research.count_request_tokens(request, tokenizer)
        if tokens + request['max_tokens'] + 4096 > 32768:
            raise ValueError('Complete isolated context does not fit; do not truncate')
        requests.append(request)
        expected.append({'request_id': request['request_id'], 'expected_verdict': verdict,
                         'rationale': rationale, 'source_artifact_hash': artifact['artifact_hash'],
                         'finding_id': fid, 'game_id': game['game_id'], 'prompt_tokens': tokens})
    requests.sort(key=lambda x: x['request_id'])
    output = Path(output)
    files = {'requests.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests).encode(),
             'expected.json': encode(expected)}
    for name, raw in files.items(): immutable(output / name, raw)
    manifest = {'version': 1, 'source_run': RUN, 'scope': 'Actual training findings; four prose and four canonical-table positive controls; eight actual invalid findings. No whole-report claim.',
        'inference_not_launched': True, 'prepare_script_sha256': training.file_hash(__file__),
        'artifacts': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
        'config_sha256': research.value_hash(config), 'model': {k: config['teacher'][k] for k in ('model_id', 'model_revision', 'quantization')},
        'limitation': 'Isolating claims AND supplying full relevant pages changes context; this is not a causal estimate of attention alone. Cases selected for source diagnosis, not random accuracy.'}
    immutable(output / 'manifest.json', encode(manifest))
    immutable(output / 'launch_payload.json', encode({'requests': requests, 'config': config, 'model_spec': manifest['model'],
        'run_id': 'bam_actual_claim_isolation_v1_20261102', 'expected_worker_hash': training.file_hash(ROOT / 'scripts/bam_inference.py')}))
    print(json.dumps({'directory': str(output), 'requests': len(requests), 'manifest_sha256': training.file_hash(output / 'manifest.json'),
                      'max_prompt_tokens': max(e['prompt_tokens'] for e in expected)}))


def score(directory, result_path, output):
    directory = Path(directory); manifest = json.loads((directory / 'manifest.json').read_text())
    for name, sha in manifest['artifacts'].items():
        if training.file_hash(directory / name) != sha: raise ValueError('Frozen cases changed')
    requests = {r['request_id']: r for r in training.read_rows(directory / 'requests.jsonl')}
    expected = {e['request_id']: e for e in json.loads((directory / 'expected.json').read_text())}
    result = json.loads(Path(result_path).read_text()); responses = result['responses']
    ids = [r['request_id'] for r in responses]
    if len(ids) != len(set(ids)) or set(ids) != set(requests): raise ValueError('Response coverage differs')
    rows = []
    for response in responses:
        rid = response['request_id']; request = requests[rid]
        row = {'request_id': rid, 'expected': expected[rid]['expected_verdict'], 'raw_response': response}
        try:
            if response.get('status') != 'complete' or response.get('finish_reason') not in ('stop', 'end_turn'): raise ValueError('Incomplete response')
            if response.get('request_sha256') != research.value_hash(request): raise ValueError('Response request identity differs')
            decision = check_decision(request, response['text'])
            row.update(valid=True, decision=decision, correct=decision['verdict'] == row['expected'])
        except (ValueError, KeyError, TypeError) as exc:
            row.update(valid=False, correct=False, error=str(exc))
        rows.append(row)
    summary = {'cases': len(rows), 'valid': sum(r['valid'] for r in rows), 'correct': sum(r['correct'] for r in rows),
        'false_accepts': sum(r['valid'] and r['expected'] != 'supported' and r['decision']['verdict'] == 'supported' for r in rows),
        'false_exclusions': sum(r['expected'] == 'supported' and (not r['valid'] or r['decision']['verdict'] != 'supported') for r in rows),
        'result_sha256': training.file_hash(result_path), 'manifest_sha256': training.file_hash(directory / 'manifest.json'),
        'scope': manifest['scope'], 'rows': rows}
    immutable(output, encode(summary)); print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('score'); s.add_argument('--directory', type=Path, required=True); s.add_argument('--result', type=Path, required=True); s.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    else: score(args.directory, args.result, args.output)
