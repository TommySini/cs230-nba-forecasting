"""Resumable full-note research, evidence checks, head training and paired evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import run_game_note_research as research

ROOT = research.ROOT
STATUS = ROOT / 'data/processed/research_experiment/status.json'


def research_by_game():
    config = research.read_json(research.CONFIG)
    research.verify_protocol(config, check_model=True)
    gate = research.read_json(research.REPORT / 'quality_gate.json')
    if not gate.get('ready_for_source_evidence_run') or gate['protocol_sha256'] != research.sha(research.REPORT / 'protocol.json'):
        raise ValueError('Research quality gate failed or is stale')
    units = research.read_rows(research.DATA / 'units.jsonl')
    games = defaultdict(list)
    for unit in units:
        games[unit['game']['game_id']].append(unit)
    mx, model, tokenizer = research.load_runtime(config)
    progress = []
    for i, (gid, game_units) in enumerate(games.items()):
        print(f'GAME {i+1}/{len(games)} {gid}', flush=True)
        research.execute_tasks(game_units, 'extract', config, model, tokenizer)
        tasks = research.verification_tasks(game_units, config)
        research.execute_tasks(tasks, 'verify', config, model, tokenizer)
        decisions = [d for task in tasks for d in research.read_json(research.DATA / 'verify' / (task['id']+'.json'))['parsed']['decisions']]
        extracted = [f for unit in game_units for f in research.read_json(research.DATA / 'extract' / (unit['id']+'.json'))['parsed']['facts']]
        decisions += [{'id': f['id'], **f['mechanical_verdict']} for f in extracted if 'mechanical_verdict' in f]
        progress.append({'game_id': gid, 'units': len(game_units), 'claims': len(decisions),
                         'supported': sum(d['verdict'] == 'supported' for d in decisions)})
        research.write_json(research.REPORT / 'progress.json', {'completed_games': len(progress), 'total_games': len(games), 'games': progress})
        mx.clear_cache()
    research.merge_reports(units, research.verification_tasks(units, config))


def review_index():
    manifest = research.read_json(research.REPORT / 'research_manifest.json')
    lines = ['# Automated full-packet evidence research', '',
             'Every source page was processed. Retained evidence has exact source quotations and an audit of model-proposed summaries. The semantic checker proved unreliable on some factual challenges; the forecaster therefore uses actual quotations and code-checked table facts, never generated paraphrases or inferred scopes/caveats.', '',
             '[Quality audit](quality_audit.md) · [Source/page validation](validation.json) · '
             '[Forecaster result](../researched_forecaster/findings.md) · [Original full PDFs and extracted pages](../full_game_notes/index.md)', '',
             '| Date | Game | Split | Pages | Accepted claims | Rejected/unclear |', '| --- | --- | --- | ---: | ---: | ---: |']
    for item in manifest['reports']:
        path = ROOT / item['report_path']
        report = research.read_json(path)
        meta = report['game']
        relative = '../../' + str(path.with_suffix('.md').relative_to(ROOT))
        lines.append(f"| {meta['game_date_et']} | [{meta['away_abbr']} at {meta['home_abbr']}]({relative}) | {meta['split']} | {report['pdf_pages']} | {len(report['facts'])} | {len(report['rejected_claims'])} |")
        body = [f"# {meta['away_abbr']} at {meta['home_abbr']} — {meta['game_date_et']}", '',
                f"Publisher: {meta['publisher']}. All {report['pdf_pages']} pages accounted for. {len(report['facts'])} accepted claims; {len(report['rejected_claims'])} rejected/unclear claims.", '', report['limitations'], '',
                '## Checked evidence', '']
        for number, fact in enumerate(report['facts'], 1):
            body += [f"### Evidence item {number}", '', f"Method: {fact['origin']}.", '']
            if fact['origin'] == 'deterministic_table':
                body += [fact['claim'], '']
            else:
                body += ['Model-proposed summary (unverified; **not a forecast input**): ' + fact['claim'], '']
            if fact['caveat']:
                body += ['Qualification: ' + fact['caveat'], '']
            body += ['Support-check explanation: ' + fact['support_check']['reason'], '']
            for citation in fact['citations']:
                body += [f"PDF page {citation['page']}, line {citation['line']}:", '', '> ' + citation['text'], '']
        body += ['## Rejected and unclear claims', '']
        for fact in report['rejected_claims']:
            body.append(f"- **{fact['support_check']['verdict']}:** {fact['claim']} — {fact['support_check']['reason']}")
        body += ['', '## Complete page coverage', '']
        body.extend(f"- PDF pages {d['pages']}: {d['disposition']}" for d in report['unit_dispositions'])
        if report['empty_pages']:
            body.append('Pages without extracted text: ' + str(report['empty_pages']))
        path.with_suffix('.md').write_text('\n'.join(body)+'\n')
    (research.REPORT / 'index.md').write_text('\n'.join(lines)+'\n')


def pipeline():
    if Path(sys.executable).parent != ROOT / '.venv/bin':
        raise ValueError('Use the repository .venv for this experiment')
    started = time.time()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = ROOT / 'data/processed/research_experiment/logs' / run_id
    logs.mkdir(parents=True, exist_ok=True)
    steps = [
        ('research', 'run_research_experiment.py', '--stage', 'research'),
        ('validate_research', 'validate_researched_forecaster.py', '--stage', 'research'),
        ('prepare_forecaster', 'run_researched_forecaster.py', '--stage', 'prepare'),
        ('encode_reports', 'run_researched_forecaster.py', '--stage', 'embed'),
        ('train_head', 'run_researched_forecaster.py', '--stage', 'train'),
        ('evaluate', 'run_researched_forecaster.py', '--stage', 'evaluate'),
        ('validate_forecaster', 'validate_researched_forecaster.py', '--stage', 'forecaster'),
    ]
    completed = []
    for name, script, *args in steps:
        log = logs / (name + '.log')
        status = {'run_id': run_id, 'status': 'running', 'stage': name, 'completed_stages': completed,
                  'log': str(log.relative_to(ROOT)), 'elapsed_seconds': round(time.time()-started)}
        research.write_json(STATUS, status)
        print(json.dumps(status), flush=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            research.write_json(STATUS, {**status, 'status': 'failed', 'returncode': result.returncode})
            raise RuntimeError(f'{name} failed; inspect {log}. Completed research responses are preserved for resumption.')
        completed.append(name)
    review_index()
    research.write_json(STATUS, {'run_id': run_id, 'status': 'complete', 'completed_stages': completed + ['review_index'],
                                'elapsed_seconds': round(time.time()-started), 'metrics': 'reports/researched_forecaster/metrics.json'})
    print(json.dumps(research.read_json(STATUS), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'research', 'index'), default='all')
    args = parser.parse_args()
    {'all': pipeline, 'research': research_by_game, 'index': review_index}[args.stage]()
