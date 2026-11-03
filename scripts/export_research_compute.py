"""Snapshot only the pregame research workload for a private compute handover."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile

import run_game_note_research as research


def main():
    root = research.ROOT
    config = research.read_json(research.CONFIG)
    protocol = research.verify_protocol(config)
    gate = research.read_json(research.REPORT / 'quality_gate.json')
    if not gate['ready_for_source_evidence_run'] or gate['protocol_sha256'] != research.sha(research.REPORT / 'protocol.json'):
        raise ValueError('Stale or failed source-evidence readiness check')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = root / 'data/probes/cloud_research_transfer' / run_id
    destination.mkdir(parents=True, exist_ok=False)
    fixed = ['scripts/run_game_note_research.py', 'scripts/run_research_experiment.py',
             'scripts/game_note_tables.py', 'scripts/check_research_quality.py',
             'config/game_note_research.json', 'requirements-research-cuda.txt',
             'reports/game_note_research/protocol.json', 'reports/game_note_research/quality_gate.json',
             'data/processed/full_game_notes/manifest.json',
             'data/processed/game_note_research/units.jsonl', 'data/processed/game_note_research/coverage.json']
    paths = {root / p for p in fixed}
    paths.update(root / p for p in protocol.get('compatible_response_protocols', {}).values())
    for relative in ('extract', 'verify', 'attempts'):
        paths.update((research.DATA / relative).rglob('*.json'))
    files = {}
    for source in sorted(paths):
        relative = str(source.relative_to(root))
        # Response files are atomically replaced by the active local runner.
        # Snapshot each complete file's bytes and hash that copy, not a later read.
        content = source.read_bytes()
        if source.suffix == '.json':
            json.loads(content)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[relative] = research.sha(target)
    for unit in research.read_rows(destination / 'data/processed/game_note_research/units.jsonl'):
        forbidden = {'home_win', 'away_win', 'labels', 'outcome', 'p_boost', 'features'}
        if forbidden & (set(unit) | set(unit['game'])):
            raise ValueError('Forecast labels/statistics must not enter research-only transfer')
        if unit['request_sha256'] != research.value_hash(research.extraction_messages(unit)):
            raise ValueError('Transfer prompt differs from frozen source request')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
                'research_protocol_sha256': research.sha(research.REPORT / 'protocol.json'),
                'model_id': config['model_id'], 'model_revision': config['model_revision'],
                'model_files': protocol['model_sha256'], 'runtime_versions': protocol['versions'],
                'prepared_mlx_command': '.venv/bin/python scripts/run_research_experiment.py --stage research',
                'allowed_compute_scope': 'Pregame research only. An alternative inference backend requires its separately reviewed execution manifest; no head fitting or target labels.',
                'scope': 'Private research-only snapshot. No target labels, booster inputs, raw ESPN, account credentials or model weight files. No cloud upload/provisioning performed. Cached response provenance remains unchanged. Validate all returned reports against local original sources before forecasting.'}
    research.write_json(destination / 'transfer_manifest.json', manifest)
    archive = destination.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        for relative in sorted(files):
            stream.add(destination / relative, arcname='nba_research/' + relative, recursive=False)
        stream.add(destination / 'transfer_manifest.json', arcname='nba_research/transfer_manifest.json', recursive=False)
    print(json.dumps({'snapshot': str(destination), 'archive': str(archive), 'files': len(files) + 1,
                      'archive_bytes': archive.stat().st_size, 'sha256': research.sha(archive),
                      'uploaded': False, 'cloud_tested': False}, indent=2))


if __name__ == '__main__':
    main()
