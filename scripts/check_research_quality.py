"""Training-note factual challenge: detect known date/table errors before the bulk run."""
import json

import run_game_note_research as research

GID = 'espn:401704882'  # Shortest training packet, manually reviewed before outcomes.


def challenge_cases(units):
    by_page = {u['lines'][0]['page']: u for u in units}
    specs = [
        (10, 'Bam Adebayo has played in every game this season, averaging over 37 minutes per game.', 'season_to_date', False),
        (12, "Haywood Highsmith's most recent game was against Dallas on 11/24/24, where he played 18:38 minutes.", 'prior_game', False),
        (14, "Josh Richardson is currently not playing due to coach's decision (DNP-CD).", 'current', False),
        (15, 'Dru Smith is ruled out for this upcoming game because of a right ACL injury on November 22, 2023.', 'current', False),
        (10, 'Bam Adebayo played 38 minutes and 36 seconds at Toronto on December 1, 2024.', 'prior_game', True),
        (11, 'Tyler Herro played 36 minutes and 49 seconds at Toronto on December 1, 2024.', 'prior_game', True),
        (11, 'Jimmy Butler played 34 minutes and 25 seconds at Toronto on December 1, 2024.', 'prior_game', True),
        (4, 'Miami is playing the second game of a road back-to-back after playing in Toronto the previous night.', 'current', True),
        (4, 'Tyler Herro has made 57 consecutive three-point shots.', 'current', False),
        (7, 'Jimmy Butler currently leads Miami in scoring.', 'current', False),
        (1, 'Duncan Robinson has made a three-point field goal in 52 consecutive regular-season games.', 'current', False),
        (10, 'Bam Adebayo scored 5 points at Toronto on December 1, 2024.', 'prior_game', False),
    ]
    tasks, expected = [], {}
    for i, (page, claim, scope, supported) in enumerate(specs):
        unit = by_page[page]
        fid = f'quality-f{i:02}'
        fact = {'id': fid, 'claim': claim, 'time_scope': scope, 'caveat': '', 'citations': unit['lines']}
        tasks.append({'id': f'quality-v{i:02}', 'game': unit['game'], 'lines': unit['lines'], 'facts': [fact]})
        expected[fid] = supported
    return tasks, expected


def main():
    config = research.read_json(research.CONFIG)
    research.verify_protocol(config, check_model=True)
    units = [u for u in research.read_rows(research.DATA / 'units.jsonl') if u['game']['game_id'] == GID]
    if len(units) != 15 or any(u['game']['split'] != 'train' for u in units):
        raise ValueError('Quality gate must use the complete 15-page training packet')
    mx, model, tokenizer = research.load_runtime(config)
    research.execute_tasks(units, 'extract', config, model, tokenizer)
    tasks = research.verification_tasks(units, config)
    research.execute_tasks(tasks, 'verify', config, model, tokenizer)
    challenge, expected = challenge_cases(units)
    # Deliberately false challenge statements remain separate from research data.
    research.DATA = research.ROOT / 'data/probes/research_quality_gate'
    research.write_json(research.DATA / 'cases.json', {'tasks': challenge, 'expected_supported': expected})
    research.execute_tasks(challenge, 'verify', config, model, tokenizer)
    results = []
    for task in challenge:
        response = research.read_json(research.DATA / 'verify' / (task['id'] + '.json'))
        decision = research.validate_verification(research.parse_json(response['text']), task)['decisions'][0]
        wanted = expected[decision['id']]
        results.append({'claim': task['facts'][0]['claim'], 'expected_supported': wanted,
                        **decision, 'passed': (decision['verdict'] == 'supported') == wanted})
    from run_researched_forecaster import prompt_messages, evidence_with_context
    from game_note_tables import table_facts
    canonical = [f for u in units for f in table_facts(u)]
    table_text = '\n'.join(f['claim'] for f in canonical)
    numerical_guard = all(s in table_text for s in ('38:36 minutes, 13 points, 20 rebounds',
                          '36:49 minutes', '34:25 minutes', '35.65 minutes and 14.00 points'))
    coverage = next(c for c in research.read_json(research.ROOT / 'data/processed/game_note_research/coverage.json') if c['game']['game_id'] == GID)
    packet = research.read_json(research.ROOT / coverage['packet_path'])
    source_line = next(line for unit in units for line in unit['lines'] if 'BACK-TO-BACK: Tonight marks' in line['text'])
    adversarial = {'id': 'guard-only', 'origin': 'language_model', 'claim': 'UNVERIFIED_INJURY_PLAYER_OUT',
                   'category': 'availability', 'time_scope': 'UNVERIFIED_SCOPE', 'caveat': 'UNVERIFIED_CAVEAT',
                   'citations': [source_line]}
    expanded = evidence_with_context([adversarial] + canonical, packet)
    prompt = str(prompt_messages({**units[0]['game'], 'p_boost': .5}, expanded))
    paraphrase_guard = all(s not in prompt for s in ('UNVERIFIED_INJURY_PLAYER_OUT', 'UNVERIFIED_SCOPE', 'UNVERIFIED_CAVEAT'))
    quote_guard = source_line['text'] in prompt and all(c['text'] in prompt for f in expanded if f['origin'] == 'language_model' for c in f['forecast_citations'])
    guards = {'canonical_table_values_and_arithmetic': numerical_guard,
              'generated_paraphrases_scopes_and_caveats_excluded_from_forecast': paraphrase_guard,
              'exact_source_quotations_and_context_preserved': quote_guard}
    report = {'game_id': GID, 'split': 'train', 'protocol_sha256': research.sha(research.REPORT / 'protocol.json'),
              'quality_runner_sha256': research.sha(__file__), 'target_game_outcomes_used': False,
              'cases': results, 'semantic_checker_all_passed': all(r['passed'] for r in results),
              'training_readiness_guards': guards, 'ready_for_source_evidence_run': all(guards.values()),
              'scope': 'Raw semantic-check failures remain visible. Readiness applies only to a forecaster using source quotations and code-checked numbers, never LM paraphrases. This is not a corpus-wide factual-accuracy estimate.'}
    research.write_json(research.REPORT / 'quality_gate.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['ready_for_source_evidence_run']:
        raise ValueError('Source-evidence readiness guard failed; do not start the bulk run')


if __name__ == '__main__':
    main()
