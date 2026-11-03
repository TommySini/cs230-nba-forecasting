"""Audit complete research provenance and replay the selected probability head."""
import argparse
from collections import Counter
import csv
import json

import numpy as np
from scipy.special import expit

from run_game_note_research import (ROOT, CONFIG, DATA as RD, REPORT as RR, read_json, read_rows,
                                    sha, value_hash, write_json, verify_protocol, extraction_messages,
                                    verification_messages, verification_tasks, parse_json,
                                    validate_extraction, validate_verification, page_lines, response_protocols)
from run_researched_forecaster import (DATA, REPORT, checked_reports, load_features, verify_selection,
                                       choose_candidate, fit_head, cavaliers_table_facts,
                                       evidence_with_context, prompt_messages, chunk_report)
from run_boosting_notes import load_prepared, log_odds
from run_game_note_research import encode_messages
from run_gradient_boosting import score_predictions
from run_text_ablation import paired_brier_interval


def research(config):
    from prepare_full_game_notes import validate as validate_full
    validate_full()
    research_protocol = verify_protocol(config, check_model=True)
    from import_modal_research import validate_execution
    registry_path = RD / 'remote_import.json'
    registry = read_json(registry_path) if registry_path.exists() else None
    if registry and sha(RR/'cloud_execution_plan.json') != registry['plan_sha256']:
        raise ValueError('Cloud execution plan changed after import')
    units = read_rows(RD / 'units.jsonl')
    by_id = {u['id']: u for u in units}
    if len(by_id) != len(units):
        raise ValueError('Duplicated research unit IDs')
    total_pages = 0
    for coverage in read_json(RD / 'coverage.json'):
        packet = read_json(ROOT / coverage['packet_path'])
        if sha(ROOT / coverage['packet_path']) != coverage['packet_sha256']:
            raise ValueError('Packet changed')
        expected = [line for p in packet['pages'] for line in page_lines(p)]
        actual = [line for uid in coverage['units'] for line in by_id[uid]['lines']]
        if actual != expected or value_hash(actual) != coverage['normalized_text_sha256']:
            raise ValueError('Full source-line coverage failed')
        total_pages += coverage['pages']
    facts = {}
    for unit in units:
        path = RD / 'extract' / (unit['id'] + '.json')
        response = read_json(path)
        relative = str(path.relative_to(ROOT))
        if response.get('execution_sha256') or registry and relative in registry['response_files']:
            validate_execution(response, relative, registry, research_protocol)
        if response['status'] != 'complete' or response['finish_reason'] != 'stop':
            raise ValueError('Incomplete research generation')
        if response['request_sha256'] != value_hash(extraction_messages(unit)) or response['protocol_sha256'] not in response_protocols():
            raise ValueError('Extraction response provenance failed')
        parsed = validate_extraction(parse_json(response['text']), unit)
        if parsed != response['parsed']:
            raise ValueError('Stored parsed extraction differs from raw generation')
        facts.update({f['id']: f for f in parsed['facts']})
    decisions = {fid: {'id': fid, **f['mechanical_verdict']} for fid, f in facts.items() if 'mechanical_verdict' in f}
    tasks = verification_tasks(units, config)
    for task in tasks:
        path = RD / 'verify' / (task['id'] + '.json')
        response = read_json(path)
        relative = str(path.relative_to(ROOT))
        if response.get('execution_sha256') or registry and relative in registry['response_files']:
            validate_execution(response, relative, registry, research_protocol)
        if response['status'] != 'complete' or response['finish_reason'] != 'stop':
            raise ValueError('Incomplete support-check generation')
        if response['request_sha256'] != value_hash(verification_messages(task)) or response['protocol_sha256'] not in response_protocols():
            raise ValueError('Support-check response provenance failed')
        parsed = validate_verification(parse_json(response['text']), task)
        if parsed != response['parsed']:
            raise ValueError('Stored parsed support check differs from raw generation')
        decisions.update({d['id']: d for d in parsed['decisions']})
    if set(decisions) != set(facts):
        raise ValueError('Not every extracted claim received a support check')
    _, reports = checked_reports(config)
    retained, rejected = set(), set()
    for report in reports.values():
        for fact in report['facts']:
            ids = [fact['id']] + fact['duplicate_ids']
            for fid in ids:
                if decisions[fid]['verdict'] != 'supported' or any(' '.join(facts[fid][k].lower().split()) != ' '.join(fact[k].lower().split()) for k in ('claim', 'category', 'time_scope', 'caveat')):
                    raise ValueError('Unsupported or mismatched claim entered report')
            for key in ('claim', 'category', 'time_scope', 'caveat', 'unit_id'):
                if fact[key] != facts[fact['id']][key]:
                    raise ValueError('Report claim changed after checking')
            expected_citations = {c['id']: c for fid in ids for c in facts[fid]['citations']}
            if {c['id']: c for c in fact['citations']} != expected_citations:
                raise ValueError('Report citations differ from source')
            retained.update(ids)
        rejected.update(f['id'] for f in report['rejected_claims'])
    if retained != {fid for fid, d in decisions.items() if d['verdict'] == 'supported'}:
        raise ValueError('Some supported facts omitted from research report')
    if rejected != {fid for fid, d in decisions.items() if d['verdict'] != 'supported'}:
        raise ValueError('Rejected claim audit incomplete')
    result = {'games': len(reports), 'pages': total_pages, 'research_units': len(units),
              'support_check_requests': len(tasks), 'raw_claims': len(facts),
              'verdicts': dict(Counter(d['verdict'] for d in decisions.values())),
              'accepted_categories': dict(Counter(facts[fid]['category'] for fid in retained)),
              'accepted_origins': dict(Counter(facts[fid]['origin'] for fid in retained)),
              'source_and_response_provenance': 'passed', 'complete_page_and_line_coverage': 'passed',
              'all_retained_claims_checked_and_cited': 'passed',
              'semantic_limit': 'Same-model support checking can retain factual errors; selected training reports were manually audited, not all claims.'}
    write_json(RR / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def forecaster(config):
    from transformers import AutoTokenizer
    protocol, rows, features = load_features(config)
    selection = verify_selection()
    if choose_candidate(read_json(REPORT / 'candidates.json'))['name'] != selection['selected_candidate']:
        raise ValueError('Selection does not minimize dev Brier')
    _, old_rows, labels = load_prepared(read_json(ROOT / 'config/boosting_notes.json'))
    label_map = {r['game_id']: r['home_win'] for r in labels}
    train = np.array([r['split'] == 'train' for r in rows])
    if selection['fit_game_ids'] != [r['game_id'] for r in rows if r['split'] == 'train']:
        raise ValueError('Head fit membership changed')
    if selection['selection_game_ids'] != [r['game_id'] for r in rows if r['split'] == 'dev']:
        raise ValueError('Dev selection membership changed')
    if selection['test_labels_used_for_selection']:
        raise ValueError('Test labels were used for selection')
    with np.load(DATA / 'head.npz', allow_pickle=False) as saved:
        center, weights, bias = saved['center'], saved['weights'], saved['intercept']
    np.testing.assert_allclose(center, features[train].mean(axis=0), atol=1e-14, rtol=0)
    if selection['selected_candidate'] == 'zero_correction':
        np.testing.assert_array_equal(np.r_[weights, bias], np.zeros(features.shape[1] + 1))
    else:
        replay, _ = fit_head(features[train] - center,
                             np.array([label_map[r['game_id']] for r in rows if r['split'] == 'train']),
                             log_odds([r['p_boost'] for r in rows if r['split'] == 'train']), selection['penalty'])
        np.testing.assert_allclose(replay, np.r_[weights, bias], atol=1e-8, rtol=0)
    all_prob = expit(log_odds([r['p_boost'] for r in rows]) + (features - center) @ weights + bias)
    prediction_map = {r['game_id']: float(p) for r, p in zip(rows, all_prob)}
    row_map = {r['game_id']: r for r in rows}
    for split in ('dev', 'test'):
        with (DATA / f'{split}_predictions.csv').open() as stream:
            predictions = list(csv.DictReader(stream))
        if {p['game_id'] for p in predictions} != {r['game_id'] for r in rows if r['split'] == split}:
            raise ValueError('Prediction membership changed')
        np.testing.assert_allclose([float(p['researched_hybrid']) for p in predictions],
                                   [prediction_map[p['game_id']] for p in predictions], atol=1e-12, rtol=0)
        if any(int(p['home_win']) != label_map[p['game_id']] for p in predictions):
            raise ValueError('Prediction labels differ from the separate original label file')
        np.testing.assert_allclose([float(p['original_boosting']) for p in predictions],
                                   [row_map[p['game_id']]['p_boost'] for p in predictions], atol=1e-12, rtol=0)
    metrics = read_json(REPORT / 'metrics.json')
    if metrics['test_predictions_sha256'] != sha(DATA / 'test_predictions.csv') or metrics['selection_sha256'] != sha(REPORT / 'selection.json'):
        raise ValueError('Evaluation artifacts changed')
    test = [r for r in rows if r['split'] == 'test']
    y_test = np.array([label_map[r['game_id']] for r in test])
    baseline = np.array([r['p_boost'] for r in test])
    hybrid = np.array([prediction_map[r['game_id']] for r in test])
    replay_metrics = {'original_boosting': score_predictions(y_test, baseline),
                      'researched_hybrid': score_predictions(y_test, hybrid)}
    for name, scores in replay_metrics.items():
        if set(scores) != set(metrics['test'][name]):
            raise ValueError('Saved evaluation metric fields differ from replay')
        for key, value in scores.items():
            saved = metrics['test'][name][key]
            if value is None:
                if saved is not None:
                    raise ValueError('Saved undefined metric differs from replay')
            else:
                np.testing.assert_allclose(value, saved, atol=1e-12, rtol=0)
    paired = paired_brier_interval(y_test, baseline, hybrid, [r['game_date_et'] for r in test], config['forecaster'])
    paired['hybrid_minus_boosting'] = paired.pop('text_minus_stats')
    for key in ('hybrid_minus_boosting', 'paired_date_bootstrap_ci95'):
        np.testing.assert_allclose(paired[key], metrics['paired_comparison'][key], atol=1e-12, rtol=0)
    for key in ('test_dates', 'replicates', 'conditional_on'):
        if paired[key] != metrics['paired_comparison'][key]:
            raise ValueError('Paired Brier comparison metadata differs from replay')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model_path'], local_files_only=True)
    coverage_by_id = {c['game']['game_id']: c for c in read_json(RD / 'coverage.json')}
    for row in rows:
        report = read_json(RD / 'reports' / (row['game_id'].replace(':', '-') + '.json'))
        coverage = coverage_by_id[row['game_id']]
        if sha(ROOT / coverage['packet_path']) != coverage['packet_sha256']:
            raise ValueError('Packet changed before table replay')
        packet = read_json(ROOT / coverage['packet_path'])
        additional, _ = cavaliers_table_facts(packet)
        if additional != row['additional_table_facts']:
            raise ValueError('Code-decoded table evidence differs from source replay')
        if [fid for chunk in row['chunks'] for fid in chunk['fact_ids']] != [f['id'] for f in report['facts'] + additional]:
            raise ValueError('Embedding input did not include every supported fact')
        evidence = evidence_with_context(report['facts'] + additional, packet)
        chunks = chunk_report(row, evidence, tokenizer, config['embedding_token_budget'])
        expected_chunks = [{'fact_ids': [f['id'] for f in chunk], 'messages': prompt_messages(row, chunk),
                            'tokens': len(encode_messages(tokenizer, prompt_messages(row, chunk)))} for chunk in chunks]
        if row['chunks'] != expected_chunks:
            raise ValueError('Actual forecast prompts differ from source-evidence replay')
    result = {'games': len(rows), 'splits': dict(Counter(r['split'] for r in rows)),
              'source_and_model_locks': 'passed', 'all_supported_facts_encoded': 'passed',
              'train_only_center_and_head_fit_replay': 'passed', 'dev_only_selection': 'passed',
              'dev_and_test_probability_replay': 'passed', 'metrics_and_paired_interval_replay': 'passed',
              'source_to_forecast_prompt_replay': 'passed', 'old_baseline_and_pilot_locks': 'passed'}
    write_json(REPORT / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('research', 'forecaster', 'all'), default='all')
    args = parser.parse_args()
    config = read_json(CONFIG)
    if args.stage in ('research', 'all'):
        research(config)
    if args.stage in ('forecaster', 'all'):
        forecaster(config)
