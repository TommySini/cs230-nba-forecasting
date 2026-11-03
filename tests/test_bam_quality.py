"""Fixed training-only benchmark labels and strict scoring boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bam_quality as quality
import bam_research as bam


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.game = {'game_id': 'g1', 'game_date_et': '2025-01-02', 'cutoff_utc': '2025-01-02T23:00:00Z', 'home_abbr': 'BOS', 'away_abbr': 'MIA', 'split': 'train', 'sources': [{'source_id': 's1', 'publisher': 'MIA', 'url': 'https://example.com/notes.pdf', 'sha256': 'a' * 64, 'archive_time': '2025-01-02T20:00:00Z', 'pages': [{'page': 1, 'lines': [{'line': 1, 'text': 'Brown is questionable tonight.'}, {'line': 2, 'text': '2024-25 current season.'}]}]}]}
        self.citations = [{'source_id': 's1', 'page': 1, 'lines': [1, 2]}]
        self.requests, labels = [], []
        for index, supported in enumerate((True, False)):
            request, expected = quality.case_request(self.game, f'case-{index}', 'Brown is questionable tonight.' if supported else 'Brown is confirmed out.', 'current', self.citations, expected_supported=supported, expected_verdict='supported' if supported else 'unsupported', category='fixture', adjudication='Local source adjudication.')
            self.requests.append(request); labels.append(expected)
        self.expected = {'benchmark_hash': bam.value_hash(self.requests), 'cases': labels}

    def response(self, index, verdict):
        return {'request_id': self.requests[index]['request_id'], 'model': 'teacher', 'text': json.dumps({'case_id': f'case-{index}', 'verdict': verdict, 'reason': 'Explicit source status.', 'citations': self.citations}), 'elapsed_seconds': 2}

    def test_labels_never_enter_model_messages(self):
        for request in self.requests:
            serialized = json.dumps(request)
            for forbidden in ('expected_supported', 'expected_verdict', 'adjudication', 'Local source adjudication'):
                self.assertNotIn(forbidden, serialized)

    def test_test_game_and_late_source_are_rejected(self):
        for key, value in [('split', 'test'), ('split', 'dev')]:
            game = copy.deepcopy(self.game); game[key] = value
            with self.assertRaisesRegex(ValueError, 'TRAIN ONLY'):
                quality.case_request(game, 'bad', 'claim', 'current', self.citations, expected_supported=True, category='x', adjudication='x')
        game = copy.deepcopy(self.game)
        game['sources'][0]['archive_time'] = '2025-01-03T00:00:00Z'
        with self.assertRaises(ValueError):
            quality.case_request(game, 'bad', 'claim', 'current', self.citations, expected_supported=True, category='x', adjudication='x')

    def test_correct_decisions_and_runtime(self):
        result = quality.score(self.requests, self.expected, [self.response(0, 'supported'), self.response(1, 'unsupported')], 'teacher')
        self.assertEqual(result['binary_support_accuracy'], 1)
        self.assertEqual(result['exact_three_way_accuracy'], 1)
        self.assertEqual(result['measured_response_seconds'], 4)

    def test_false_accept_and_false_exclusion_are_distinct(self):
        result = quality.score(self.requests, self.expected, [self.response(0, 'unclear'), self.response(1, 'supported')], 'teacher')
        self.assertEqual(result['false_accept_count'], 1)
        self.assertEqual(result['false_exclusion_count'], 1)
        self.assertEqual(result['binary_support_accuracy'], 0)

    def test_missing_malformed_or_duplicate_responses_never_count_as_correct_rejections(self):
        for rows in ([], [self.response(1, 'unsupported')] * 2, [{'request_id': self.requests[1]['request_id'], 'text': 'not JSON'}]):
            result = quality.score(self.requests, self.expected, rows, 'teacher')
            self.assertEqual(result['binary_support_accuracy'], 0)
            self.assertEqual(result['missing_or_invalid_responses'], 2)

    def test_source_citation_case_identity_and_finish_reason_are_checked(self):
        response = self.response(0, 'supported')
        value = json.loads(response['text']); value['citations'][0]['lines'] = [999]
        response['text'] = json.dumps(value)
        with self.assertRaises(ValueError):
            quality.parse_response(self.requests[0], response)
        response = self.response(0, 'supported'); response['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            quality.parse_response(self.requests[0], response)
        response = self.response(1, 'supported')
        with self.assertRaises(ValueError):
            quality.parse_response(self.requests[0], response)

    def test_changed_benchmark_and_mixed_models_are_rejected(self):
        changed = copy.deepcopy(self.requests); changed[0]['max_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'frozen'):
            quality.score(changed, self.expected, [], 'teacher')
        response = self.response(0, 'supported'); response['model'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Mixed model'):
            quality.score(self.requests, self.expected, [response], 'teacher')

    def test_model_comparison_requires_identical_benchmark(self):
        first = quality.score(self.requests, self.expected, [], 'first')
        second = quality.score(self.requests, self.expected, [], 'second')
        result = quality.compare_scores([first, second])
        self.assertFalse(result['all_responses_valid'])
        second['benchmark_hash'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identical'):
            quality.compare_scores([first, second])

    def source_first_response(self):
        response = self.response(0, 'supported')
        output = json.loads(response['text'])
        output.update(source_evidence=[{'source_id': 's1', 'page': 1, 'lines': [1], 'quotation': 'Brown is questionable tonight.', 'source_reading': 'Brown is questionable for tonight.'}], attribute_checks=[{'attribute': attribute, 'claim_value': 'Questionable tonight or no assertion', 'source_value': 'Questionable tonight or not applicable', 'assessment': 'match'} for attribute in quality.ATTRIBUTES], contradictions=[])
        response['text'] = json.dumps(output)
        return response

    def test_source_first_variant_preserves_cases_context_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'v1'; output = Path(directory) / 'v2'
            quality.write_rows(source / 'requests.jsonl', self.requests)
            quality.write_json(source / 'expected.json', self.expected)
            quality.prepare_source_first(source, output, Path(directory) / 'reports')
            requests = quality.read_rows(output / 'requests.jsonl')
            expected = json.loads((output / 'expected.json').read_text())
            for old, new in zip(self.requests, requests):
                before = json.loads(old['messages'][1]['content']); after = json.loads(new['messages'][1]['content'])
                self.assertEqual(before['candidate'], after['candidate'])
                self.assertEqual(before['source_context'], after['source_context'])
                self.assertNotEqual(old['request_id'], new['request_id'])
                self.assertEqual(list(after['output_schema'])[-1], 'verdict')
            self.assertEqual([r['expected_supported'] for r in self.expected['cases']], [r['expected_supported'] for r in expected['cases']])
            self.assertEqual(quality.read_rows(source / 'requests.jsonl'), self.requests)

    def test_source_first_requires_complete_grounded_matrix(self):
        request = self.requests[0] | {'benchmark_variant': 'source_first_v2'}
        response = self.source_first_response()
        self.assertEqual(quality.parse_response(request, response)['verdict'], 'supported')
        value = json.loads(response['text']); value['attribute_checks'].pop()
        response['text'] = json.dumps(value)
        with self.assertRaisesRegex(ValueError, 'attribute matrix'):
            quality.parse_response(request, response)

    def test_source_first_rejects_self_contradiction_and_fabricated_quote(self):
        request = self.requests[0] | {'benchmark_variant': 'source_first_v2'}
        for corruption in ('contradiction', 'quote'):
            response = self.source_first_response(); value = json.loads(response['text'])
            if corruption == 'contradiction':
                value['attribute_checks'][0]['assessment'] = 'contradiction'
            else:
                value['source_evidence'][0]['quotation'] = 'Invented words.'
            response['text'] = json.dumps(value)
            with self.assertRaises(ValueError): quality.parse_response(request, response)

    def test_shared_batch_runtime_is_counted_once(self):
        responses = [self.response(0, 'supported'), self.response(1, 'unsupported')]
        for row in responses:
            row.update(batch_seconds=7.5, batch_size=2, execution_sha256='one-batch')
        result = quality.score(self.requests, self.expected, responses, 'teacher')
        self.assertEqual(result['measured_batch_seconds'], 7.5)
        self.assertEqual(result['complete_timed_batches'], 1)
        result = quality.score(self.requests, self.expected, responses[:1], 'teacher')
        self.assertIsNone(result['measured_batch_seconds'])

    def test_frozen_benchmark_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'requests.jsonl'
            quality.write_frozen_rows(path, self.requests)
            quality.write_frozen_rows(path, self.requests)
            changed = copy.deepcopy(self.requests); changed[0]['max_tokens'] += 1
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                quality.write_frozen_rows(path, changed)
            self.assertEqual(quality.read_rows(path), self.requests)

    def test_v3_copies_canonical_evidence_without_requiring_model_quotation(self):
        request = self.requests[0] | {'benchmark_variant': 'source_refs_v3'}
        response = self.source_first_response(); value = json.loads(response['text'])
        value['source_evidence'][0].pop('quotation')
        response['text'] = json.dumps(value)
        output = quality.parse_response(request, response)
        self.assertEqual(output['source_evidence'][0]['quotations'][0]['text'], 'Brown is questionable tonight.')
        value['source_evidence'][0]['lines'] = [999]; response['text'] = json.dumps(value)
        with self.assertRaises(ValueError): quality.parse_response(request, response)


if __name__ == '__main__':
    unittest.main()
