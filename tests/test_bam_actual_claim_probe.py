import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bam_actual_claim_probe as probe
from tests.test_bam_research import game, finding, review_decision


class ActualClaimProbeTests(unittest.TestCase):
    def setUp(self):
        self.game = game()
        self.finding = finding()
        self.artifact = {'artifact_hash': 'original-hash', 'request_id': 'original-request',
                         'stage': 'module_partial', 'model': 'original-student'}
        self.request = probe.isolated_request(self.game, self.artifact, self.finding)

    def test_whole_source_page_and_original_candidate_preserved_without_labels(self):
        payload = json.loads(self.request['messages'][1]['content'])
        self.assertEqual(payload['candidate'], self.finding)
        self.assertEqual(payload['pregame_source_context'][0]['pages'][0]['lines'], self.game['sources'][0]['pages'][0]['lines'])
        serialized = json.dumps(self.request)
        self.assertNotIn('home_win', serialized)
        self.assertNotIn('NEVER PASS THIS', serialized)
        self.assertNotIn('expected_verdict', serialized)
        self.assertNotIn('Assess the entire report too', serialized)
        self.assertIn('Do not make a whole-report coverage judgment', serialized)
        self.assertEqual(self.request['original_finding_id'], self.finding['id'])

    def test_heldout_and_unknown_pages_rejected(self):
        for split in ('dev', 'test'):
            with self.assertRaisesRegex(ValueError, 'training-only'):
                probe.isolated_request(game(split), self.artifact, self.finding)
        with self.assertRaisesRegex(ValueError, 'page missing'):
            probe.isolated_request(self.game, self.artifact, self.finding, extra_pages=[('s1', 99)])

    def test_valid_source_reference_materialized(self):
        decision = probe.check_decision(self.request, review_decision(self.finding))
        self.assertEqual(decision['verdict'], 'supported')
        self.assertEqual(decision['canonical_evidence'][0]['text'], self.game['sources'][0]['pages'][0]['lines'][0]['text'])

    def test_foreign_candidate_source_and_line_rejected(self):
        decision = review_decision(self.finding)
        bad = copy.deepcopy(decision); bad['id'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'Foreign'): probe.check_decision(self.request, bad)
        for field, value in [('source_id', 'foreign'), ('page', 2), ('lines', [99])]:
            bad = copy.deepcopy(decision); bad['required_citations'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'absent'): probe.check_decision(self.request, bad)

    def test_missing_original_citation_withholds_support_without_promoting_rejection(self):
        decision = review_decision(self.finding); decision['required_citations'][0]['lines'] = [1, 2]
        result = probe.check_decision(self.request, decision)
        self.assertEqual(result['model_verdict'], 'supported')
        self.assertEqual(result['verdict'], 'unclear')
        self.assertEqual(result['missing_original_citations'], [('s1', 1, 2)])
        decision['verdict'] = 'unsupported'
        self.assertEqual(probe.check_decision(self.request, decision)['verdict'], 'unsupported')

    def test_incomplete_schema_does_not_receive_default_support(self):
        for key in probe.SCHEMA:
            decision = review_decision(self.finding); del decision[key]
            with self.assertRaises(ValueError): probe.check_decision(self.request, decision)

    def test_frozen_artifacts_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixed.json'
            probe.immutable(path, b'original'); probe.immutable(path, b'original')
            with self.assertRaisesRegex(ValueError, 'Frozen'): probe.immutable(path, b'changed')


if __name__ == '__main__':
    unittest.main()
