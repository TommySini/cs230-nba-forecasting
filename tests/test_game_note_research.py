"""Evidence provenance and research leakage boundary checks."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_game_note_research import (page_lines, metadata, extraction_messages, validate_extraction,
                                    validate_verification, verification_messages, encode_messages, parse_json)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.unit = {'id': 'sample-u000', 'game': {'game_date_et': '2024-12-02', 'home_abbr': 'BOS',
                     'away_abbr': 'MIA', 'publisher': 'MIA', 'home_win': 0, 'p_boost': .7654321},
                     'lines': page_lines({'page': 10, 'text': 'BAM ADEBAYO RECENT GAMES\nDATE OPP MIN\n12/01/24 at TOR 38:36\n'})}
        self.raw = {'facts': [{'claim': 'Bam played 38:36 at Toronto on December 1.',
                    'category': 'workload', 'time_scope': 'prior_game', 'lines': [1, 2, 3],
                    'caveat': 'Does not confirm availability for this game.'}], 'disposition': 'Player workloads reviewed.'}

    def test_citations_are_copied_from_actual_page_lines(self):
        fact = validate_extraction(self.raw, self.unit)['facts'][0]
        self.assertEqual(fact['citations'], self.unit['lines'])
        self.assertEqual(fact['citations'][-1]['id'], 'P010L0003')
        for wrong in ([999], ['3'], [], [True]):
            raw = copy.deepcopy(self.raw)
            raw['facts'][0]['lines'] = wrong
            rejected = validate_extraction(raw, self.unit)['facts'][0]
            self.assertEqual(rejected['citations'], [])
            self.assertEqual(rejected['mechanical_verdict']['verdict'], 'unclear')
        rejected = validate_extraction({'facts': [None], 'disposition': 'Malformed model output'}, self.unit)['facts'][0]
        self.assertEqual(rejected['mechanical_verdict']['verdict'], 'unclear')
        self.assertIsNone(rejected['raw_rejected_fact'])

    def test_invalid_metadata_stays_in_audit_without_invalidating_source_quote(self):
        raw = copy.deepcopy(self.raw)
        raw['facts'][0]['time_scope'] = '2024-25'
        raw['facts'][0]['category'] = ['unexpected']
        result = validate_extraction(raw, self.unit)['facts'][0]
        self.assertEqual(result['time_scope'], 'unspecified')
        self.assertEqual(result['category'], 'other')
        self.assertEqual(result['citations'], self.unit['lines'])
        self.assertEqual(result['model_metadata_before_normalization']['time_scope'], '2024-25')

    def test_decision_array_normalization_preserves_decision_contents(self):
        result = parse_json('[{"id":"0","verdict":"unsupported","reason":"Wrong column"}]')
        self.assertEqual(result, {'decisions': [{'id': '0', 'verdict': 'unsupported', 'reason': 'Wrong column'}]})
        with self.assertRaises(ValueError):
            parse_json('[{"claim":"Unrelated extraction object"}]')

    def test_source_formatting_normalization_keeps_content(self):
        result = page_lines({'page': 4, 'text': ' A  B\n\nC\x00 D \n'})
        self.assertEqual([r['text'] for r in result], ['A B', 'C D'])
        self.assertEqual([r['line'] for r in result], [1, 3])

    def test_research_prompt_excludes_probability_and_outcome(self):
        prompt = str(extraction_messages(self.unit))
        for forbidden in ('home_win', 'p_boost', '0.7654321'):
            self.assertNotIn(forbidden, prompt)

    def test_support_check_requires_exactly_one_decision_per_fact(self):
        facts = validate_extraction(self.raw, self.unit)['facts']
        decision = {'decisions': [{'id': '0', 'verdict': 'supported', 'reason': 'Dated row and player header agree.'}]}
        checked = validate_verification(decision, {'facts': facts})
        self.assertEqual(checked['decisions'][0]['id'], facts[0]['id'])
        with self.assertRaises(ValueError):
            validate_verification({'decisions': []}, {'facts': facts})
        with self.assertRaises(ValueError):
            validate_verification({'decisions': decision['decisions'] * 2}, {'facts': facts})

    def test_support_checker_receives_uncited_later_rows(self):
        facts = validate_extraction(self.raw, self.unit)['facts']
        facts[0]['citations'] = facts[0]['citations'][:2]
        task = {'game': self.unit['game'], 'facts': facts, 'lines': self.unit['lines']}
        text = str(verification_messages(task))
        self.assertIn('12/01/24 at TOR 38:36', text)
        self.assertNotIn('p_boost', text)
        self.assertNotIn('home_win', text)

    def test_tokenizer_mapping_cannot_be_counted_as_token_ids(self):
        class BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        with self.assertRaises(ValueError):
            encode_messages(BrokenTokenizer(), [])


if __name__ == '__main__':
    unittest.main()
