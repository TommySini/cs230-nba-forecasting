"""Ensure exact source columns, dates and arithmetic anchor numerical research."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from game_note_tables import HEADER, decode_recent_tables, table_facts, mechanical_claim_check
from run_game_note_research import page_lines


class GameNoteTableTests(unittest.TestCase):
    def setUp(self):
        text = '\n'.join(['BAM ADEBAYO RECENT GAMES', HEADER,
                          '11/29/24 vs TOR 34:36 6 16 2 6 0 1 2 8 10 10 2 3 1 0 0 14',
                          '12/01/24 at TOR 38:36 5 13 1 2 2 2 4 16 20 7 3 1 5 0 +15 13'])
        self.unit = {'id': 'unit', 'game': {'game_date_et': '2024-12-02'},
                     'lines': page_lines({'page': 10, 'text': text})}

    def test_column_mapping_keeps_field_goals_distinct_from_points(self):
        table = decode_recent_tables(self.unit)[0]
        latest = table['rows'][-1]
        self.assertEqual(latest['fg'], 5)
        self.assertEqual(latest['points'], 13)
        self.assertEqual(latest['rebounds'], 20)
        self.assertEqual(latest['seconds'], 38 * 60 + 36)
        facts = table_facts(self.unit)
        self.assertIn('13 points', facts[0]['claim'])
        self.assertIn('2024-12-01', facts[0]['claim'])
        self.assertIn('36.60 minutes and 13.50 points', facts[1]['claim'])
        self.assertEqual(facts[0]['citations'][-1], self.unit['lines'][-1])

    def test_corrupt_columns_and_target_game_data_are_rejected(self):
        wrong = copy.deepcopy(self.unit)
        wrong['lines'][-1]['text'] = wrong['lines'][-1]['text'][:-2] + '5'
        with self.assertRaises(ValueError):
            table_facts(wrong)
        future = copy.deepcopy(self.unit)
        future['game']['game_date_et'] = '2024-12-01'
        with self.assertRaises(ValueError):
            table_facts(future)

    def test_prior_nonparticipation_does_not_become_current_absence(self):
        self.unit['lines'][-1]['text'] = '12/01/24 at TOR NWT'
        fact = table_facts(self.unit)[0]
        self.assertEqual(fact['time_scope'], 'prior_game')
        self.assertIn('status NWT', fact['claim'])
        self.assertIn('does not confirm availability', fact['caveat'])

    def test_language_model_cannot_override_decoded_numeric_table(self):
        claim = {'claim': 'Bam scored five points.', 'category': 'form', 'time_scope': 'prior_game',
                 'citations': [self.unit['lines'][-1]]}
        self.assertEqual(mechanical_claim_check(claim, self.unit)['verdict'], 'unclear')
        self.unit['lines'][1]['text'] = 'UNKNOWN COLUMN ORDER'
        self.assertEqual(table_facts(self.unit), [])
        self.assertEqual(mechanical_claim_check(claim, self.unit)['verdict'], 'unclear')


if __name__ == '__main__':
    unittest.main()
