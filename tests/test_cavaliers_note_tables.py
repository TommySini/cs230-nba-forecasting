"""Cleveland table fixtures cover right-hand sidebars, missing values and dates."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cavaliers_note_tables import HEADER, packet_facts, parse_row


class CavaliersTableTests(unittest.TestCase):
    def packet(self):
        lines = ['CLEVELAND CAVALIERS 2024-25 PLAYER PAGES 11/04/24 VS. MILWAUKEE',
                 '#2 Ty Jerome', 'RECENT GAMES', HEADER + ' FTA 6 2 times 7 at LAL 2/10/20',
                 '11/2 at MIL W 8:46 1-2 0-0 1-1 0 2 2 3 1 0 0 0 3 OREB 1 3 times 5 at PHX 12/29/21',
                 '11/1 vs. ORL W 21:00 3-9 1-3 2-2 0 6 6 2 0 2 0 0 9 DREB 6 vs. ORL 11/1/24 7 vs. UTA 12/28/22',
                 '10/30 vs. LAL W ASTS -- 3 2 times', 'NOTES:',
                 '#3 Caris LeVert', 'RECENT GAMES', HEADER,
                 '11/2 at MIL W Inactive - Injury OREB 1 2 times 5 vs. IND 10/30/19',
                 '11/1 vs. ORL W 22:20 5-12 2-7 1-2 0 3 3 7 2 0 2 0 13', 'NOTES:']
        return {'game_id': 'sample', 'publisher': 'CLE', 'game_date_et': '2024-11-04',
                'pages': [{'page': 12, 'text': '\n'.join(lines)}]}

    def test_sidebar_numbers_do_not_enter_player_statistics(self):
        facts, issues = packet_facts(self.packet())
        text = '\n'.join(f['claim'] for f in facts)
        self.assertIn('8:46 minutes, 3 points, 2 rebounds and 3 assists', text)
        self.assertIn('14.88 minutes and 6.00 points', text)
        self.assertIn('Caris LeVert', text)
        self.assertIn('status Inactive - Injury', text)
        self.assertTrue(any('preserved as unknown' in issue['reason'] for issue in issues))
        self.assertTrue(all('current availability' in f['caveat'] or 'does not confirm availability' in f['caveat'] or 'Not a full-season average' in f['caveat'] for f in facts))

    def test_blank_entry_is_unknown_not_an_inferred_absence(self):
        for raw in ('11/2 at MIL W', '11/2 at MIL W ASTS -- 3 2 times'):
            row = parse_row(raw, 2024, '2024-11-04')
            self.assertTrue(row['missing_detail'])
            self.assertEqual(row['status'], 'participation detail not specified in extracted text')

    def test_bad_scoring_identity_excludes_whole_affected_table(self):
        packet = self.packet()
        packet['pages'][0]['text'] = packet['pages'][0]['text'].replace('0 0 3 OREB', '0 0 2 OREB')
        facts, issues = packet_facts(packet)
        self.assertFalse(any('Ty Jerome' in f['claim'] for f in facts))
        self.assertTrue(any('Caris LeVert' in f['claim'] for f in facts))
        self.assertTrue(any('Entire table excluded' in issue['reason'] for issue in issues))

    def test_nba_season_sets_row_year_and_target_game_is_rejected(self):
        suffix = ' at MIL W 8:46 1-2 0-0 1-1 0 2 2 3 1 0 0 0 3'
        self.assertEqual(parse_row('12/30' + suffix, 2024, '2025-01-02')['date'], '2024-12-30')
        self.assertEqual(parse_row('1/1' + suffix, 2024, '2025-01-02')['date'], '2025-01-01')
        with self.assertRaises(ValueError):
            parse_row('1/2' + suffix, 2024, '2025-01-02')


if __name__ == '__main__':
    unittest.main()
