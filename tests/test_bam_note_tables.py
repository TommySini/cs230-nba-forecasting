import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bam_note_tables import decode_snapshot_tables


def game(lines, day='2024-01-29'):
    return {'game_date_et':day,'sources':[{'source_id':'sha256:'+'a'*64,'pages':[
        {'page':1,'lines':[{'line':i+1,'text':text} for i,text in enumerate(lines)]}]}]}


PLAYER = [
    '11/8 @ 76ers 7:00 pm --       PLAYER / 2023-24 REGULAR SEASON AVERAGES',
    '11/11 vs Raptors 7:00 pm --   #0        Jayson Tatum       F • 6-8 • 210 • Duke/USA',
    '11/15 @ 76ers 7:30 pm --          GP/GS       PPG     RPG    APG   SPG   BPG   MPG',
    '11/17 @ Raptors 7:30 pm --        43/43       26.8    8.4    4.3   1.0   0.6   35.8',
]
TEAM = [
    '2023-24 REGULAR SEASON  PPG  OPP. PPG  FG%  3FG%  FT%  RPG  APG  SPG  TO  BPG',
    'Cavaliers 114.8 109.3 .482 .362 .772 45.1 26.9 7.74 13.6 4.76',
    'Raptors   114.3 117.5 .479 .360 .748 43.3 29.4 7.56 13.4 4.88',
    '2023-24 REGULAR SEASON INDIVIDUAL BESTS',
    'Points 31 Caris LeVert at TOR 1/1/24',
]


class SeasonTables(unittest.TestCase):
    def test_player_card_keeps_sample_season_units_and_complete_citations(self):
        facts,failures=decode_snapshot_tables(game(PLAYER))
        self.assertFalse(failures)
        self.assertEqual(len(facts),1)
        fact=facts[0]
        self.assertEqual(fact['parsed_values']['player'],'Jayson Tatum')
        self.assertEqual(fact['parsed_values']['games_played'],43)
        self.assertEqual(fact['parsed_values']['points_per_game'],26.8)
        self.assertEqual(fact['citations'][0]['lines'],[1,2,3,4])
        self.assertEqual(fact['time_scope'],'season_to_date')
        self.assertIn('does not confirm availability',fact['uncertainty'])

    def test_preseason_and_prior_season_are_historical(self):
        lines=PLAYER.copy();lines[0]=lines[0].replace('REGULAR SEASON','PRESEASON')
        facts,_=decode_snapshot_tables(game(lines))
        self.assertEqual(facts[0]['time_scope'],'historical')
        self.assertEqual(facts[0]['parsed_values']['season_type'],'preseason')
        facts,_=decode_snapshot_tables(game(PLAYER,'2025-01-29'))
        self.assertEqual(facts[0]['time_scope'],'historical')

    def test_right_side_prose_does_not_become_extra_columns(self):
        lines=PLAYER.copy();lines[-1]+='     NOTES: Scored 35 points last night.'
        facts,failures=decode_snapshot_tables(game(lines))
        self.assertEqual(len(facts),1)
        self.assertFalse(failures)
        self.assertNotIn('35 points',facts[0]['fact'])

    def test_blank_card_row_with_left_schedule_and_right_schedule_are_separate(self):
        lines=PLAYER.copy()
        lines.insert(3,'11/15 @ 76ers 117-107 W 9-2')
        lines[-1]+='         10/30 vs. Lakers 134-110 W 5-0'
        facts,failures=decode_snapshot_tables(game(lines))
        self.assertEqual(len(facts),1)
        self.assertFalse(failures)
        self.assertEqual(facts[0]['citations'][0]['lines'],[1,2,3,5])

    def test_no_heading_no_name_or_changed_header_stays_undecoded(self):
        for remove in (0,1):
            lines=PLAYER.copy();lines[remove]='unrelated paragraph'
            self.assertFalse(decode_snapshot_tables(game(lines))[0])
        lines=PLAYER.copy();lines[2]=lines[2].replace('PPG     RPG','RPG     PPG')
        self.assertFalse(decode_snapshot_tables(game(lines))[0])

    def test_ambiguous_counts_extra_values_and_future_season_fail(self):
        for old,new in [('43/43','43/44'),('26.8','-26.8'),('35.8','35.8 1.2')]:
            lines=PLAYER.copy();lines[-1]=lines[-1].replace(old,new)
            facts,failures=decode_snapshot_tables(game(lines))
            self.assertFalse(facts)
            self.assertTrue(failures)
        facts,failures=decode_snapshot_tables(game(PLAYER,'2023-01-29'))
        self.assertFalse(facts);self.assertTrue(failures)

    def test_second_player_has_own_identity(self):
        lines=PLAYER+['#7   Jaylen Brown       G • 6-6',
            'GP/GS PPG RPG APG SPG BPG MPG','40/40 22.6 5.4 3.5 1.1 0.5 33.0']
        facts,failures=decode_snapshot_tables(game(lines))
        self.assertFalse(failures)
        self.assertEqual([f['parsed_values']['player'] for f in facts],['Jayson Tatum','Jaylen Brown'])
        self.assertEqual(facts[1]['citations'][0]['lines'],[1,5,6,7])

    def test_wrapped_or_unknown_next_player_cannot_inherit_previous_name(self):
        lines=PLAYER+['#8             C • 7-3 • 240', '        Kristaps Porzingis',
                     'GP/GS PPG RPG APG SPG BPG MPG','33/33 19.5 6.8 2.0 0.7 1.8 29.7']
        facts,failures=decode_snapshot_tables(game(lines))
        self.assertEqual([f['parsed_values']['player'] for f in facts],['Jayson Tatum','Kristaps Porzingis'])
        self.assertEqual(facts[1]['citations'][0]['lines'],[1,5,6,7,8])
        lines[5]='unreadable 123'
        facts,failures=decode_snapshot_tables(game(lines))
        self.assertEqual(len(facts),1)
        self.assertEqual(facts[0]['parsed_values']['games_played'],43)

    def test_new_scope_and_horizontal_misalignment_cannot_reuse_season_header(self):
        simple=['PLAYER / 2023-24 REGULAR SEASON AVERAGES','#0   Jayson Tatum',
                'GP/GS PPG RPG APG SPG BPG MPG','43/43 26.8 8.4 4.3 1.0 0.6 35.8']
        for scope in ['LAST 5 GAMES','RECENT GAMES','CAREER AVERAGES','2023-24 PLAYOFFS']:
            lines=simple+[scope,'#7   Jaylen Brown','GP/GS PPG RPG APG SPG BPG MPG',
                          '5/5 22.6 5.4 3.5 1.1 0.5 33.0']
            self.assertEqual(len(decode_snapshot_tables(game(lines))[0]),1)
        lines=simple.copy();lines[-1]=' '*50+lines[-1]
        facts,failures=decode_snapshot_tables(game(lines))
        self.assertFalse(facts);self.assertTrue(failures)
        for scope in ['LAST 5 GAMES','PLAYER / 2022-23 REGULAR SEASON AVERAGES']:
            lines=simple[:-1]+[scope,'5/5 31.0 7.0 5.0 1.0 1.0 36.0']
            self.assertFalse(decode_snapshot_tables(game(lines))[0])

    def test_team_rows_keep_shooting_fractions_without_new_arithmetic(self):
        facts,failures=decode_snapshot_tables(game(TEAM))
        self.assertFalse(failures);self.assertEqual(len(facts),2)
        self.assertEqual(facts[0]['parsed_values']['three_point_fraction'],.362)
        self.assertEqual(facts[0]['parsed_values']['opponent_points_per_game'],109.3)
        self.assertEqual(facts[0]['citations'][0]['lines'],[1,2])
        self.assertIn('does not specify a game-count',facts[0]['uncertainty'])

    def test_team_header_change_and_missing_column_never_shift_values(self):
        lines=TEAM.copy();lines[0]=lines[0].replace('FG%  3FG%','3FG%  FG%')
        self.assertFalse(decode_snapshot_tables(game(lines))[0])
        lines=TEAM.copy();lines[1]=lines[1].replace(' .482','')
        self.assertFalse(decode_snapshot_tables(game(lines))[0])
        lines=TEAM.copy();lines[1]=lines[1].replace('.362','36.2')
        facts,failures=decode_snapshot_tables(game(lines))
        self.assertFalse(facts);self.assertTrue(failures)

    def test_raw_input_and_stable_ids_preserved(self):
        packet=game(PLAYER+TEAM);original=copy.deepcopy(packet)
        a,_=decode_snapshot_tables(packet);b,_=decode_snapshot_tables(packet)
        self.assertEqual(a,b);self.assertEqual(packet,original)


if __name__=='__main__':
    unittest.main()
