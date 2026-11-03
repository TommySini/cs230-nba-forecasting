from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bam_pdf_notes import warriors_header_checks


class WarriorsHeaderTests(unittest.TestCase):
    def setUp(self):
        self.game = {"home_abbr": "GSW", "away_abbr": "CLE", "season": 2023, "game_date_et": "2024-01-01"}
        self.history = [{"home_abbr": "GSW", "away_abbr": "CLE", "season": 2023,
                         "game_date_et": "2023-12-" + str(20 + i), "home_score": score, "away_score": 90}
                        for i, score in enumerate((100, 80, 100))]
        self.text = "GOLDEN STATE WARRIORS 2023-24 GAME NOTES\nWARRIORS (2-1) VS. CAVALIERS (1-2)\nMONDAY, JANUARY 1, 2024\nGAME #4 / HOME GAME #2\n"

    def test_original_nickname_records_date_and_game_number_all_verified(self):
        reasons, checks = warriors_header_checks(self.text, self.game, self.history)
        self.assertEqual(reasons, [])
        self.assertEqual(checks["away_record"]["actual"], (1, 2))
        self.assertEqual(checks["publisher_game_number"]["expected"], 4)
        self.assertEqual(checks["literal_matchup_line"], "WARRIORS (2-1) VS. CAVALIERS (1-2)")

    def test_wrong_opponent_record_date_weekday_number_or_routing_rejected(self):
        for old, new in [("CAVALIERS", "HEAT"), ("(1-2)", "(2-1)"), ("JANUARY 1", "JANUARY 2"),
                         ("MONDAY", "SUNDAY"), ("GAME #4", "GAME #5"), ("HOME GAME", "ROAD GAME"),
                         (" VS. ", " AT "), ("2023-24", "2022-23")]:
            with self.subTest(change=(old, new)):
                self.assertTrue(warriors_header_checks(self.text.replace(old, new), self.game, self.history)[0])

    def test_ambiguous_duplicate_header_and_unrelated_schedule_date_cannot_rescue(self):
        self.assertTrue(warriors_header_checks(self.text + self.text, self.game, self.history)[0])
        no_header_date = self.text.replace("MONDAY, JANUARY 1, 2024", "01/01 vs CLE 5:30 pm")
        self.assertIn("missing_or_ambiguous_full_warriors_header_date", warriors_header_checks(no_header_date, self.game, self.history)[0])


if __name__ == "__main__":
    unittest.main()
