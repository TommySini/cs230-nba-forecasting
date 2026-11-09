"""Checks that documents showing their own game's result are flagged, and pregame content passes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from postgame_screen import postgame_flags

GAME = {"home_team": "New York Knicks", "away_team": "Philadelphia 76ers", "home_abbr": "NYK", "away_abbr": "PHI",
        "home_score": 112, "away_score": 105, "game_date_et": "2026-10-30",
        "home_record_before": (3, 2), "away_record_before": (2, 3)}


class PostgameScreenTests(unittest.TestCase):
    def test_recap_sentence_with_score_is_flagged(self):
        flags = postgame_flags("The Knicks beat the 76ers 112-105 on Oct. 30 at Madison Square Garden.", GAME)
        self.assertTrue(any(f.startswith("final_score") for f in flags))
        self.assertTrue(any(f.startswith("result_sentence") for f in flags))

    def test_game_log_row_for_this_game_is_flagged(self):
        self.assertTrue(postgame_flags("10/28 @ CHI W 120-99\n10/30 vs. PHI W 112-105\n", GAME))

    def test_score_with_both_teams_and_no_date_is_flagged(self):
        self.assertTrue(postgame_flags("Knicks 112, 76ers... final: NYK 112-105 PHI", GAME))

    def test_updated_header_record_is_flagged(self):
        flags = postgame_flags("NEW YORK KNICKS (4-2) vs. PHILADELPHIA 76ERS (2-4)", GAME)
        self.assertEqual(sorted(f.split(":")[0] for f in flags), ["post_game_record", "post_game_record"])

    def test_tonights_win_is_flagged(self):
        self.assertTrue(postgame_flags("After tonight's win over Philadelphia, New York is 4-2.", GAME))

    def test_pregame_notes_pass(self):
        text = ("NEW YORK KNICKS (3-2) vs. PHILADELPHIA 76ERS (2-3) | October 30, 2026\n"
                "The Knicks beat the 76ers 112-105 on Oct. 20 in the season opener.\n"
                "10/30/26 vs PHILADELPHIA\nRecord in last five: 3-2. Brunson 7-15 FG on Tuesday.\n"
                "Oct. 30, 1999: Patrick Ewing scored 30 as New York defeated Philadelphia.\n"
                "A win tonight would give New York its fourth straight victory.")
        self.assertEqual(postgame_flags(text, GAME), [])

    def test_old_score_near_page_header_date_passes(self):
        text = "(NYK L, 112-105) @ Boston, 03/31/24\n\n=== PDF PAGE 32 ===\n10/30/26 vs PHILADELPHIA\n"
        self.assertEqual(postgame_flags(text, GAME), [])


if __name__ == "__main__":
    unittest.main()
