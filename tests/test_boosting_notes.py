"""Regression checks for leakage boundaries and the probability-correction protocol."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_game_notes import clean_front_page, content_checks
from run_boosting_notes import log_odds, prompt_for, whole_date_splits


class BoostingNotesTests(unittest.TestCase):
    def test_zero_correction_preserves_boosting(self):
        p = np.array([.001, .1, .5, .8, .999])
        np.testing.assert_allclose(1 / (1 + np.exp(-log_odds(p))), p, atol=1e-14)

    def test_split_keeps_dates_and_booster_fit_out_of_dev_test(self):
        rows = [{"game_id": str(i), "game_date_et": f"2025-11-{i // 2 + 1:02}"} for i in range(24)]
        config = {"requested_fractions": [.75, .125, .125], "minimum_split_games": {"train": 8, "dev": 2, "test": 2}, "model_revision_date": "2024-09-25"}
        result = whole_date_splits(rows, {str(i) for i in range(16)}, config)
        self.assertFalse({r["game_id"] for r in result if r["split"] != "train"} & {str(i) for i in range(16)})
        self.assertTrue(all(len({r["split"] for r in result if r["game_date_et"] == d}) == 1 for d in {r["game_date_et"] for r in result}))
        with self.assertRaises(ValueError):
            whole_date_splits(rows, {str(i) for i in range(23)}, config)

    def test_schedule_removal_preserves_player_stats_and_dated_header(self):
        raw = "GAME NOTES 11/08/23\n11/8 @ Thunder 8:00 pm --\n7/7 13.7 5.7 3.3\n10/25 NEW ORLEANS L 104-111\nPlayer scored 20 points in the prior game."
        result = clean_front_page(raw)
        self.assertNotIn("@ Thunder", result)
        self.assertNotIn("L 104-111", result)
        self.assertIn("7/7 13.7", result)
        self.assertIn("11/08/23", result)

    def test_wrong_date_and_wrong_prior_record_are_rejected(self):
        game = {"game_date_et": "2025-11-08", "season": 2025, "home_team": "Cleveland Cavaliers", "home_abbr": "CLE", "away_team": "New York Knicks", "away_abbr": "NYK"}
        header = "2025-26 GAME NOTES 11/08/25 CLEVELAND CAVALIERS (0-0) VS NEW YORK KNICKS (0-0) SEASON GAME #1 HOME GAME #1\nLAST GAME STARTERS\n"
        text = header + "Historical player and matchup information. " * 40
        self.assertEqual(content_checks(text, game, "CLE", [])[0], [])
        self.assertIn("header_date_mismatch_or_missing", content_checks(text.replace("11/08/25", "11/07/25"), game, "CLE", [])[0])
        self.assertIn("home_record_disagrees_with_prior_history", content_checks(text.replace("CAVALIERS (0-0)", "CAVALIERS (1-0)"), game, "CLE", [])[0])
        self.assertIn("header_year_mismatch", content_checks(text.replace("11/08/25", "November 8, 2024"), game, "CLE", [])[0])

    def test_hybrid_prompt_uses_probability_and_notes_not_label_or_raw_stats(self):
        record = {"home_abbr": "CLE", "away_abbr": "NYK", "p_boost": .63, "note_evidence": "Verified pregame player news.",
                  "home_win": 1, "game_id": "secret-target-id", "features": {"home_season_win_rate": .12345678}, "raw_espn": "Recap with final score"}
        prompt = prompt_for(record)
        self.assertIn("0.630000", prompt)
        self.assertIn("Verified pregame player news.", prompt)
        for excluded in ("home_win", "secret-target-id", "home_season_win_rate", "0.12345678", "Recap with final score"):
            self.assertNotIn(excluded, prompt)


if __name__ == "__main__":
    unittest.main()
