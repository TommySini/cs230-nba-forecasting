"""Regression checks for temporal leakage and historical-source mislabeling."""
import copy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_pilot import parse_injury_pages, prior_features, preview_decision, get_injuries, stamp, sample_windows, select_sample, output_paths, collect_games


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.game = {"game_id": "espn:3", "espn_id": "3", "season": 2024,
                     "game_date_et": "2024-11-03", "home_abbr": "BOS", "away_abbr": "NYK",
                     "home_team": "Boston Celtics", "away_team": "New York Knicks",
                     "home_score": 110, "away_score": 100}
        self.cutoff = stamp("2024-11-03T23:00:00Z")

    def test_future_and_target_outcomes_do_not_change_features(self):
        prior = dict(self.game, game_id="espn:1", game_date_et="2024-11-01", home_score=90, away_score=100)
        future = dict(self.game, game_id="espn:4", game_date_et="2024-11-04")
        history = [prior, self.game, future]
        expected = prior_features(self.game, history, "home")
        altered = copy.deepcopy(history)
        altered[1]["home_score"] = 999
        altered[2]["home_score"] = 999
        self.assertEqual(expected, prior_features(self.game, altered, "home"))
        self.assertEqual(expected["prior_game_ids"], ["espn:1"])
        self.assertEqual(expected["last_5_point_differential"], -10)

    def test_seasons_do_not_mix(self):
        prior = dict(self.game, season=2023, game_date_et="2023-11-02")
        self.assertEqual(prior_features(self.game, [prior], "home")["previous_games"], 0)

    def test_same_calendar_date_results_are_excluded(self):
        same_day = dict(self.game, game_id="espn:2", home_score=999)
        self.assertEqual(prior_features(self.game, [same_day, self.game], "home")["previous_games"], 0)

    def test_cross_year_windows_preserve_season_and_sample_independently_of_outcomes(self):
        config = {"seed": 230, "seasons": [{"start_year": 2024}], "sample_windows": [
            {"name": "midseason", "year_offset": 1, "start_month_day": "01-15",
             "end_month_day": "01-24", "sample_size": 2}]}
        history = [dict(self.game, game_id=f"espn:{i}", game_date_et=f"2025-01-{15+i}",
                        scheduled_start_utc=f"2025-01-{15+i}T23:00:00Z") for i in range(5)]
        history.append(dict(history[0], game_id="wrong-season", season=2025))
        selected = select_sample(history, config)
        self.assertEqual(sample_windows(config)[0]["start"], "2025-01-15")
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(g["season"] == 2024 for g in selected))
        for game in history:
            game["home_score"] = 999
        self.assertEqual([g["game_id"] for g in selected], [g["game_id"] for g in select_sample(history, config)])

    def test_overlapping_sample_windows_are_rejected(self):
        window = {"name": "midseason", "year_offset": 1, "start_month_day": "01-15",
                  "end_month_day": "01-24", "sample_size": 1}
        with self.assertRaises(ValueError):
            sample_windows({"seasons": [{"start_year": 2024}], "sample_windows": [window, window]})

    def test_seasonal_outputs_cannot_replace_original_pilot(self):
        self.assertNotEqual(output_paths({}), output_paths({"output_name": "seasonal"}))
        with self.assertRaises(ValueError):
            output_paths({"output_name": "../"})

    def test_missing_schedule_day_aborts_and_successful_retry_clears_failure(self):
        config = {"seasons": [{"start_year": 2024, "sample_size": 1}],
                  "history_start_month_day": "10-01", "sample_start_month_day": "10-01",
                  "sample_end_month_day": "10-01", "output_name": "seasonal"}

        class FakeCache:
            body = None

            def get(self, url, kind, suffix):
                return self.body, {"url": url, "status": "network_error" if self.body is None else 200}

        cache = FakeCache()
        with TemporaryDirectory() as directory, patch("collect_pilot.ROOT", Path(directory)):
            with self.assertRaisesRegex(RuntimeError, "Incomplete schedule history"):
                collect_games(cache, config)
            failure_path = Path(directory) / "reports/seasonal/schedule_failures.json"
            self.assertTrue(failure_path.exists())
            cache.body = b'{"events": []}'
            self.assertEqual(collect_games(cache, config), [])
            self.assertFalse(failure_path.exists())

    def test_preview_endpoint_returning_recap_is_rejected(self):
        article = {"type": "Recap", "gameId": "3", "story": "The home team won."}
        self.assertEqual(preview_decision(article, self.game, self.cutoff), "rejected_not_preview")

    def test_preview_updated_after_cutoff_is_rejected(self):
        article = {"type": "Preview", "gameId": "3", "story": "Pregame text.",
                   "published": "2024-11-03T12:00:00Z", "lastModified": "2024-11-04T03:00:00Z"}
        self.assertEqual(preview_decision(article, self.game, self.cutoff), "rejected_after_cutoff")
        article["lastModified"] = "2024-11-03T12:00:00Z"
        self.assertEqual(preview_decision(article, self.game, self.cutoff), "candidate_requires_content_audit")

    def test_preview_wrong_game_is_rejected(self):
        self.assertEqual(preview_decision({"type": "Preview", "gameId": "999"}, self.game, self.cutoff),
                         "rejected_game_mismatch")

    def test_injury_pages_preserve_game_boundaries_and_dst(self):
        prefix = "Injury Report: 11/03/24 05:30 PM Page 1 of 2 Game Date Game Time Matchup Team Player Name Current Status Reason "
        pages = [prefix + "11/03/2024 07:00 (ET) NYK@BOS New York Knicks Player, One Out Injury",
                 "Injury Report: 11/03/24 05:30 PM Page 2 of 2 Game Date Game Time Matchup Team Player Name Current Status Reason Boston Celtics Player, Two Questionable Illness 08:00 (ET) LAL@GSW Los Angeles Lakers NOT YET SUBMITTED 11/04/2024 07:00 (ET) CLE@MIL Cleveland Cavaliers Player, Three Out"]
        timestamp, blocks = parse_injury_pages(pages)
        self.assertEqual(timestamp, "2024-11-03T22:30:00Z")  # Standard time after DST ends.
        self.assertEqual(len(blocks), 3)
        self.assertIn("Boston Celtics", blocks[0]["text"])
        self.assertNotIn("Los Angeles", blocks[0]["text"])
        self.assertEqual(blocks[2]["game_date_et"], "2024-11-04")

    def test_internal_timestamp_overrides_filename(self):
        class FakeCache:
            def get(self, url, kind, suffix):
                return b"pdf", {"url": url, "status": 200}
        block = {"game_date_et": "2024-11-03", "matchup": "NYK@BOS", "text": "Boston Celtics Player, One Out"}
        with patch("collect_pilot.injury_urls", return_value=["05PM.pdf", "04PM.pdf"]), patch(
            "collect_pilot.parse_injury", side_effect=[("2024-11-03T23:30:00Z", [block]), ("2024-11-03T22:30:00Z", [block])]
        ):
            report, attempts = get_injuries(FakeCache(), self.game, self.cutoff)
        self.assertEqual(attempts[0]["decision"], "after_cutoff")
        self.assertEqual(report["report_time_utc"], "2024-11-03T22:30:00Z")

    def test_matchups_sharing_a_printed_time_are_separate(self):
        page = "Injury Report: 11/01/23 05:30 PM 11/01/2023 07:00 (ET) MIL@TOR Milwaukee Bucks Player, One Out POR@DET Portland Trail Blazers Player, Two Out"
        _, blocks = parse_injury_pages([page])
        self.assertEqual([b["matchup"] for b in blocks], ["MIL@TOR", "POR@DET"])
        self.assertNotIn("Portland", blocks[0]["text"])
        self.assertEqual(blocks[1]["game_date_et"], "2023-11-01")


if __name__ == "__main__":
    unittest.main()
