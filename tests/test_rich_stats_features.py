"""Leakage, arithmetic and complete-window guarantees for rich box-score inputs."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rich_stats_features import (PLAYER_FEATURE_NAMES, TEAM_FEATURE_NAMES, RichStatsIndex,
                                 augment_inputs)
from run_gradient_boosting import FEATURE_NAMES


def fixtures(count=13):
    history, boxes = [], []
    for i in range(1, count + 1):
        game = {"game_id": f"g{i}", "season": 2024, "game_date_et": f"2024-11-{i:02}",
                "scheduled_start_utc": f"2024-11-{i:02}T23:00:00Z",
                "home_abbr": "BOS", "away_abbr": "NYK", "home_score": 110, "away_score": 105}
        history.append(game)
        box = {k: game[k] for k in ("game_id", "season", "game_date_et", "scheduled_start_utc")}
        box.update(complete=True, provenance={"provider": "fixture", "sha256": f"raw-{i}"})
        box["teams"] = {}
        for side, team in (("home", "BOS"), ("away", "NYK")):
            home = side == "home"
            stats = dict(team_abbr=team, minutes=240, game_minutes=48, complete=True, players_complete=True,
                         fgm=40 if home else 38, fga=80 if home else 90,
                         fg3m=10 if home else 9, fg3a=30 if home else 25,
                         ftm=20, fta=25 if home else 24, oreb=10 if home else 8,
                         dreb=30 if home else 32, reb=40, tov=12 if home else 15,
                         pts=110 if home else 105)
            minutes = [36, 34, 32, 30, 28, 24, 22, 18, 10, 6]
            attempts = [16, 14, 12, 10, 8, 6, 5, 4, 3, 2]
            free_throws = [6, 5, 4, 3, 2, 1, 1, 1, 1, 1]
            turnovers = [2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
            if not home:
                attempts[0] += 10
                free_throws[0] -= 1
                turnovers[0] += 3
            stats["players"] = [dict(player_id=f"{team}-p{j}", minutes=minutes[j], fga=attempts[j],
                                     fta=free_throws[j], tov=turnovers[j]) for j in range(10)]
            box["teams"][side] = stats
        boxes.append(box)
    return history, boxes


class RichStatsFeatureTests(unittest.TestCase):
    def setUp(self):
        self.history, self.boxes = fixtures()

    def build(self, target=5, include_players=True, history=None, boxes=None):
        history = self.history if history is None else history
        boxes = self.boxes if boxes is None else boxes
        return RichStatsIndex(history, boxes).for_game(history[target], include_players=include_players)

    def test_feature_names_windows_and_home_away_difference(self):
        row = self.build()
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertEqual(list(row["features"]), list(TEAM_FEATURE_NAMES + PLAYER_FEATURE_NAMES))
        self.assertEqual(len(TEAM_FEATURE_NAMES), 144)
        self.assertEqual(len(PLAYER_FEATURE_NAMES), 72)
        self.assertEqual(row["coverage"]["home"]["window_games"], {"season": 5, "last5": 5, "last10": 5})
        self.assertEqual(row["coverage"]["home"]["valid_previous_games"], 5)
        for name in TEAM_FEATURE_NAMES + PLAYER_FEATURE_NAMES:
            if name.startswith("difference_"):
                suffix = name[len("difference_"):]
                self.assertAlmostEqual(row["features"][name], row["features"]["home_" + suffix]
                                       - row["features"]["away_" + suffix])
        team_only = self.build(include_players=False)
        self.assertEqual(list(team_only["features"]), list(TEAM_FEATURE_NAMES))
        self.assertEqual(team_only["features"], {k: row["features"][k] for k in TEAM_FEATURE_NAMES})

    def test_target_and_future_box_scores_players_and_outcomes_are_never_read(self):
        before = self.build()
        changed_history, changed_boxes = copy.deepcopy(self.history), copy.deepcopy(self.boxes)
        for game in changed_history[5:]:
            game["home_score"], game["away_score"] = 999, 0
        for box in changed_boxes[5:]:
            box["teams"] = {"home": {"players": "corrupt future roster"}, "away": {"pts": float("nan")}}
            box["provenance"] = {"recap": "target-game winner and future stories"}
            box["complete"] = False
        after = self.build(history=changed_history, boxes=changed_boxes)
        self.assertEqual(before, after)
        self.assertEqual([r["game_id"] for r in before["lineage"]["home"]], [f"g{i}" for i in range(1, 6)])

    def test_same_day_and_previous_season_rows_are_excluded(self):
        before = self.build()
        history, boxes = copy.deepcopy(self.history), copy.deepcopy(self.boxes)
        for name, day, season, start in (("same-day", "2024-11-06", 2024, "2024-11-06T15:00:00Z"),
                                         ("old-season", "2024-04-01", 2023, "2024-04-01T23:00:00Z")):
            history.append({**self.history[0], "game_id": name, "game_date_et": day,
                            "season": season, "scheduled_start_utc": start, "home_score": 999})
            boxes.append({"game_id": name, "teams": None})
        after = self.build(history=history, boxes=boxes)
        self.assertEqual(before, after)

    def test_ratios_use_summed_counts_not_average_percentages(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["fga"] = 160
        row = self.build(boxes=boxes, include_players=False)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertAlmostEqual(row["features"]["home_season_effective_fg_pct"], 5 * 45 / (160 + 4 * 80))
        self.assertNotAlmostEqual(row["features"]["home_season_effective_fg_pct"], (45 / 160 + 4 * 45 / 80) / 5)

    def test_possession_estimate_rating_pace_and_overtime_minutes(self):
        row = self.build(include_players=False)
        estimated = .5 * ((80 - 10 + 12 + .44 * 25) + (90 - 8 + 15 + .44 * 24))
        self.assertAlmostEqual(row["features"]["home_last5_estimated_offensive_rating"], 100 * 110 / estimated)
        self.assertAlmostEqual(row["features"]["home_last5_estimated_defensive_rating"], 100 * 105 / estimated)
        self.assertAlmostEqual(row["features"]["home_last5_estimated_pace"], estimated)
        boxes = copy.deepcopy(self.boxes)
        for box in boxes[:5]:
            for side in ("home", "away"):
                box["teams"][side]["game_minutes"] = 53
                box["teams"][side]["minutes"] = 264 if side == "home" else 266
        overtime = self.build(boxes=boxes, include_players=False)
        self.assertTrue(overtime["eligible"], overtime["reasons"])
        self.assertAlmostEqual(overtime["features"]["home_last5_estimated_pace"], estimated * 48 / 53)
        self.assertEqual(overtime["lineage"]["home"][0]["duration_basis"], "normalized_game_minutes")

    def test_last_five_and_ten_are_actual_chronological_windows(self):
        boxes = copy.deepcopy(self.boxes)
        for box in boxes[:2]:
            box["teams"]["home"]["fga"] = 160
        row = self.build(target=12, boxes=boxes, include_players=False)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertEqual(row["coverage"]["home"]["window_games"], {"season": 12, "last5": 5, "last10": 10})
        self.assertAlmostEqual(row["features"]["home_last5_effective_fg_pct"], 45 / 80)
        self.assertAlmostEqual(row["features"]["home_last10_effective_fg_pct"], 45 / 80)
        self.assertNotAlmostEqual(row["features"]["home_season_effective_fg_pct"], 45 / 80)

    def test_conflicting_game_duration_sources_are_rejected(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["game_minutes"] = 53
        self.assertFalse(self.build(boxes=boxes, include_players=False)["eligible"])
        del boxes[0]["game_minutes"]
        boxes[0]["teams"]["away"]["game_minutes"] = 53
        self.assertFalse(self.build(boxes=boxes, include_players=False)["eligible"])

    def test_team_turnovers_need_not_equal_sum_of_player_turnovers(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["tov"] += 7
        row = self.build(boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertFalse(row["player_issues"])
        self.assertEqual(row["features"]["home_last5_player_boxscore_missing"], 0)

    def test_missing_box_score_invalidates_window_not_replaced_by_older_game(self):
        boxes = [r for r in self.boxes if r["game_id"] != "g10"]
        row = self.build(target=12, boxes=boxes)
        self.assertFalse(row["eligible"])
        self.assertFalse(row["features"])
        self.assertEqual(row["coverage"]["home"]["expected_previous_games"], 12)
        self.assertEqual(row["coverage"]["home"]["valid_previous_games"], 11)
        self.assertIn("g10", " ".join(row["reasons"]))

    def test_player_missingness_keeps_paired_cohort_and_marks_statistics_unknown(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["players_complete"] = False
        row = self.build(boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertEqual(len(row["player_issues"]), 1)
        self.assertTrue(math.isnan(row["features"]["home_last5_max_player_minutes_per_game"]))
        self.assertTrue(math.isnan(row["features"]["difference_last5_max_player_minutes_per_game"]))
        self.assertTrue(math.isfinite(row["features"]["away_last5_max_player_minutes_per_game"]))
        self.assertAlmostEqual(row["features"]["home_last5_player_boxscore_coverage"], .8)
        self.assertEqual(row["features"]["home_last5_player_boxscore_missing"], 1)
        self.assertEqual(row["coverage"]["home"]["window_player_games"], {"season": 4, "last5": 4, "last10": 4})
        team_only = self.build(boxes=boxes, include_players=False)
        self.assertTrue(team_only["eligible"], team_only["reasons"])
        self.assertEqual(team_only["features"], {k: row["features"][k] for k in TEAM_FEATURE_NAMES})

    def test_missing_player_game_does_not_replace_actual_window_or_poison_later_windows(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["players_complete"] = False
        row = self.build(target=12, boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertTrue(math.isnan(row["features"]["home_season_max_player_minutes_per_game"]))
        self.assertTrue(math.isfinite(row["features"]["home_last5_max_player_minutes_per_game"]))
        self.assertTrue(math.isfinite(row["features"]["home_last10_max_player_minutes_per_game"]))
        self.assertEqual(row["coverage"]["home"]["window_player_games"], {"season": 11, "last5": 5, "last10": 10})
        boxes[9]["teams"]["home"]["players_complete"] = False
        interior = self.build(target=12, boxes=boxes)
        self.assertEqual(interior["coverage"]["home"]["window_player_games"], {"season": 10, "last5": 4, "last10": 9})
        self.assertTrue(math.isnan(interior["features"]["home_last5_max_player_minutes_per_game"]))
        self.assertAlmostEqual(interior["features"]["home_last5_player_boxscore_coverage"], .8)

    def test_player_workload_absent_games_count_zero_and_continuity_changes(self):
        row = self.build()
        self.assertAlmostEqual(row["features"]["home_last5_max_player_minutes_per_game"], 36)
        self.assertAlmostEqual(row["features"]["home_last5_top5_minutes_share"], 160 / 240)
        self.assertAlmostEqual(row["features"]["home_last5_consecutive_game_minutes_overlap"], 1)
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["players"][0]["player_id"] = "past-replacement"
        replacement = self.build(boxes=boxes)
        self.assertAlmostEqual(replacement["features"]["home_last5_max_player_minutes_per_game"], 34)
        self.assertAlmostEqual(replacement["features"]["home_last5_consecutive_game_minutes_overlap"], 1 - (36 / 240) / 4)
        self.assertLess(replacement["features"]["home_last5_minutes_concentration"], row["features"]["home_last5_minutes_concentration"])

    def test_reliable_availability_after_cutoff_is_rejected_but_fetch_time_is_not(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["provenance"]["retrieved_at_utc"] = "2026-11-03T00:00:00Z"
        self.assertTrue(self.build(boxes=boxes)["eligible"])
        boxes[0]["stat_available_at_utc"] = "2024-11-06T22:00:01Z"
        rejected = self.build(boxes=boxes)
        self.assertFalse(rejected["eligible"])
        self.assertIn("unavailable at cutoff", " ".join(rejected["reasons"]))
        boxes[0]["stat_available_at_utc"] = "2024-11-06T22:00:00Z"
        self.assertTrue(self.build(boxes=boxes)["eligible"])

    def test_identity_and_team_counter_corruption_are_rejected(self):
        mutations = (
            lambda r: r.update(season=2023),
            lambda r: r["teams"]["home"].update(team_abbr="LAL"),
            lambda r: r["teams"]["home"].update(pts=111),
            lambda r: r["teams"]["home"].update(fga=float("nan")),
            lambda r: r["teams"]["home"].update(fga=80.5),
            lambda r: r["teams"]["home"].update(fgm=100),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                boxes = copy.deepcopy(self.boxes)
                mutate(boxes[0])
                self.assertFalse(self.build(boxes=boxes)["eligible"])

    def test_corrupt_player_rows_are_flagged_without_fabricated_values(self):
        for updates in ({"player_id": None}, {"player_id": "BOS-p1"}, {"minutes": None}, {"fga": 0}):
            with self.subTest(updates=updates):
                boxes = copy.deepcopy(self.boxes)
                boxes[0]["teams"]["home"]["players"][0].update(updates)
                row = self.build(boxes=boxes)
                self.assertTrue(row["eligible"], row["reasons"])
                self.assertTrue(row["player_issues"])
                self.assertTrue(math.isnan(row["features"]["home_last5_max_player_minutes_per_game"]))
                self.assertEqual(row["features"]["home_last5_player_boxscore_missing"], 1)

    def test_augmentation_preserves_baseline_and_reports_excluded_games(self):
        first = {**self.history[4], "features": {name: float(i) for i, name in enumerate(FEATURE_NAMES)}}
        second = {**self.history[5], "features": {name: float(i) for i, name in enumerate(FEATURE_NAMES)}}
        original = copy.deepcopy([first, second])
        augmented, excluded = augment_inputs([first, second], self.history, self.boxes)
        self.assertEqual([first, second], original)
        self.assertEqual([r["game_id"] for r in augmented], ["g6"])
        self.assertEqual([r["game_id"] for r in excluded], ["g5"])
        self.assertEqual(len(augmented[0]["features"]), 234)
        for name in FEATURE_NAMES:
            self.assertEqual(augmented[0]["features"][name], second["features"][name])


if __name__ == "__main__":
    unittest.main()
