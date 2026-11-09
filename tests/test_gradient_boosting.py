"""Guard the boosting baseline: prior-only features, coverage handling, split isolation and scoring."""
import copy
import hashlib
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from box_score_features import (BOX_SCORE_FEATURE_NAMES, PLAYER_FEATURE_NAMES, TEAM_COVERAGE_FEATURE_NAMES,
                                TEAM_FEATURE_NAMES, BoxScoreIndex)
import run_gradient_boosting as gb
from run_gradient_boosting import (BASIC_FEATURE_NAMES, FEATURE_NAMES, arrays, bootstrap_brier, franchise_game,
                                   new_model, predict, prepare_examples, score_predictions, season_splits, select_and_fit)
from validate_gradient_boosting import verify_hashes

CONFIG = {"seed": 230, "cutoff_minutes": 60, "min_previous_games": 5, "min_box_score_games": 5,
          "split_seasons": {"train": [2021, 2023], "dev": [2024, 2024], "test": [2025, 2025]},
          "selection_metric": "brier", "model_parameters": {"n_iter_no_change": None},
          "candidates": [{"max_depth": 1}, {"max_depth": 2}]}


def fixtures(count=13):
    history, boxes = [], []
    for i in range(1, count + 1):
        game = {"game_id": f"g{i}", "season": 2024, "game_date_et": f"2024-11-{i:02}",
                "scheduled_start_utc": f"2024-11-{i:02}T23:00:00Z",
                "home_abbr": "BOS", "away_abbr": "NYK", "home_score": 110, "away_score": 105}
        history.append(game)
        box = {k: game[k] for k in ("game_id", "season", "game_date_et", "scheduled_start_utc")}
        box.update(complete=True, provenance={"provider": "fixture", "sha256": f"raw-{i}"}, teams={})
        for side, team in (("home", "BOS"), ("away", "NYK")):
            home = side == "home"
            stats = dict(team_abbr=team, minutes=240, game_minutes=48, complete=True, players_complete=True,
                         fgm=40 if home else 38, fga=80 if home else 90, fg3m=10 if home else 9,
                         fg3a=30 if home else 25, ftm=20, fta=25 if home else 24, oreb=10 if home else 8,
                         dreb=30 if home else 32, reb=40, tov=12 if home else 15, pts=110 if home else 105)
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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.history, self.boxes = fixtures()

    def examples(self, history=None, boxes=None):
        return prepare_examples(self.history if history is None else history,
                                self.boxes if boxes is None else boxes, CONFIG)

    def test_feature_schema_keeps_rich_column_order_and_merges_basic_values(self):
        self.assertEqual(len(FEATURE_NAMES), 252)
        self.assertEqual(FEATURE_NAMES[:234], BASIC_FEATURE_NAMES + TEAM_FEATURE_NAMES + PLAYER_FEATURE_NAMES)
        inputs, labels, excluded = self.examples()
        self.assertEqual([r["game_id"] for r in excluded], [f"g{i}" for i in range(1, 6)])
        row = inputs[0]
        self.assertEqual((row["game_id"], row["split"]), ("g6", "dev"))
        self.assertEqual(list(row["features"]), list(FEATURE_NAMES))
        self.assertEqual(row["features"]["home_season_win_rate"], 1.0)
        self.assertEqual(row["features"]["away_last_5_point_differential"], -5.0)
        self.assertEqual(row["features"]["home_last5_team_boxscore_coverage"], 1.0)
        self.assertEqual(row["prior_game_ids"]["home"], [f"g{i}" for i in range(1, 6)])
        self.assertEqual(row["missing_box_score_game_ids"], {"home": [], "away": []})
        self.assertFalse({"home_win", "home_score", "away_score"} & set(row))
        self.assertEqual(labels[0], {"game_id": "g6", "home_win": 1})

    def test_outcomes_at_or_after_target_cannot_change_target_features(self):
        before, labels, _ = self.examples()
        history = copy.deepcopy(self.history)
        for game in history[5:]:
            game["home_score"], game["recap"] = 1, "The away team won."
        after, new_labels, _ = self.examples(history=history)
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(labels[0], new_labels[0])

    def test_same_day_outcome_is_excluded_from_target(self):
        same_day = dict(self.history[5], game_id="early-same-day", scheduled_start_utc="2024-11-06T15:00:00Z", home_score=999)
        before, _, _ = self.examples()
        after, _, _ = self.examples(history=self.history + [same_day])
        target = next(r for r in after if r["game_id"] == "g6")
        self.assertEqual(before[0]["features"], target["features"])
        self.assertNotIn("early-same-day", target["prior_game_ids"]["home"])

    def test_unknown_player_statistics_serialize_as_null(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["players_complete"] = False
        row = self.examples(boxes=boxes)[0][0]
        self.assertIsNone(row["features"]["home_season_max_player_minutes_per_game"])
        self.assertEqual(row["features"]["home_season_player_boxscore_missing"], 1.0)

    def test_splits_are_chronological_and_every_season_assigned(self):
        self.assertEqual(season_splits(CONFIG), {2021: "train", 2022: "train", 2023: "train", 2024: "dev", 2025: "test"})
        for bad in ({"train": [2021, 2024], "dev": [2024, 2024], "test": [2025, 2025]},
                    {"train": [2021, 2023], "dev": [2025, 2025], "test": [2024, 2024]}):
            with self.assertRaises(ValueError):
                season_splits({"split_seasons": bad})
        history = copy.deepcopy(self.history)
        history[0]["season"] = 2019
        with self.assertRaisesRegex(ValueError, "Unassigned"):
            self.examples(history=history)

    def test_all_star_exhibitions_are_not_franchise_games(self):
        self.assertTrue(franchise_game({"home_abbr": "GS", "away_abbr": "NYK"}))
        self.assertFalse(franchise_game({"home_abbr": "LEB", "away_abbr": "GIA"}))

    def test_misaligned_outcomes_are_rejected(self):
        inputs, labels, _ = self.examples()
        with self.assertRaisesRegex(ValueError, "align"):
            arrays(inputs, labels[::-1])


class HistoricalProvenanceTests(unittest.TestCase):
    def test_pinned_source_does_not_excuse_changed_data_or_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, body in {"scripts/model.py": b"original code", "config.json": b"{}",
                               "data/input": b"original data", "data/predictions": b"frozen predictions"}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            def git(*args):
                return subprocess.run(["git", "-C", directory, *args], check=True, capture_output=True, text=True).stdout.strip()
            git("init", "-q")
            git("add", "scripts/model.py", "config.json")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Original source")
            revision = git("rev-parse", "HEAD")
            digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            results = {"source_sha256": {name: digest(name) for name in ("scripts/model.py", "config.json", "data/input")},
                       "artifact_sha256": {"data/predictions": digest("data/predictions")}}
            (root / "scripts/model.py").write_text("new reporting path")
            with self.assertRaisesRegex(AssertionError, "Source hash"):
                verify_hashes(results, root=root)
            verify_hashes(results, revision, root)
            with self.assertRaisesRegex(ValueError, "immutable"):
                verify_hashes(results, "HEAD", root)
            wrong = copy.deepcopy(results)
            wrong["source_sha256"]["scripts/model.py"] = "0" * 64
            with self.assertRaisesRegex(AssertionError, "Source hash"):
                verify_hashes(wrong, revision, root)
            for name in ("config.json", "data/input", "data/predictions"):
                original = (root / name).read_bytes()
                (root / name).write_bytes(b"changed")
                with self.assertRaisesRegex(AssertionError, "hash changed"):
                    verify_hashes(results, revision, root)
                (root / name).write_bytes(original)


class FakeModel:
    def __init__(self, candidate):
        self.candidate = candidate

    def predict_proba(self, x):
        odd = x[:, 0] % 2
        p = 0.4 + 0.2 * odd if self.candidate["max_depth"] == 1 else 0.8 - 0.6 * odd
        return np.column_stack([1 - p, p])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(12, dtype=float).reshape(-1, 1)
        self.y = np.array([0, 1] * 6)
        self.splits = np.array(["train"] * 8 + ["dev"] * 2 + ["test"] * 2)
        self.seasons = np.array([2021] * 4 + [2022] * 4 + [2024] * 2 + [2025] * 2)

    def run_selection(self, seasons):
        calls = []

        def fake_fit(config, candidate, x, y):
            calls.append((candidate, x[:, 0].tolist(), y.tolist()))
            return FakeModel(candidate)

        with patch.object(gb, "fit", side_effect=fake_fit):
            result = select_and_fit(self.x, self.y, self.splits, seasons, CONFIG, jobs=1)
        return result, calls

    def test_selection_fits_train_only_and_train_forecasts_use_earlier_seasons(self):
        seasons = np.array([2020] * 3 + [2021] * 3 + [2022] * 2 + [2024] * 2 + [2025] * 2)
        (best, trials, final, prior), calls = self.run_selection(seasons)
        self.assertEqual(best, 0)
        self.assertAlmostEqual(trials[0]["brier"], 0.16)
        self.assertEqual([c[1] for c in calls[:2]], [list(range(8))] * 2)
        self.assertEqual(sorted(prior), [2021, 2022])
        self.assertEqual(calls[2], ({"max_depth": 1}, [0, 1, 2], [0, 1, 0]))
        self.assertEqual(calls[3], ({"max_depth": 1}, [0, 1, 2, 3, 4, 5], [0, 1, 0, 1, 0, 1]))
        self.assertFalse(any(row >= 8 for _, rows, _ in calls for row in rows), "DEV/TEST rows reached fitting")
        p, produced = predict(final, prior, self.x, self.splits, seasons)
        self.assertEqual(produced.tolist(), [None] * 3 + ["prior_seasons"] * 5 + ["train_fit"] * 4)
        np.testing.assert_allclose(p[10:], [0.4, 0.6])

    def test_single_train_season_has_no_train_forecasts(self):
        seasons = np.array([2023] * 8 + [2024] * 2 + [2025] * 2)
        (_, _, final, prior), calls = self.run_selection(seasons)
        self.assertEqual((prior, len(calls)), ({}, 2))
        _, produced = predict(final, prior, self.x, self.splits, seasons)
        self.assertEqual(produced.tolist(), [None] * 8 + ["train_fit"] * 4)

    def test_imputer_medians_come_from_fit_rows_only(self):
        config = {"seed": 230, "model_parameters": {"loss": "log_loss", "learning_rate": .05, "min_samples_leaf": 1,
                                                    "subsample": 1., "n_iter_no_change": None}}
        model = new_model(config, {"max_depth": 1, "n_estimators": 2})
        model.fit(np.array([[1., 2.], [3., np.nan], [5., 4.], [7., 6.]]), np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, [4., 4.])

    def test_binary_brier_scale_and_bootstrap_pairing(self):
        y, p = np.array([0, 1, 1, 0]), np.full(4, 0.5)
        self.assertEqual(score_predictions(y, p)["brier"], 0.25)
        result = bootstrap_brier(y, {"gradient_boosting": p, "historical_home_rate": p},
                                 ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], 50, 230, "historical_home_rate")
        self.assertEqual(result["difference_vs_reference"]["gradient_boosting"], [0.0, 0.0])
        self.assertEqual(result["brier_intervals"]["gradient_boosting"], [0.25, 0.25])


class BoxScoreFeatureTests(unittest.TestCase):
    def setUp(self):
        self.history, self.boxes = fixtures()

    def build(self, target=5, history=None, boxes=None):
        history = self.history if history is None else history
        boxes = self.boxes if boxes is None else boxes
        return BoxScoreIndex(history, boxes).for_game(history[target])

    def test_feature_names_windows_and_home_away_difference(self):
        row = self.build()
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertEqual(list(row["features"]), list(BOX_SCORE_FEATURE_NAMES))
        self.assertEqual((len(TEAM_FEATURE_NAMES), len(PLAYER_FEATURE_NAMES), len(TEAM_COVERAGE_FEATURE_NAMES)), (144, 72, 18))
        self.assertEqual(row["coverage"]["home"], {"previous_games": 5, "valid_box_scores": 5, "valid_player_box_scores": 5})
        for name in BOX_SCORE_FEATURE_NAMES:
            if name.startswith("difference_"):
                suffix = name[len("difference_"):]
                self.assertAlmostEqual(row["features"][name], row["features"]["home_" + suffix] - row["features"]["away_" + suffix])

    def test_target_and_future_box_scores_players_and_outcomes_are_never_read(self):
        before = self.build()
        history, boxes = copy.deepcopy(self.history), copy.deepcopy(self.boxes)
        for game in history[5:]:
            game["home_score"], game["away_score"] = 999, 0
        for box in boxes[5:]:
            box["teams"] = {"home": {"players": "corrupt future roster"}, "away": {"pts": float("nan")}}
            box["provenance"] = {"recap": "target-game winner and future stories"}
            box["complete"] = False
        self.assertEqual(before, self.build(history=history, boxes=boxes))

    def test_same_day_and_previous_season_rows_are_excluded(self):
        before = self.build()
        history, boxes = copy.deepcopy(self.history), copy.deepcopy(self.boxes)
        for name, day, season, start in (("same-day", "2024-11-06", 2024, "2024-11-06T15:00:00Z"),
                                         ("old-season", "2024-04-01", 2023, "2024-04-01T23:00:00Z")):
            history.append({**self.history[0], "game_id": name, "game_date_et": day,
                            "season": season, "scheduled_start_utc": start, "home_score": 999})
            boxes.append({"game_id": name, "teams": None})
        self.assertEqual(before, self.build(history=history, boxes=boxes))

    def test_ratios_use_summed_counts_not_average_percentages(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["fga"] = 160
        boxes[0]["teams"]["home"]["players"][0]["fga"] += 80
        row = self.build(boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertAlmostEqual(row["features"]["home_season_effective_fg_pct"], 5 * 45 / (160 + 4 * 80))

    def test_possession_estimate_rating_pace_and_overtime_minutes(self):
        row = self.build()
        estimated = .5 * ((80 - 10 + 12 + .44 * 25) + (90 - 8 + 15 + .44 * 24))
        self.assertAlmostEqual(row["features"]["home_last5_estimated_offensive_rating"], 100 * 110 / estimated)
        self.assertAlmostEqual(row["features"]["home_last5_estimated_defensive_rating"], 100 * 105 / estimated)
        self.assertAlmostEqual(row["features"]["home_last5_estimated_pace"], estimated)
        boxes = copy.deepcopy(self.boxes)
        for box in boxes[:5]:
            for side in ("home", "away"):
                box["teams"][side]["game_minutes"] = 53
                box["teams"][side]["minutes"] = 264 if side == "home" else 266
        overtime = self.build(boxes=boxes)
        self.assertTrue(overtime["eligible"], overtime["reasons"])
        self.assertAlmostEqual(overtime["features"]["home_last5_estimated_pace"], estimated * 48 / 53)

    def test_last_five_and_ten_are_actual_chronological_windows(self):
        boxes = copy.deepcopy(self.boxes)
        for box in boxes[:2]:
            box["teams"]["home"]["fga"] = 160
        row = self.build(target=12, boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertAlmostEqual(row["features"]["home_last5_effective_fg_pct"], 45 / 80)
        self.assertAlmostEqual(row["features"]["home_last10_effective_fg_pct"], 45 / 80)
        self.assertNotAlmostEqual(row["features"]["home_season_effective_fg_pct"], 45 / 80)

    def test_team_turnovers_need_not_equal_sum_of_player_turnovers(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["tov"] += 7
        row = self.build(boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertEqual(row["features"]["home_last5_player_boxscore_missing"], 0)

    def test_missing_prior_box_reduces_coverage_without_backfilling_older_games(self):
        boxes = [r for r in copy.deepcopy(self.boxes) if r["game_id"] != "g10"]
        boxes[6]["teams"]["home"]["fga"] = 160  # g7: just outside the actual last five (g8-g12)
        row = self.build(target=12, boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertEqual(row["coverage"]["home"], {"previous_games": 12, "valid_box_scores": 11, "valid_player_box_scores": 10})
        self.assertEqual([m["game_id"] for m in row["missing_box_scores"]["home"]], ["g10"])
        features = row["features"]
        self.assertAlmostEqual(features["home_season_team_boxscore_coverage"], 11 / 12)
        self.assertAlmostEqual(features["home_last5_team_boxscore_coverage"], .8)
        self.assertAlmostEqual(features["home_last10_team_boxscore_coverage"], .9)
        self.assertEqual(features["home_last5_team_boxscore_missing"], 1)
        self.assertEqual(features["difference_last5_team_boxscore_coverage"], 0)
        self.assertAlmostEqual(features["home_last5_effective_fg_pct"], 45 / 80)
        self.assertTrue(math.isnan(features["home_last5_max_player_minutes_per_game"]))
        self.assertAlmostEqual(features["home_last5_player_boxscore_coverage"], .8)

    def test_window_without_valid_boxes_is_missing_and_too_few_boxes_exclude(self):
        boxes = [r for r in self.boxes if r["game_id"] not in {f"g{i}" for i in range(8, 13)}]
        row = self.build(target=12, boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertTrue(math.isnan(row["features"]["home_last5_estimated_net_rating"]))
        self.assertEqual(row["features"]["home_last5_team_boxscore_coverage"], 0)
        self.assertTrue(math.isfinite(row["features"]["home_season_estimated_net_rating"]))
        short = self.build(boxes=[r for r in self.boxes if r["game_id"] != "g3"])
        self.assertFalse(short["eligible"])
        self.assertEqual(short["reason"], "fewer_than_min_box_scores")
        self.assertEqual(self.build(target=3)["reason"], "fewer_than_min_previous_games")

    def test_invalid_or_late_records_are_treated_as_missing(self):
        boxes = [r for r in self.boxes if r["game_id"] != "g1"]
        missing = self.build(target=12, boxes=boxes)
        mutations = (
            lambda r: r.update(season=2023),
            lambda r: r["teams"]["home"].update(team_abbr="LAL"),
            lambda r: r["teams"]["home"].update(pts=111),
            lambda r: r["teams"]["home"].update(fga=float("nan")),
            lambda r: r["teams"]["home"].update(fga=80.5),
            lambda r: r["teams"]["home"].update(fgm=100),
            lambda r: r.update(game_minutes=53),
            lambda r: r.update(stat_available_at_utc="2024-11-13T22:00:01Z"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                boxes = copy.deepcopy(self.boxes)
                mutate(boxes[0])
                row = self.build(target=12, boxes=boxes)
                self.assertTrue(row["eligible"], row["reasons"])
                self.assertEqual(row["features"].keys(), missing["features"].keys())
                for name, value in row["features"].items():
                    self.assertTrue(value == missing["features"][name] or (math.isnan(value) and math.isnan(missing["features"][name])), name)
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["provenance"]["retrieved_at_utc"] = "2026-11-03T00:00:00Z"
        boxes[0]["stat_available_at_utc"] = "2024-11-13T22:00:00Z"
        self.assertEqual(self.build(target=12, boxes=boxes)["coverage"]["home"]["valid_box_scores"], 12)

    def test_player_missingness_marks_statistics_unknown(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["players_complete"] = False
        row = self.build(boxes=boxes)
        self.assertTrue(row["eligible"], row["reasons"])
        self.assertTrue(math.isnan(row["features"]["home_last5_max_player_minutes_per_game"]))
        self.assertTrue(math.isnan(row["features"]["difference_last5_max_player_minutes_per_game"]))
        self.assertTrue(math.isfinite(row["features"]["away_last5_max_player_minutes_per_game"]))
        self.assertAlmostEqual(row["features"]["home_last5_player_boxscore_coverage"], .8)
        self.assertEqual(row["features"]["home_last5_player_boxscore_missing"], 1)
        self.assertEqual(row["features"]["home_last5_team_boxscore_coverage"], 1)

    def test_missing_player_game_does_not_poison_later_windows(self):
        boxes = copy.deepcopy(self.boxes)
        boxes[0]["teams"]["home"]["players_complete"] = False
        row = self.build(target=12, boxes=boxes)
        self.assertTrue(math.isnan(row["features"]["home_season_max_player_minutes_per_game"]))
        self.assertTrue(math.isfinite(row["features"]["home_last5_max_player_minutes_per_game"]))
        self.assertTrue(math.isfinite(row["features"]["home_last10_max_player_minutes_per_game"]))
        boxes[9]["teams"]["home"]["players_complete"] = False
        interior = self.build(target=12, boxes=boxes)
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

    def test_corrupt_player_rows_are_flagged_without_fabricated_values(self):
        for updates in ({"player_id": None}, {"player_id": "BOS-p1"}, {"minutes": None}, {"fga": 0}):
            with self.subTest(updates=updates):
                boxes = copy.deepcopy(self.boxes)
                boxes[0]["teams"]["home"]["players"][0].update(updates)
                row = self.build(boxes=boxes)
                self.assertTrue(row["eligible"], row["reasons"])
                self.assertTrue(math.isnan(row["features"]["home_last5_max_player_minutes_per_game"]))
                self.assertEqual(row["features"]["home_last5_player_boxscore_missing"], 1)


if __name__ == "__main__":
    unittest.main()
