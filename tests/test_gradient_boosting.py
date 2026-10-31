"""Guard the statistics baseline's temporal boundaries and probability scoring."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_gradient_boosting import FEATURE_NAMES, arrays, bootstrap_brier, prepare_examples, score_predictions, select_candidate


class GradientBoostingTests(unittest.TestCase):
    def setUp(self):
        self.config = {"seed": 230, "min_previous_games": 2, "cutoff_minutes": 60,
                       "split_seasons": {"train": 2023, "validation": 2024, "test": 2025},
                       "selection_metric": "brier", "model_parameters": {"n_iter_no_change": None},
                       "candidates": [{"max_depth": 1}, {"max_depth": 2}]}
        self.history = [{"game_id": f"game-{i}", "season": 2024, "home_abbr": "BOS", "away_abbr": "NYK",
                         "scheduled_start_utc": f"2025-01-{i:02}T23:00:00Z", "game_date_et": f"2025-01-{i:02}",
                         "home_score": 110 if i % 2 else 90, "away_score": 100}
                        for i in range(1, 6)]

    def test_outcomes_at_or_after_target_cannot_change_target_features(self):
        original, labels, _ = prepare_examples(self.history, self.config)
        changed = copy.deepcopy(self.history)
        for game in changed[2:]:
            game["home_score"] = 1
            game["recap"] = "The away team won."
        rebuilt, new_labels, _ = prepare_examples(changed, self.config)
        self.assertEqual(original[0], rebuilt[0])
        self.assertNotEqual(labels[0], new_labels[0])
        self.assertEqual(original[0]["features"]["home_season_win_rate"], 0.5)
        self.assertEqual(original[0]["features"]["home_last_5_point_differential"], 0.0)
        self.assertEqual(original[0]["prior_game_ids"]["home"], ["game-1", "game-2"])
        self.assertEqual(set(original[0]["features"]), set(FEATURE_NAMES))
        self.assertFalse({"home_win", "home_score", "away_score", "recap"} & set(original[0]))

    def test_same_day_outcome_is_excluded_from_target(self):
        same_day = dict(self.history[2], game_id="early-same-day", scheduled_start_utc="2025-01-03T15:00:00Z", home_score=999)
        original, _, _ = prepare_examples(self.history, self.config)
        with_same_day, _, _ = prepare_examples(self.history + [same_day], self.config)
        target = next(r for r in with_same_day if r["game_id"] == "game-3")
        self.assertEqual(original[0]["features"], target["features"])
        self.assertNotIn("early-same-day", target["prior_game_ids"]["home"])

    def test_warmup_exclusion_and_reversed_split_rejection(self):
        inputs, _, excluded = prepare_examples(self.history, self.config)
        self.assertEqual([r["game_id"] for r in excluded], ["game-1", "game-2"])
        self.assertEqual(len(inputs), 3)
        self.config["split_seasons"] = {"train": 2025, "validation": 2024, "test": 2023}
        with self.assertRaises(ValueError):
            prepare_examples(self.history, self.config)

    def test_misaligned_outcomes_are_rejected(self):
        inputs, labels, _ = prepare_examples(self.history, self.config)
        with self.assertRaisesRegex(ValueError, "align"):
            arrays(inputs, labels[::-1])

    def test_selection_fits_only_training_rows_and_uses_validation_brier(self):
        calls = []

        class FakeModel:
            def __init__(self, candidate):
                self.candidate = candidate

            def fit(self, x, y):
                calls.append((x.copy(), y.copy()))
                return self

            def predict_proba(self, x):
                self.assertion = np.testing.assert_array_equal(x, [[10.0], [11.0]])
                p = np.array([0.4, 0.6]) if self.candidate["max_depth"] == 1 else np.array([0.8, 0.2])
                return np.column_stack([1 - p, p])

        train_x, train_y = np.array([[1.0], [2.0]]), np.array([0, 1])
        with patch("run_gradient_boosting.new_model", side_effect=lambda config, candidate: FakeModel(candidate)):
            chosen, scores = select_candidate(train_x, train_y, np.array([[10.0], [11.0]]), np.array([0, 1]), self.config)
        self.assertEqual(chosen, {"max_depth": 1})
        self.assertAlmostEqual(scores[0]["brier"], 0.16)
        for x, y in calls:
            np.testing.assert_array_equal(x, train_x)
            np.testing.assert_array_equal(y, train_y)

    def test_binary_brier_scale_and_bootstrap_pairing(self):
        y = np.array([0, 1, 1, 0])
        p = np.full(4, 0.5)
        self.assertEqual(score_predictions(y, p)["brier"], 0.25)
        predictions = {name: p for name in ("gradient_boosting", "historical_home_rate", "constant_50_percent")}
        result = bootstrap_brier(y, predictions, ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], 50, 230)
        self.assertEqual(result["gb_minus_home_rate_interval"], [0.0, 0.0])
        self.assertEqual(result["brier_intervals"]["gradient_boosting"], [0.25, 0.25])


if __name__ == "__main__":
    unittest.main()
