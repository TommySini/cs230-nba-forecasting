"""Selection and imputation must not learn from held-out inputs or labels."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rich_gradient_boosting import choose_on_dev, numeric_features, paired_intervals


class RichGradientBoostingTests(unittest.TestCase):
    def config(self):
        return {"seed": 230, "model_parameters": {"loss": "log_loss", "learning_rate": .05,
                "min_samples_leaf": 1, "subsample": 1., "n_iter_no_change": None},
                "candidates": [{"max_depth": 1, "n_estimators": 2}, {"max_depth": 2, "n_estimators": 2}]}

    def test_dev_selection_imputer_fits_only_train(self):
        train_x = np.array([[1., 2.], [3., np.nan], [5., 4.], [7., 6.]])
        train_y = np.array([0, 0, 1, 1])
        dev_x = np.array([[1e9, np.nan], [-1e9, 1e9]])
        _, _, fitted = choose_on_dev(train_x, train_y, dev_x, np.array([0, 1]), self.config())
        for model, _ in fitted:
            np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, [4., 4.])
    def test_selection_never_fits_validation_rows(self):
        seen = []
        class Fake:
            def fit(self, x, y):
                seen.append((x.copy(), y.copy()))
                return self
            def predict_proba(self, x):
                np.testing.assert_array_equal(x, [[9.], [10.]])
                return np.array([[.8, .2], [.2, .8]])
        with patch("run_rich_gradient_boosting.model_for", return_value=Fake()):
            selected, _, _ = choose_on_dev(np.array([[1.], [2.]]), np.array([0, 1]),
                                           np.array([[9.], [10.]]), np.array([0, 1]), self.config())
        self.assertEqual(selected, 0)
        for x, y in seen:
            np.testing.assert_array_equal(x, [[1.], [2.]])
            np.testing.assert_array_equal(y, [0, 1])

    def test_unknown_player_value_serializes_null_and_team_missing_is_rejected(self):
        self.assertEqual(numeric_features({"team": 2, "player": float("nan")}, {"player"}),
                         {"team": 2., "player": None})
        with self.assertRaisesRegex(ValueError, "mandatory"):
            numeric_features({"team": float("nan")}, {"player"})
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            numeric_features({"player": float("inf")}, {"player"})

    def test_identical_paired_forecasts_have_zero_difference_interval(self):
        predictions = {"original_fixed": np.array([.3, .6, .7]), "team_selected": np.array([.3, .6, .7])}
        result = paired_intervals(np.array([0, 1, 1]), predictions, ["a", "a", "b"], 100, 230)
        self.assertEqual(result["brier_difference_vs_original"]["team_selected"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
