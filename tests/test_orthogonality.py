"""Orthogonality evaluation: joins outcomes from labels.jsonl, keeps splits apart, rejects post-cutoff text."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_injury_orthogonality as injury
import evaluate_notes_orthogonality as notes
from evaluate_orthogonality import day_folds, join_forecasts, logit, sigmoid, statistics


PREDICTIONS = [
    {"game_id": "g1", "season": 2013, "game_date_et": "2014-01-01", "split": "train", "p_boost": 0.6, "produced_by": "prior_seasons"},
    {"game_id": "g2", "season": 2024, "game_date_et": "2025-01-01", "split": "dev", "p_boost": 0.4, "produced_by": "train_fit"},
    {"game_id": "g3", "season": 2025, "game_date_et": "2026-01-01", "split": "test", "p_boost": 0.7, "produced_by": "train_fit"}]


class BoostingForecastTests(unittest.TestCase):
    def test_outcomes_joined_from_labels_and_splits_filtered(self):
        labels = [{"game_id": "g1", "home_win": 1}, {"game_id": "g2", "home_win": 0}]
        forecasts = join_forecasts(PREDICTIONS, labels, ["train", "dev"])
        self.assertEqual(sorted(forecasts), ["g1", "g2"])
        self.assertEqual(forecasts["g2"], {"split": "dev", "game_date_et": "2025-01-01", "home_win": 0, "p_boost": 0.4})

    def test_unknown_split_and_missing_label_rejected(self):
        with self.assertRaises(ValueError):
            join_forecasts(PREDICTIONS, [], ["validation"])
        with self.assertRaises(ValueError):
            join_forecasts(PREDICTIONS, [{"game_id": "g1", "home_win": 1}], ["dev"])


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 3000
        self.base_logit = rng.normal(0, 1, self.n)
        self.signal = rng.normal(0, 1, self.n)
        self.y = (rng.random(self.n) < sigmoid(self.base_logit + self.signal)).astype(int)
        self.base = sigmoid(self.base_logit)
        self.dates = np.array([f"d{i // 10}" for i in range(self.n)])

    def test_informative_forecast_detected(self):
        out = statistics(self.base, sigmoid(self.base_logit + self.signal), self.y, day_folds(self.dates, 5, 1))
        self.assertGreater(out["delta_coef"], 0.5)
        self.assertLess(out["lr_p"], 1e-6)
        self.assertGreater(out["oof_brier_gain"], 0)

    def test_noise_forecast_adds_nothing(self):
        noise = np.random.default_rng(1).normal(0, 1, self.n)
        out = statistics(self.base, sigmoid(self.base_logit + noise), self.y, day_folds(self.dates, 5, 1))
        self.assertLess(abs(out["delta_coef"]), 0.15)
        self.assertLess(out["oof_brier_gain"], 0.002)

    def test_day_folds_keep_each_date_together(self):
        folds = day_folds(self.dates, 5, 1)
        for d in np.unique(self.dates):
            self.assertEqual(len(np.unique(folds[self.dates == d])), 1)
        self.assertTrue(np.allclose(logit(sigmoid(np.array([-2.0, 0.0, 2.0]))), [-2, 0, 2]))


class NotesTests(unittest.TestCase):
    FORECASTS = {"g1": {"split": "dev", "game_date_et": "2025-01-14", "home_win": 1, "p_boost": 0.55}}

    def game(self, *sources, game_id="g1"):
        return {"game_id": game_id, "cutoff_utc": "2025-01-15T00:00:00Z",
                "sources": [{"sha256": sha, "publisher": team, "archive_time": time} for sha, team, time in sources]}

    def test_one_row_per_game_with_every_document_text(self):
        cohort = [self.game(("a", "NYK", "2025-01-14T20:00:00Z"), ("b", "POR", "2025-01-14T21:00:00Z")),
                  self.game(("c", "NYK", "2025-01-14T20:00:00Z"), game_id="g9")]
        rows = notes.game_rows(cohort, self.FORECASTS, {"a": "notes from NYK", "b": "notes from POR", "c": "x"})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["publishers"], ["NYK", "POR"])
        self.assertEqual(rows[0]["text"], "notes from NYK\n\nnotes from POR")
        self.assertEqual((rows[0]["y"], rows[0]["base"], rows[0]["split"]), (1, 0.55, "dev"))

    def test_document_public_after_cutoff_rejected(self):
        with self.assertRaises(ValueError):
            notes.game_rows([self.game(("a", "NYK", "2025-01-15T00:00:01Z"))], self.FORECASTS, {"a": "late"})

    def test_text_correlated_with_residual_is_found_and_permutation_p_small(self):
        rng = np.random.default_rng(3)
        rows = []
        for i in range(300):
            good = i % 2 == 0
            y = int(rng.random() < (0.8 if good else 0.2))
            words = " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "omega"], 20))
            rows.append({"game_id": f"g{i}", "date": f"2025-01-{i % 28 + 1:02d}", "split": "dev", "y": y,
                         "base": float(rng.uniform(0.35, 0.65)),
                         "publishers": ["NYK"], "text": words + (" healthy rested" if good else " injured tired")})
        args = SimpleNamespace(folds=5, seed=1, replicates=50, permutations=20, splits=["dev"])
        point = notes.analyse(rows, args)
        self.assertGreater(point["delta_residual"], 0.3)
        self.assertLess(point["permutation_p_one_sided"], 0.1)


class InjuryTests(unittest.TestCase):
    ROW = {"home_team": "Cleveland Cavaliers", "away_team": "Toronto Raptors", "cutoff_utc": "2025-10-31T22:30:00Z",
           "injury_report": {"report_time_utc": "2025-10-31T22:00:00Z",
                             "text": "Toronto Raptors Lawson, A.J. Out G League - Two- Way Poeltl, Jakob Out Injury/Illness - Back "
                                     "Cleveland Cavaliers Allen, Jarrett Out Injury/Illness - Hand Garland, Darius Questionable Injury/Illness - Toe"}}

    def test_burden_ignores_assignments_and_favours_healthier_home(self):
        self.assertEqual(injury.feature(self.ROW), 1.0 - 1.5)

    def test_report_after_cutoff_rejected(self):
        row = dict(self.ROW, game_id="g1", split="dev",
                   injury_report=dict(self.ROW["injury_report"], report_time_utc="2025-10-31T23:00:00Z"))
        forecasts = {"g1": {"split": "dev", "game_date_et": "2025-10-31", "home_win": 1, "p_boost": 0.6}}
        with self.assertRaises(ValueError):
            injury.load("dev", forecasts, [row])


if __name__ == "__main__":
    unittest.main()
