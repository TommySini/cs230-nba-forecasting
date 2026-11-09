"""Orthogonality evaluation: joins outcomes from labels.jsonl, keeps splits apart, rejects post-cutoff text."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_injury_orthogonality as injury
import evaluate_notes_orthogonality as notes
from evaluate_orthogonality import boosting_forecasts, day_folds, logit, sigmoid, statistics


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def html_note(game_id, team, published, cutoff="2025-01-15T00:00:00Z", **extra):
    return {"game_id": game_id, "team": team, "kind": "html_post", "tier": "site_timestamped",
            "published_utc": published, "cutoff_utc": cutoff, "text_path": f"texts/{game_id}-{team}.txt", "source": {}, **extra}


class BoostingForecastTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        write_jsonl(self.dir / "predictions.jsonl", [
            {"game_id": "g1", "season": 2013, "game_date_et": "2014-01-01", "split": "train", "p_boost": 0.6, "produced_by": "prior_seasons"},
            {"game_id": "g2", "season": 2024, "game_date_et": "2025-01-01", "split": "dev", "p_boost": 0.4, "produced_by": "train_fit"},
            {"game_id": "g3", "season": 2025, "game_date_et": "2026-01-01", "split": "test", "p_boost": 0.7, "produced_by": "train_fit"}])
        write_jsonl(self.dir / "labels.jsonl", [{"game_id": g, "home_win": w} for g, w in (("g1", 1), ("g2", 0), ("g3", 1))])

    def test_outcomes_joined_from_labels_and_splits_filtered(self):
        forecasts = boosting_forecasts(["train", "dev"], self.dir)
        self.assertEqual(sorted(forecasts), ["g1", "g2"])
        self.assertEqual(forecasts["g2"], {"split": "dev", "game_date_et": "2025-01-01", "home_win": 0, "p_boost": 0.4})

    def test_unknown_split_and_missing_label_rejected(self):
        with self.assertRaises(ValueError):
            boosting_forecasts(["validation"], self.dir)
        write_jsonl(self.dir / "labels.jsonl", [{"game_id": "g1", "home_win": 1}])
        with self.assertRaises(ValueError):
            boosting_forecasts(["dev"], self.dir)


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
    def test_revised_posts_dropped_and_late_documents_rejected(self):
        kept, dropped = notes.eligible_documents([
            html_note("g1", "NYK", "2025-01-14T20:00:00Z"),
            html_note("g1", "POR", "2025-01-14T20:00:00Z", modified_after_cutoff=True)])
        self.assertEqual([d["team"] for d in kept], ["NYK"])
        self.assertEqual(dropped["html_modified_after_cutoff"], 1)
        with self.assertRaises(ValueError):
            notes.eligible_documents([html_note("g1", "NYK", "2025-01-14T20:00:00Z", modified_utc="2025-01-15T01:00:00Z")])

    def test_one_row_per_game_with_both_teams_text(self):
        root = Path(tempfile.mkdtemp())
        (root / "texts").mkdir()
        docs = [html_note("g1", "NYK", "2025-01-14T20:00:00Z"), html_note("g1", "POR", "2025-01-14T21:00:00Z"),
                html_note("g9", "NYK", "2025-01-14T20:00:00Z")]
        for d in docs:
            (root / d["text_path"]).write_text(f"notes from {d['team']}", encoding="utf-8")
        write_jsonl(root / "approved.jsonl", docs)
        forecasts = {"g1": {"split": "dev", "game_date_et": "2025-01-14", "home_win": 1, "p_boost": 0.55}}
        rows, _ = notes.load(["dev"], root / "approved.jsonl", root, forecasts)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["publishers"], ["NYK", "POR"])
        self.assertIn("notes from NYK", rows[0]["text"])
        self.assertIn("notes from POR", rows[0]["text"])

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
