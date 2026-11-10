"""Synthetic leakage and integrity checks; never download weights or use NBA outcomes."""
import copy
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_gradient_boosting import FEATURE_NAMES, bootstrap_brier, sha256
import validate_tabpfn as validator
import run_tabpfn as runner


def fixture():
    inputs = []
    for i, (split, date) in enumerate((("train", "2023-11-01"), ("train", "2023-11-02"),
                                     ("dev", "2024-11-01"), ("dev", "2024-11-02"))):
        inputs.append({"game_id": f"g{i}", "split": split, "season": int(date[:4]),
                       "game_date_et": date, "scheduled_start_utc": date + "T23:00:00Z",
                       "cutoff_utc": date + "T22:00:00Z",
                       "features": {name: float(i) for name in FEATURE_NAMES},
                       "prior_game_ids": {"home": [], "away": []}})
    labels = [{"game_id": row["game_id"], "home_win": i % 2} for i, row in enumerate(inputs)]
    manifest = {"stage": "dev", "feature_names": list(FEATURE_NAMES),
                "fit_game_ids": ["g0", "g1"], "query_game_ids": ["g2", "g3"]}
    rows = [{"game_id": row["game_id"], "game_date_et": row["game_date_et"],
             "tabpfn": probability, "rich_gradient_boosting": reference}
            for row, probability, reference in zip(inputs[2:], [.3, .7], [.4, .6])]
    return manifest, inputs, labels, rows


class PredictionValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.inputs, self.labels, self.rows = fixture()

    def validate(self):
        return validator.validate_prediction_rows(self.manifest, self.inputs, self.labels, self.rows)

    def test_same_game_comparison_recomputes_metrics(self):
        y, predictions, metrics = self.validate()
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(predictions["tabpfn"], [.3, .7])
        self.assertAlmostEqual(metrics["tabpfn"]["brier"], .09)
        self.assertAlmostEqual(metrics["rich_gradient_boosting"]["brier"], .16)

    def test_feature_order_is_bound_to_current_baseline(self):
        self.manifest["feature_names"].reverse()
        with self.assertRaisesRegex(ValueError, "Feature order"):
            self.validate()

    def test_extra_feature_and_inline_outcome_rejected(self):
        self.inputs[0]["features"]["current_game_final_score"] = 100
        with self.assertRaisesRegex(ValueError, "feature schema"):
            self.validate()
        del self.inputs[0]["features"]["current_game_final_score"]
        self.inputs[0]["home_win"] = 1
        with self.assertRaisesRegex(ValueError, "separate"):
            self.validate()

    def test_outcomes_must_be_separate_unique_aligned_binary_records(self):
        original = copy.deepcopy(self.labels)
        mutations = [list(reversed(original)), original[:-1], original + [original[0]],
                     [dict(original[0], home_win=True)] + original[1:],
                     [dict(original[0], home_win=.5)] + original[1:],
                     [dict(original[0], result="postgame text")] + original[1:]]
        for labels in mutations:
            with self.subTest(labels=labels):
                self.labels = labels
                with self.assertRaises(ValueError):
                    self.validate()

    def test_fit_query_overlap_and_missing_membership_rejected(self):
        for fit_ids in (["g0", "g2"], ["g0"], ["g0", "g1", "g1"]):
            with self.subTest(fit_ids=fit_ids):
                self.manifest["fit_game_ids"] = fit_ids
                with self.assertRaisesRegex(ValueError, "distinct"):
                    self.validate()

    def test_no_test_rows_or_stages(self):
        for side in (0, 2):
            old = self.inputs[side]["split"]
            self.inputs[side]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "TEST is excluded"):
                self.validate()
            self.inputs[side]["split"] = old
        self.manifest["stage"] = "test"
        with self.assertRaisesRegex(ValueError, "Only TRAIN"):
            self.validate()

    def test_train_smoke_does_not_admit_dev_rows(self):
        self.manifest["stage"] = "train-smoke"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            self.validate()

    def test_same_day_or_future_fit_outcomes_rejected(self):
        for date in ("2024-11-01", "2025-11-01"):
            with self.subTest(date=date):
                self.inputs[1]["game_date_et"] = date
                with self.assertRaisesRegex(ValueError, "earlier calendar dates"):
                    self.validate()

    def test_predictions_must_match_query_order_and_dates(self):
        self.rows.reverse()
        with self.assertRaisesRegex(ValueError, "membership/order"):
            self.validate()
        self.rows.reverse()
        self.rows[0]["game_date_et"] = "2024-12-01"
        with self.assertRaisesRegex(ValueError, "dates"):
            self.validate()

    def test_predictions_cannot_contain_outcomes(self):
        self.rows[0]["home_win"] = 0
        with self.assertRaisesRegex(ValueError, "only metadata"):
            self.validate()

    def test_both_models_require_finite_probabilities_in_range(self):
        for name in ("tabpfn", "rich_gradient_boosting"):
            for value in (float("nan"), float("inf"), -0.01, 1.01):
                with self.subTest(name=name, value=value):
                    old = self.rows[0][name]
                    self.rows[0][name] = value
                    with self.assertRaisesRegex(ValueError, "finite probability"):
                        self.validate()
                    self.rows[0][name] = old

    def test_identical_forecasts_have_zero_paired_difference_interval(self):
        y, predictions, _ = self.validate()
        predictions["tabpfn"] = predictions["rich_gradient_boosting"].copy()
        result = bootstrap_brier(y, predictions, [row["game_date_et"] for row in self.rows],
                                 100, 230, "rich_gradient_boosting")
        self.assertEqual(result["difference_vs_reference"]["tabpfn"], [0., 0.])


class SavedRunValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "tabpfn" / "dev"
        self.output.mkdir(parents=True)
        self.config = {"output_dir": "tabpfn", "seed": 230, "bootstrap_replicates": 100,
                       "model": {"package_version": "synthetic", "sha256": "synthetic-checkpoint"}}
        self.manifest, self.inputs, self.labels, self.rows = fixture()
        (self.output / "manifest.json").write_text(json.dumps(self.manifest))
        with (self.output / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        y, predictions, metrics = validator.validate_prediction_rows(
            self.manifest, self.inputs, self.labels, self.rows)
        uncertainty = bootstrap_brier(y, predictions, [row["game_date_et"] for row in self.rows],
                                      100, 230, "rich_gradient_boosting")
        self.run = {
            "schema_version": 2, "stage": "dev", "status": "complete", "model": self.config["model"],
            "versions": {"tabpfn": "synthetic"},
            "manifest_sha256": sha256(self.output / "manifest.json"),
            "predictions_sha256": runner.prediction_digest(self.rows),
            "metrics": metrics, "uncertainty": uncertainty,
            "primary_comparison": {
                "brier_difference": metrics["tabpfn"]["brier"] - metrics["rich_gradient_boosting"]["brier"],
                "difference_95_interval": uncertainty["difference_vs_reference"]["tabpfn"],
            },
        }
        self.save_run()

    def save_run(self):
        (self.output / "run.json").write_text(json.dumps(self.run))

    def validate(self):
        with patch.object(validator, "ROOT", self.root), patch(
                "run_tabpfn.load_prepared", return_value=(self.manifest, self.inputs, self.labels)), patch(
                "run_tabpfn.record_directory", return_value=self.output), patch(
                "run_tabpfn.load_comparison_rows", side_effect=lambda *args: self.rows):
            return validator.validate_run(self.config, "dev")

    def test_saved_run_metrics_reproduce_without_loading_model(self):
        result = self.validate()
        self.assertEqual(result["query_games"], 2)
        self.assertAlmostEqual(result["primary_comparison"]["brier_difference"], -.07)

    def test_changed_prediction_bytes_fail_before_scoring(self):
        self.rows[0]["tabpfn"] = 0.1
        with self.assertRaisesRegex(ValueError, "Changed run artifact"):
            self.validate()

    def test_changed_prepared_manifest_is_rejected(self):
        with (self.output / "manifest.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.validate()

    def test_missing_prediction_hash_is_rejected(self):
        self.run.pop("predictions_sha256")
        self.save_run()
        with self.assertRaisesRegex(ValueError, "bind the prediction"):
            self.validate()

    def test_invented_scores_and_intervals_are_rejected(self):
        for field, replacement in (("metrics", {}), ("uncertainty", {}), ("primary_comparison", {})):
            with self.subTest(field=field):
                old = self.run[field]
                self.run[field] = replacement
                self.save_run()
                with self.assertRaises(ValueError):
                    self.validate()
                self.run[field] = old

    def test_wrong_stage_status_or_model_pin_is_rejected(self):
        for field, value in (("stage", "test"), ("status", "running"), ("model", {}),
                             ("versions", {"tabpfn": "different"})):
            with self.subTest(field=field):
                old = self.run[field]
                self.run[field] = value
                self.save_run()
                with self.assertRaises(ValueError):
                    self.validate()
                self.run[field] = old


class RunnerBoundaryTests(unittest.TestCase):
    def test_interrupted_inference_resumes_without_fitting_query_labels_or_rewriting_completed_run(self):
        manifest, inputs, labels, _ = fixture()
        manifest["stage"] = "train-smoke"
        manifest["baseline_config"] = {"candidates": [{}]}
        manifest["baseline_comparison"] = "synthetic"
        for row in inputs:
            row["split"] = "train"
        fit_calls, query_calls = [], []

        class FakeTabPFN:
            classes_ = np.array([0, 1])
            calls = 0
            audit_override = {}
            def fit(self, x, y):
                fit_calls.append((x.copy(), y.copy()))
                self.n_train_samples_ = self.audit_override.get("rows", len(x))
                self.inference_config_ = SimpleNamespace(SUBSAMPLE_SAMPLES=self.audit_override.get("limit"))
                self.ensemble_preprocessor_ = SimpleNamespace(subsample_row_indices=self.audit_override.get("indices"))
                return self
            def predict_proba(self, x):
                self.calls += 1
                query_calls.append(x.copy())
                if self.calls == 2:
                    raise RuntimeError("synthetic interruption")
                return np.tile([.3, .7], (len(x), 1))

        class FakeBooster:
            classes_ = np.array([0, 1])
            def fit(self, x, y):
                fit_calls.append((x.copy(), y.copy()))
                return self
            def predict_proba(self, x):
                return np.tile([.4, .6], (len(x), 1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "tabpfn" / "train-smoke"
            output.mkdir(parents=True)
            (output / "manifest.json").write_text(json.dumps(manifest))
            config = {"output_dir": "tabpfn", "compute_platform": "local", "seed": 230,
                      "query_batch_size": 1, "bootstrap_replicates": 100, "max_runtime_seconds": 100,
                      "model": {"device": "cpu", "max_context_rows": 100, "package_version": "synthetic"}}
            fake = FakeTabPFN()
            with patch.object(runner, "ROOT", root), patch.object(
                    runner, "load_prepared", return_value=(manifest, inputs, labels)), patch.object(
                    runner, "model_for", return_value=fake), patch.object(
                    runner, "new_model", return_value=FakeBooster()), patch.object(
                    runner, "version", return_value="synthetic"), patch.object(runner, "write_report"), patch(
                    "nba_dataset.store_predictions") as store:
                for override in ({"rows": 1}, {"limit": 1}, {"indices": [0]}):
                    with self.subTest(context_audit=override):
                        fake.audit_override = override
                        with self.assertRaisesRegex(ValueError, "silently subsampled"):
                            runner.fit_and_predict(config, "train-smoke")
                        self.assertFalse((output / "prediction_progress.json").exists())
                        self.assertEqual(query_calls, [])
                fake.audit_override = {}
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    runner.fit_and_predict(config, "train-smoke")
                self.assertFalse((output / "run.json").exists())
                progress = json.loads((output / "prediction_progress.json").read_text())
                self.assertEqual([row["game_id"] for row in progress["predictions"]], ["g2"])
                result = runner.fit_and_predict(config, "train-smoke")
                self.assertEqual(result["status"], "complete")
                with self.assertRaisesRegex(ValueError, "Predictions already exist"):
                    runner.fit_and_predict(config, "train-smoke")
            for x, y in fit_calls:
                np.testing.assert_array_equal(x[:, 0], [0., 1.])
                np.testing.assert_array_equal(y, [0, 1])
            self.assertEqual([x[:, 0].tolist() for x in query_calls], [[2.], [3.], [3.]])

    def test_excluded_split_label_partitions_are_never_opened(self):
        class FakeDataset:
            def inputs(self, splits):
                self.input_splits = set(splits)
                return [{"game_id": "train", "split": "train"}]
            def labels(self, splits):
                self.label_splits = set(splits)
                if self.label_splits != {"train"}:
                    raise AssertionError("Forbidden labels opened")
                return [{"game_id": "train", "home_win": 1}]
        dataset = FakeDataset()
        inputs, labels = runner.read_source_rows(dataset, {"train"})
        self.assertEqual(dataset.label_splits, {"train"})
        self.assertEqual(labels, {"train": {"game_id": "train", "home_win": 1}})
        with self.assertRaisesRegex(ValueError, "TEST is excluded"):
            runner.read_source_rows(dataset, {"test"})

    def test_source_alignment_and_duplicate_ids_are_checked(self):
        from unittest.mock import Mock
        dataset = Mock()
        dataset.inputs.return_value = [{"game_id": "a", "split": "train"}]
        for labels in ([{"game_id": "b", "home_win": 1}], [],
                       [{"game_id": "a", "home_win": 1}] * 2):
            dataset.labels.return_value = labels
            with self.assertRaises(ValueError):
                runner.read_source_rows(dataset, {"train"})

    def test_prepared_manifest_cannot_subsample_or_reorder_cohort(self):
        from unittest.mock import Mock
        manifest, inputs, labels, _ = fixture()
        config = {"smoke_games": 32, "smoke_context_games": 16, "model": {}}
        manifest.update(schema_version=2, config=config, dataset_id="fixed",
                        implementation_sha256=runner.sha256(Path(runner.__file__)),
                        baseline_config={"cutoff_minutes": 60, "split_seasons": {
                            "train": [2023, 2023], "dev": [2024, 2024], "test": [2025, 2025]}})
        dataset = Mock()
        dataset.manifest = {"dataset_id": "fixed"}
        dataset.inputs.return_value = inputs
        dataset.labels.return_value = labels
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(runner, "shared_dataset", return_value=dataset), patch.object(
                    runner, "record_directory", return_value=output):
                (output / "manifest.json").write_text(json.dumps(manifest))
                runner.load_prepared(config, "dev")
                for key, ids in (("fit_game_ids", ["g0"]), ("query_game_ids", ["g2"]),
                                 ("fit_game_ids", ["g1", "g0"]), ("query_game_ids", ["g3", "g2"])):
                    changed = {**manifest, key: ids}
                    (output / "manifest.json").write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "complete fixed cohort"):
                        runner.load_prepared(config, "dev")

    def test_inference_rejects_staged_dataset_before_writing(self):
        with self.assertRaisesRegex(ValueError, "validation-only"):
            runner.fit_and_predict({"dataset_dir": "data/staging"}, "dev")

    def test_fixed_train_smoke_is_chronological_and_outcome_independent(self):
        inputs, labels = [], {}
        for index in range(40):
            date = (datetime(2023, 11, 1) + timedelta(days=index)).strftime("%Y-%m-%d")
            gid = f"g{index:02d}"
            inputs.append({"game_id": gid, "split": "train", "game_date_et": date,
                           "cutoff_utc": date + "T22:00:00Z"})
            labels[gid] = {"game_id": gid, "home_win": index % 2}
        combined, _, fit_ids, query_ids = runner.select_cohort(list(reversed(inputs)), labels, "train-smoke")
        self.assertEqual(fit_ids, [f"g{i:02d}" for i in range(24)])
        self.assertEqual(query_ids, [f"g{i:02d}" for i in range(24, 32)])
        changed = {gid: dict(row, home_win=1 - row["home_win"]) for gid, row in labels.items()}
        other, _, other_fit, other_query = runner.select_cohort(inputs, changed, "train-smoke")
        self.assertEqual((combined, fit_ids, query_ids), (other, other_fit, other_query))

    def test_train_smoke_keeps_same_day_on_one_side_of_boundary(self):
        inputs, labels = [], {}
        for index in range(32):
            date = (datetime(2023, 11, 1) + timedelta(days=index // 5)).strftime("%Y-%m-%d")
            gid = f"g{index:02d}"
            inputs.append({"game_id": gid, "split": "train", "game_date_et": date,
                           "cutoff_utc": date + "T22:00:00Z"})
            labels[gid] = {"game_id": gid, "home_win": index % 2}
        combined, _, fit_ids, query_ids = runner.select_cohort(inputs, labels, "train-smoke")
        by_id = {row["game_id"]: row for row in combined}
        self.assertEqual(len(fit_ids), 20)
        self.assertEqual(len(query_ids), 12)
        self.assertLess(max(by_id[gid]["game_date_et"] for gid in fit_ids),
                        min(by_id[gid]["game_date_et"] for gid in query_ids))

    def test_no_test_or_dev_rows_in_train_smoke(self):
        _, inputs, labels, _ = fixture()
        label_map = {row["game_id"]: row for row in labels}
        with self.assertRaisesRegex(ValueError, "Unexpected split"):
            runner.select_cohort(inputs, label_map, "train-smoke")
        with self.assertRaisesRegex(ValueError, "TEST is excluded"):
            runner.select_cohort(inputs, label_map, "test")

    def test_positive_class_is_located_by_label_not_column_position(self):
        class ReversedClasses:
            classes_ = np.array([1, 0])
            def predict_proba(self, x):
                return np.array([[.8, .2], [.3, .7]])
        np.testing.assert_array_equal(runner.positive_probabilities(ReversedClasses(), np.zeros((2, 1))), [.8, .3])

    def test_invalid_classes_and_probability_matrices_fail(self):
        class Fake:
            classes_ = np.array([0, 1])
            def predict_proba(self, x):
                return self.matrix
        model = Fake()
        for matrix in ([[.2, .3]], [[float("nan"), .5]], [[1.1, -.1]], [[.2, .8], [.2, .8]], [[.5]]):
            with self.subTest(matrix=matrix):
                model.matrix = matrix
                with self.assertRaisesRegex(ValueError, "probability matrix"):
                    runner.positive_probabilities(model, np.zeros((1, 1)))
        model.matrix = [[.2, .8]]
        for classes in ([0], [1, 2], [0, 0]):
            model.classes_ = np.asarray(classes)
            with self.assertRaisesRegex(ValueError, "binary outcome classes"):
                runner.positive_probabilities(model, np.zeros((1, 1)))

    def test_cutoff_and_current_game_lineage_are_checked(self):
        _, inputs, labels, _ = fixture()
        config = {"cutoff_minutes": 60, "split_seasons": {
            "train": [2023, 2023], "dev": [2024, 2024], "test": [2025, 2025]}}
        runner.validate_rows(inputs, labels, list(FEATURE_NAMES), config)
        inputs[0]["cutoff_utc"] = inputs[0]["scheduled_start_utc"]
        with self.assertRaisesRegex(ValueError, "cutoff"):
            runner.validate_rows(inputs, labels, list(FEATURE_NAMES), config)
        inputs[0]["cutoff_utc"] = "2023-11-01T22:00:00Z"
        inputs[0]["prior_game_ids"]["home"] = [inputs[0]["game_id"]]
        with self.assertRaisesRegex(ValueError, "own feature history"):
            runner.validate_rows(inputs, labels, list(FEATURE_NAMES), config)


if __name__ == "__main__":
    unittest.main()
