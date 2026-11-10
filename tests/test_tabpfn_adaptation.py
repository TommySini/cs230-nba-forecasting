"""Leakage and integrity checks for the three frozen TabPFN adaptation arms."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_gradient_boosting import FEATURE_NAMES, bootstrap_brier, sha256
import validate_tabpfn as validator
import run_tabpfn as runner


def fixture(variant="combined"):
    inputs = []
    for i in range(6):
        season = 2022 + i // 2
        day = f"{season}-11-{i % 2 + 1:02d}"
        inputs.append({"game_id": f"g{i}", "season": season,
                       "split": "dev" if season == 2024 else "train",
                       "game_date_et": day,
                       "features": {name: float(i) for name in FEATURE_NAMES}})
    labels = [{"game_id": row["game_id"], "home_win": i % 2} for i, row in enumerate(inputs)]
    manifest = {"stage": "dev", "feature_names": list(FEATURE_NAMES),
                "fit_game_ids": ["g0", "g1", "g2", "g3"], "query_game_ids": ["g4", "g5"],
                "baseline_config": {"split_seasons": {"train": [2012, 2023], "dev": [2024, 2024], "test": [2025, 2025]}}}
    protocol = {key: copy.deepcopy(manifest[key]) for key in (
        "feature_names", "fit_game_ids", "query_game_ids", "baseline_config")}
    protocol.update(inner_fit_game_ids=["g0", "g1"], inner_validation_game_ids=["g2", "g3"],
                    config={"seed": 230, "bootstrap_replicates": 50, "generator": {"n_samples": 3}})
    baseline = [{"game_id": row["game_id"], "game_date_et": row["game_date_et"],
                 "tabpfn": p, "rich_gradient_boosting": q}
                for row, p, q in zip(inputs[4:], [.3, .7], [.4, .6])]
    rows = [dict(row, **{variant: p}) for row, p in zip(baseline, [.2, .8])]
    return protocol, manifest, inputs, labels, baseline, rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AdaptationCohortTests(unittest.TestCase):
    def test_all_arms_recompute_on_the_same_baseline(self):
        for variant in validator.ADAPTATION_VARIANTS:
            with self.subTest(variant=variant):
                protocol, manifest, inputs, labels, baseline, rows = fixture(variant)
                validator.validate_adaptation_protocol(protocol, manifest, inputs)
                y, predictions, metrics = validator.validate_adaptation_rows(
                    variant, manifest, inputs, labels, rows, baseline)
                np.testing.assert_array_equal(y, [0, 1])
                self.assertAlmostEqual(metrics[variant]["brier"], .04)
                np.testing.assert_array_equal(predictions["tabpfn"], [.3, .7])

    def test_changed_comparator_is_rejected(self):
        _, manifest, inputs, labels, baseline, rows = fixture()
        for name in ("tabpfn", "rich_gradient_boosting"):
            with self.subTest(name=name):
                changed = copy.deepcopy(rows)
                changed[0][name] += .01
                with self.assertRaisesRegex(ValueError, "comparison probabilities changed"):
                    validator.validate_adaptation_rows("combined", manifest, inputs, labels, changed, baseline)

    def test_leaked_outcome_extra_forecast_and_missing_games_are_rejected(self):
        _, manifest, inputs, labels, baseline, rows = fixture()
        for changed in (rows[:-1], list(reversed(rows)),
                        [dict(rows[0], home_win=0), rows[1]],
                        [dict(rows[0], synthetic=.5), rows[1]]):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validator.validate_adaptation_rows("combined", manifest, inputs, labels, changed, baseline)

    def test_adaptation_probabilities_must_be_finite_and_bounded(self):
        _, manifest, inputs, labels, baseline, rows = fixture()
        for p in (float("nan"), float("inf"), -.01, 1.01):
            rows[0]["combined"] = p
            with self.subTest(p=p), self.assertRaisesRegex(ValueError, "finite probability"):
                validator.validate_adaptation_rows("combined", manifest, inputs, labels, rows, baseline)

    def test_inner_partition_cannot_admit_validation_or_dev_for_fitting(self):
        protocol, manifest, inputs, _, _, _ = fixture()
        for fitting in (["g0", "g1", "g2"], ["g0", "g4"], ["g0"], ["g1", "g0"]):
            protocol["inner_fit_game_ids"] = fitting
            with self.subTest(fitting=fitting), self.assertRaisesRegex(ValueError, "chronological TRAIN"):
                validator.validate_adaptation_protocol(protocol, manifest, inputs)

    def test_no_test_variant_or_test_rows(self):
        _, manifest, inputs, labels, baseline, rows = fixture()
        with self.assertRaisesRegex(ValueError, "TEST is excluded"):
            validator.validate_adaptation_rows("test", manifest, inputs, labels, rows, baseline)
        inputs[-1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "TEST is excluded"):
            validator.validate_adaptation_rows("combined", manifest, inputs, labels, rows, baseline)


class SavedAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "data/processed/tabpfn/adaptation"
        self.directory.mkdir(parents=True)
        self.protocol, self.manifest, self.inputs, self.labels, self.baseline, _ = fixture()
        (self.directory / "manifest.json").write_text(json.dumps(self.protocol))
        baseline_dir = self.root / "data/processed/tabpfn/dev"
        baseline_dir.mkdir()
        write_csv(baseline_dir / "predictions.csv", self.baseline)
        synthetic_dir = self.directory / "synthetic_data"
        synthetic_dir.mkdir()
        np.save(synthetic_dir / "features.npy", np.zeros((3, len(FEATURE_NAMES))))
        np.save(synthetic_dir / "labels.npy", np.array([0, 1, 0]))
        self.synthetic = {"feature_names": list(FEATURE_NAMES), "source_game_ids": ["g0", "g1"],
                          "input_partition_game_ids": ["g0", "g1"],
                          "artifact_sha256": {name: sha256(synthetic_dir / name)
                                              for name in ("features.npy", "labels.npy")}}
        (synthetic_dir / "manifest.json").write_text(json.dumps(self.synthetic))
        self.runs = {}
        for variant in validator.ADAPTATION_VARIANTS:
            output = self.directory / variant
            output.mkdir()
            _, _, _, _, _, rows = fixture(variant)
            write_csv(output / "predictions.csv", rows)
            y, predictions, metrics = validator.validate_adaptation_rows(
                variant, self.manifest, self.inputs, self.labels, rows, self.baseline)
            uncertainty = bootstrap_brier(y, predictions, [row["game_date_et"] for row in rows], 50, 230, "tabpfn")
            synthetic_count = 0 if variant == "finetune" else 3
            checkpoint = output / "selected.ckpt"
            checkpoint.write_bytes(b"synthetic checkpoint fixture")
            training = None if variant == "synthetic" else {
                "fit_game_ids": ["g0", "g1"], "validation_game_ids": ["g2", "g3"],
                "synthetic_rows": synthetic_count, "checkpoint_path": str(checkpoint.relative_to(self.root)),
                "checkpoint_sha256": sha256(checkpoint),
                "weights_before_sha256": hashlib.sha256(b"before").hexdigest(),
                "weights_after_sha256": hashlib.sha256(b"after").hexdigest()}
            self.runs[variant] = {
                "stage": "adapt-dev", "status": "complete", "variant": variant,
                "manifest_sha256": sha256(self.directory / "manifest.json"),
                "artifact_sha256": {"predictions.csv": sha256(output / "predictions.csv")},
                "metrics": metrics, "uncertainty": uncertainty,
                "primary_comparison": {"brier_difference": metrics[variant]["brier"] - metrics["tabpfn"]["brier"],
                                       "difference_95_interval": uncertainty["difference_vs_reference"][variant]},
                "context_audit": {"original_train_rows": 4, "synthetic_rows": synthetic_count,
                                  "submitted_rows": 4 + synthetic_count, "model_n_train_samples": 4 + synthetic_count,
                                  "row_subsampling_applied": False},
                "training": training,
                "synthetic_manifest_sha256": None if variant == "finetune" else sha256(synthetic_dir / "manifest.json")}
            self.save(variant)

    def save(self, variant):
        (self.directory / variant / "run.json").write_text(json.dumps(self.runs[variant]))

    def validate(self, variant="combined"):
        with patch.object(validator, "ROOT", self.root), patch("run_tabpfn.load_adaptation", create=True,
                return_value=(self.protocol, self.manifest, self.inputs, self.labels)), patch(
                "run_tabpfn.load_comparison_rows", return_value=copy.deepcopy(self.baseline)):
            return validator.validate_adaptation(variant)

    def rebind_synthetic(self):
        path = self.directory / "synthetic_data/manifest.json"
        path.write_text(json.dumps(self.synthetic))
        for variant in ("synthetic", "combined"):
            self.runs[variant]["synthetic_manifest_sha256"] = sha256(path)
            self.save(variant)

    def test_saved_arms_reproduce_without_model_inference(self):
        for variant in validator.ADAPTATION_VARIANTS:
            with self.subTest(variant=variant):
                result = self.validate(variant)
                self.assertEqual(result["query_games"], 2)
                self.assertAlmostEqual(result["primary_comparison"]["brier_difference"], -.05)

    def test_generator_validation_leakage_fails_even_with_rebound_hash(self):
        self.synthetic["source_game_ids"].append("g2")
        self.rebind_synthetic()
        with self.assertRaisesRegex(ValueError, "unique earlier TRAIN"):
            self.validate()

    def test_generator_input_partition_cannot_include_validation(self):
        self.synthetic["input_partition_game_ids"].append("g2")
        self.rebind_synthetic()
        with self.assertRaisesRegex(ValueError, "input partition"):
            self.validate()

    def test_corrupt_synthetic_bytes_fail_before_scores(self):
        with (self.directory / "synthetic_data/features.npy").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Synthetic artifact changed"):
            self.validate()

    def test_query_outcomes_cannot_enter_fine_tuning_validation(self):
        self.runs["combined"]["training"]["validation_game_ids"].append("g4")
        self.save("combined")
        with self.assertRaisesRegex(ValueError, "chronological TRAIN partitions"):
            self.validate()

    def test_changed_selected_checkpoint_is_rejected(self):
        (self.directory / "combined/selected.ckpt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            self.validate()

    def test_row_subsampling_and_wrong_context_count_are_rejected(self):
        for field, value in (("row_subsampling_applied", True), ("original_train_rows", 3),
                             ("model_n_train_samples", 3), ("submitted_rows", 3)):
            old = self.runs["combined"]["context_audit"][field]
            self.runs["combined"]["context_audit"][field] = value
            self.save("combined")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "context_audit"):
                self.validate()
            self.runs["combined"]["context_audit"][field] = old

    def test_augmentation_size_cannot_be_changed_after_freezing(self):
        self.runs["combined"]["context_audit"]["synthetic_rows"] = 2
        self.save("combined")
        with self.assertRaisesRegex(ValueError, "frozen augmentation size"):
            self.validate()

    def test_fabricated_metric_and_missing_training_audit_fail(self):
        self.runs["combined"]["metrics"]["combined"]["brier"] = .001
        self.save("combined")
        with self.assertRaisesRegex(ValueError, "metrics"):
            self.validate()
        self.runs["combined"]["training"] = None
        self.save("combined")
        with self.assertRaisesRegex(ValueError, "Fine-tuning evidence"):
            self.validate()


class AdaptationManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "data/processed/tabpfn/adaptation"
        self.directory.mkdir(parents=True)
        self.protocol, self.manifest, self.inputs, self.labels, _, _ = fixture()
        self.config = {"output_dir": "data/processed/tabpfn/adaptation", "baseline_config": "config/tabpfn.json"}
        baseline_dir = self.directory.parent / "dev"
        baseline_dir.mkdir()
        (baseline_dir / "manifest.json").write_text(json.dumps(self.manifest))
        (self.root / "implementation.py").write_text("original implementation")
        self.protocol.update(config=self.config, baseline_revision=runner.BASELINE_REVISION,
                             baseline_manifest_sha256=sha256(baseline_dir / "manifest.json"),
                             source_sha256={"implementation.py": sha256(self.root / "implementation.py")})
        self.save()

    def save(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.protocol))

    def load(self):
        with patch.object(runner, "ROOT", self.root), patch.object(
                runner, "adaptation_config", return_value=self.config), patch.object(
                runner, "load_config", return_value={"output_dir": "data/processed/tabpfn"}), patch.object(
                runner, "load_prepared", return_value=(self.manifest, self.inputs, self.labels)), patch.object(
                runner, "record_directory", return_value=self.directory.parent / "dev"):
            return runner.load_adaptation()

    def test_bound_manifest_loads_without_model_inference(self):
        self.assertEqual(self.load()[0], self.protocol)

    def test_modified_implementation_fails_before_experiment(self):
        (self.root / "implementation.py").write_text("changed after preparation")
        with self.assertRaisesRegex(ValueError, "Changed frozen artifact"):
            self.load()

    def test_forged_fitting_partition_rejected_even_if_source_files_match(self):
        self.protocol["inner_fit_game_ids"].append("g4")
        self.save()
        with self.assertRaisesRegex(ValueError, "partition or feature metadata changed"):
            self.load()

    def test_changed_baseline_manifest_rejected(self):
        with (self.directory.parent / "dev/manifest.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Frozen baseline manifest changed"):
            self.load()

    def test_historical_code_verification_is_explicit_and_never_applies_to_data(self):
        historical = b"original historical source"
        digest = hashlib.sha256(historical).hexdigest()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/run_tabpfn.py").write_bytes(b"new source")
        (self.root / "data/labels.jsonl").write_bytes(b"changed outcomes")
        with patch.object(runner, "ROOT", self.root), patch.object(runner.subprocess, "check_output", return_value=historical):
            with self.assertRaisesRegex(ValueError, "Changed frozen artifact"):
                runner.verify_hashes({"scripts/run_tabpfn.py": digest})
            runner.verify_hashes({"scripts/run_tabpfn.py": digest}, source_revision=runner.BASELINE_REVISION)
            with self.assertRaisesRegex(ValueError, "Changed frozen artifact"):
                runner.verify_hashes({"data/labels.jsonl": digest}, source_revision=runner.BASELINE_REVISION)


class SyntheticFeatureTests(unittest.TestCase):
    def test_quantile_encoding_decodes_from_train_medians_and_separate_missing_category(self):
        from tabpfn_synthetic import decode_columns, encode_columns
        x = np.array([[0., 2.], [1., 2.], [2., 2.], [3., np.nan], [np.nan, 2.]])
        encoded, specs = encode_columns(x, ["continuous", "missing"], max_bins=2)
        generated = np.column_stack([np.array([0, 1, 0, 1, 0]), encoded])
        decoded = decode_columns(generated, specs)
        np.testing.assert_array_equal(np.isnan(decoded["continuous"]), [False, False, False, False, True])
        np.testing.assert_array_equal(np.isnan(decoded["missing"]), [False, False, False, True, False])
        self.assertTrue(set(decoded["continuous"][:4]).issubset({.5, 2.5}))
        self.assertNotIn("home_win", decoded)
        generated[0, 1] = 100
        with self.assertRaisesRegex(ValueError, "unknown category"):
            decode_columns(generated, specs)

    def test_generated_features_preserve_exact_identities_and_missingness(self):
        from tabpfn_synthetic import primitive_names, reconstruct_features
        from box_score_features import PLAYER_STAT_METRICS, WINDOWS
        names = list(FEATURE_NAMES)
        primitive = primitive_names(names)
        values = {name: np.array([.5, .5]) for name in primitive}
        for side in ("home", "away"):
            values[f"{side}_previous_games"] = np.array([10., 12.])
            values[f"{side}_days_since_previous_game"] = np.array([1., 2.])
            for window, _ in WINDOWS:
                values[f"{side}_{window}_estimated_offensive_rating"] = np.array([110., 100.])
                values[f"{side}_{window}_estimated_defensive_rating"] = np.array([100., 105.])
                values[f"{side}_{window}_team_boxscore_coverage"] = np.array([1., .25])
                values[f"{side}_{window}_player_boxscore_coverage"] = np.array([1., .5])
        # At ten prior games, last10 and season are the same underlying window.
        values["home_last10_estimated_offensive_rating"][0] = 150.
        result = reconstruct_features(values, names)
        by_name = {name: result[:, i] for i, name in enumerate(names)}
        for name in names:
            if name.startswith("difference_"):
                tail = name.removeprefix("difference_")
                np.testing.assert_allclose(by_name[name], by_name["home_" + tail] - by_name["away_" + tail], equal_nan=True)
        for side in ("home", "away"):
            np.testing.assert_array_equal(by_name[f"{side}_back_to_back"], [1., 0.])
            for window, _ in WINDOWS:
                np.testing.assert_array_equal(by_name[f"{side}_{window}_estimated_net_rating"], [10., -5.])
                np.testing.assert_array_equal(by_name[f"{side}_{window}_player_boxscore_missing"], [0., 1.])
                self.assertTrue(np.all(by_name[f"{side}_{window}_player_boxscore_coverage"]
                                       <= by_name[f"{side}_{window}_team_boxscore_coverage"]))
                for metric in PLAYER_STAT_METRICS:
                    self.assertTrue(np.isnan(by_name[f"{side}_{window}_{metric}"][1]))
                    self.assertTrue(np.isfinite(by_name[f"{side}_{window}_{metric}"][0]))


class CombinedTrainingBoundaryTests(unittest.TestCase):
    def test_combined_adds_only_generated_rows_to_earlier_train_and_keeps_validation_real(self):
        protocol, manifest, inputs, labels, _, _ = fixture()
        config = {"output_dir": "data/processed/tabpfn/adaptation", "baseline_config": "config/tabpfn.json",
                  "finetuning": {"max_runtime_seconds": 60}}
        protocol["config"] = config
        generated_x = np.vstack([np.full(len(FEATURE_NAMES), 99.), np.full(len(FEATURE_NAMES), 100.)])
        generated_y = np.array([1, 0])
        synthetic_manifest = {"input_partition_game_ids": ["g0", "g1"], "source_game_ids": ["g0", "g1"]}
        calls = []

        def train(settings, x, y, x_val, y_val, output):
            calls.append((x.copy(), y.copy(), x_val.copy(), y_val.copy()))
            return {"runtime_seconds": 1., "optimizer_steps": 1, "epochs_completed": 1,
                    "initial_inner_log_loss": .7, "best_inner_log_loss": .6,
                    "selected_pretrained_weights": False}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / config["output_dir"]
            output.mkdir(parents=True)
            (output / "manifest.json").write_text(json.dumps(protocol))
            with patch.object(runner, "ROOT", root), patch.object(
                    runner, "load_adaptation", return_value=(protocol, manifest, inputs, labels)), patch.object(
                    runner, "load_config", return_value={"model": {}}), patch.object(
                    runner, "adaptation_synthetic", return_value=(generated_x, generated_y, synthetic_manifest)), patch(
                    "tabpfn_finetuning.train", side_effect=train), patch.object(runner.signal, "signal"), patch.object(runner.signal, "alarm"):
                for variant in ("finetune", "combined"):
                    runner.run_adaptation("adapt-fit", variant)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0][0][:, 0], [0., 1.])
        np.testing.assert_array_equal(calls[0][1], [0, 1])
        np.testing.assert_array_equal(calls[1][0][:, 0], [0., 1., 99., 100.])
        np.testing.assert_array_equal(calls[1][1], [0, 1, 1, 0])
        for _, _, x_val, y_val in calls:
            np.testing.assert_array_equal(x_val[:, 0], [2., 3.])
            np.testing.assert_array_equal(y_val, [0, 1])


if __name__ == "__main__":
    unittest.main()
