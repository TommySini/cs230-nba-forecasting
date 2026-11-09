"""Storage migration contracts: exact data, split isolation and tamper rejection."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nba_dataset import (Dataset, SPLITS, read_rows, remap_paths, remap_raw_path,
                         resolve_source_path, sha256, stage_dataset, store_predictions, publish_stage, cleanup_dataset,
                         relocate_raw_sources, write_json, write_table)


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(root):
    data = root / "data/processed"
    history, inputs, labels, boxes, excluded, forecasts = [], [], [], [], [], []
    names = ["home_previous_games", "home_season_win_rate"]
    for season, split in zip((2023, 2024, 2025), SPLITS):
        for index in range(2):
            gid = f"espn:{season}{index}"
            start = datetime(season, 11, index + 1, 23, tzinfo=timezone.utc)
            iso = lambda value: value.isoformat().replace("+00:00", "Z")
            game = {"game_id": gid, "espn_id": gid.split(":")[1], "season": season,
                    "scheduled_start_utc": iso(start), "game_date_et": start.date().isoformat(),
                    "schedule_source": f"data/raw/scoreboards/{gid.split(':')[1]}.json",
                    "home_team": "Boston Celtics", "home_abbr": "BOS", "home_score": 101,
                    "away_team": "New York Knicks", "away_abbr": "NYK", "away_score": 100}
            history.append(game)
            coverage = {side: {"previous_games": index, "valid_box_scores": index,
                               "valid_player_box_scores": index} for side in ("home", "away")}
            if index:
                record = {key: game[key] for key in ("game_id", "season", "scheduled_start_utc", "game_date_et", "home_abbr", "away_abbr")}
                record.update(split=split, cutoff_utc=iso(start - timedelta(hours=1)),
                              features={names[0]: 1.0, names[1]: None if split == "dev" else 0.12345678901234568},
                              prior_game_ids={side: [f"espn:{season}0"] for side in ("home", "away")},
                              missing_box_score_game_ids={"home": [], "away": []}, box_score_coverage=coverage)
                inputs.append(record)
                labels.append({"game_id": gid, "home_win": 1})
                forecasts.append({"game_id": gid, "season": season, "game_date_et": game["game_date_et"],
                                  "split": split, "p_boost": 0.9876543210987654, "produced_by": "train_fit"})
            else:
                excluded.append({"game_id": gid, "season": season, "split": split, "game_date_et": game["game_date_et"],
                                 "reason": "fewer_than_min_previous_games", "details": ["not enough prior games"], "coverage": coverage})
            box = {"schema_version": "fixture", "game_id": gid, "season": season, "game_date_et": game["game_date_et"],
                   "scheduled_start_utc": game["scheduled_start_utc"], "game_minutes": 48, "duration_basis": "fixture",
                   "complete": True, "players_complete": True, "time_semantics": "retrospective",
                   "provenance": {"provider": "ESPN", "raw_path": f"data/raw/summaries/{season}{index}.json",
                                  "sha256": "0" * 64, "url": "https://example.invalid", "source_game_id": gid,
                                  "retrieved_at_utc": "2026-11-09T00:00:00Z"},
                   "teams": {side: {"team_abbr": game[side + "_abbr"], "pts": game[side + "_score"],
                                    "players": [{"player_id": str(index), "minutes": 24.0, "starter": True}],
                                    "player_issues": [], "complete": True}
                             for side in ("home", "away")}}
            if index:
                box["teams"]["home"]["team_rebounds"] = 3
                box["provenance"]["normalization"] = "preserve optional provenance"
            boxes.append(box)
    config = {"histories": [{"path": "data/processed/history/game_history.jsonl", "config": "config/seasonal.json"}],
              "box_scores": "data/processed/box_scores", "cutoff_minutes": 60,
              "split_seasons": {"train": [2023, 2023], "dev": [2024, 2024], "test": [2025, 2025]}}
    write_lines(data / "history/game_history.jsonl", history)
    write_lines(data / "box_scores/games.jsonl", boxes)
    write_lines(data / "box_scores/failures.jsonl", [])
    write_json(data / "box_scores/manifest.json", {"artifact_sha256": {"games.jsonl": sha256(data / "box_scores/games.jsonl")},
                                                  "limitation": "Historical correction times unknown"})
    for name, rows in (("inputs", inputs), ("labels", labels), ("excluded", excluded), ("predictions", forecasts)):
        write_lines(data / f"gradient_boosting/{name}.jsonl", rows)
    (data / "gradient_boosting/model.joblib").write_bytes(b"not-loaded-by-converter")
    artifacts = {str(path.relative_to(root)): sha256(path) for path in (data / "gradient_boosting").iterdir()}
    write_json(root / "reports/gradient_boosting/metrics.json", {"config": config, "feature_names": names, "artifact_sha256": artifacts})
    return {"inputs": inputs, "labels": labels, "history": history, "boxes": boxes, "excluded": excluded, "forecasts": forecasts}


class SharedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = fixture(self.root)
        self.stage = self.root / "staging"
        stage_dataset(self.root, self.stage)

    def dataset(self):
        return Dataset(self.root, self.stage)

    def rehash(self, table):
        manifest = json.loads((self.stage / "manifest.json").read_text())
        manifest["tables"][table]["sha256"] = sha256(self.stage / (table + ".parquet"))
        write_json(self.stage / "manifest.json", manifest)

    def test_exact_round_trip_and_typed_float_precision(self):
        dataset = self.dataset()
        self.assertEqual(dataset.inputs(), self.expected["inputs"])
        self.assertEqual(dataset.labels(SPLITS), self.expected["labels"])
        self.assertEqual(dataset.history(), remap_paths(self.expected["history"]))
        self.assertEqual(dataset.box_scores(), remap_paths(self.expected["boxes"]))
        self.assertEqual(dataset.excluded(), self.expected["excluded"])
        self.assertEqual(dataset.predictions(), self.expected["forecasts"])
        self.assertEqual(dataset.verify()["features"], 3)
        self.assertNotIn("team_rebounds", dataset.box_scores()[0]["teams"]["home"])
        self.assertEqual(dataset.box_scores()[1]["teams"]["home"]["team_rebounds"], 3)
        self.assertEqual(len(dataset.labels(["train"], eligible_only=False)), 2)

    def test_train_read_never_opens_dev_or_test_labels(self):
        dataset = self.dataset()
        opened = []
        original = dataset.table
        def table(name):
            opened.append(name)
            if name in ("labels_dev", "labels_test"):
                self.fail("TRAIN loader opened a forbidden outcome partition")
            return original(name)
        with patch.object(dataset, "table", side_effect=table):
            self.assertEqual(len(dataset.inputs(["train"])), 1)
            self.assertEqual(len(dataset.labels(["train"])), 1)
            dataset.verify(label_splits={"train"})
        self.assertEqual(set(name for name in opened if name.startswith("labels_")), {"labels_train"})

    def test_input_only_validation_never_opens_outcomes_or_completed_box_statistics(self):
        dataset = self.dataset()
        original = dataset.table
        def table(name):
            self.assertFalse(name.startswith("labels_") or name in ("box_score_games", "team_box_scores", "player_box_scores"))
            return original(name)
        with patch.object(dataset, "table", side_effect=table):
            dataset.verify(label_splits=())

    def test_data_corruption_is_rejected(self):
        path = self.stage / "features.parquet"
        path.write_bytes(path.read_bytes() + b"tampering")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.dataset().inputs()

    def test_reordered_feature_columns_are_rejected_even_with_updated_hash(self):
        rows = read_rows(self.stage / "features.parquet")
        write_table(self.stage / "features.parquet", rows,
                    pa.schema([("game_id", pa.string()), ("home_season_win_rate", pa.float64()), ("home_previous_games", pa.float64())]))
        self.rehash("features")
        with self.assertRaisesRegex(ValueError, "column order"):
            self.dataset().verify()

    def test_future_game_in_lineage_is_rejected(self):
        rows = read_rows(self.stage / "feature_lineage.parquet")
        rows[0]["prior_game_id"] = rows[0]["game_id"]
        write_table(self.stage / "feature_lineage.parquet", rows)
        self.rehash("feature_lineage")
        with self.assertRaisesRegex(ValueError, "Current/future"):
            self.dataset().verify()

    def test_duplicate_ids_are_rejected(self):
        rows = read_rows(self.stage / "features.parquet")
        rows[1]["game_id"] = rows[0]["game_id"]
        write_table(self.stage / "features.parquet", rows)
        self.rehash("features")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.dataset().inputs()

    def test_predictions_are_idempotent_and_do_not_change_dataset_identity(self):
        for path in self.stage.iterdir():
            if path.is_file():
                shutil.copyfile(path, self.root / "data/processed" / path.name)
        identity = Dataset(self.root).manifest["dataset_id"]
        rows = [{"game_id": "espn:20231", "p_home_win": 0.75, "produced_by": "fixture"}]
        self.assertEqual(store_predictions(self.root, rows, "new_model")["added"], 1)
        self.assertEqual(store_predictions(self.root, rows, "new_model")["added"], 0)
        self.assertEqual(Dataset(self.root).manifest["dataset_id"], identity)
        rows[0]["p_home_win"] = 0.76
        with self.assertRaisesRegex(ValueError, "overwrite"):
            store_predictions(self.root, rows, "new_model")

    def _interrupted_prediction_append(self, interruption, corrupt_candidate=False):
        publish_stage(self.root, self.stage, relocate_raw=False)
        before = Dataset(self.root).table("predictions")
        identity = Dataset(self.root).manifest["dataset_id"]
        original_replace = Path.replace
        def interrupted(path, target):
            if interruption == "after_table" and path.name == ".predictions.next.parquet":
                original_replace(path, target)
                raise OSError("simulated termination after prediction replacement")
            if interruption == "manifest" and path.name == ".manifest.next.json" and Path(target).name == "manifest.json":
                raise OSError("simulated termination during manifest publication")
            return original_replace(path, target)
        with patch.object(Path, "replace", interrupted):
            with self.assertRaisesRegex(OSError, "simulated termination"):
                store_predictions(self.root, [{"game_id": "espn:20231", "p_home_win": 0.75}], "recovery_probe")
        directory = self.root / "data/processed"
        # The published manifest is always complete JSON, even before recovery.
        json.loads((directory / "manifest.json").read_text())
        if corrupt_candidate:
            (directory / ".manifest.next.json").write_text("incomplete candidate")
        recovered = Dataset(self.root)
        after = recovered.table("predictions")
        self.assertEqual(after[:len(before)], before)
        self.assertEqual(len(after), len(before) + (0 if corrupt_candidate else 1))
        self.assertEqual(recovered.manifest["dataset_id"], identity)
        recovered.verify(label_splits=())
        self.assertFalse((directory / ".prediction_transaction.json").exists())
        self.assertFalse(list(directory.glob("*.previous.*")))
        self.assertFalse(list(directory.glob("*.next.*")))
        # Retry finishes a rolled-back transaction or is an exact no-op after roll-forward.
        result = store_predictions(self.root, [{"game_id": "espn:20231", "p_home_win": 0.75}], "recovery_probe")
        self.assertEqual(result["added"], 1 if corrupt_candidate else 0)

    def test_prediction_append_recovers_after_table_replacement(self):
        self._interrupted_prediction_append("after_table")

    def test_prediction_append_recovers_during_atomic_manifest_publication(self):
        self._interrupted_prediction_append("manifest")

    def test_prediction_append_rolls_back_when_pending_manifest_is_unavailable(self):
        self._interrupted_prediction_append("after_table", corrupt_candidate=True)

    def test_existing_stage_is_never_overwritten(self):
        before = sha256(self.stage / "manifest.json")
        with self.assertRaisesRegex(ValueError, "empty"):
            stage_dataset(self.root, self.stage)
        self.assertEqual(before, sha256(self.stage / "manifest.json"))

    def test_altered_excluded_game_outcome_fails_original_source_pin(self):
        history_path = self.root / "data/processed/history/game_history.jsonl"
        metrics_path = self.root / "reports/gradient_boosting/metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["source_sha256"] = {str(history_path.relative_to(self.root)): sha256(history_path)}
        write_json(metrics_path, metrics)
        history = read_rows(history_path)
        history[0]["home_score"] += 1
        write_lines(history_path, history)
        with self.assertRaisesRegex(ValueError, "Frozen source checksum"):
            stage_dataset(self.root, self.root / "changed-stage")

    def test_raw_mapping_and_source_containment(self):
        self.assertEqual(remap_raw_path("data/raw/summaries"), "data/raw/espn/summaries")
        old = self.root / "data/raw/summaries/test.json"
        old.parent.mkdir(parents=True)
        old.write_text("{}")
        self.assertEqual(resolve_source_path(self.root, "data/raw/espn/summaries/test.json"), old)
        with self.assertRaises(ValueError):
            resolve_source_path(self.root, "data/raw/../../outside")
        outside = self.root / "outside.json"
        outside.write_text("{}")
        escape = old.parent / "escape.json"
        escape.symlink_to(outside)
        with self.assertRaises(ValueError):
            resolve_source_path(self.root, "data/raw/espn/summaries/escape.json")

    def test_publish_and_cleanup_preserve_raw_bytes_and_unique_models(self):
        raw = self.root / "data/raw/summaries/a.json"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'{"exact": "bytes"}\n')
        sidecar = raw.with_suffix(".meta.json")
        sidecar.write_bytes(b'{"raw_path":"data/raw/summaries/a.json"}\n')
        original = {p.name: p.read_bytes() for p in raw.parent.iterdir()}
        manifest = publish_stage(self.root, self.stage)
        self.assertEqual(manifest["publication"]["raw_files"], 2)
        self.assertFalse(raw.exists())
        for name, body in original.items():
            self.assertEqual((self.root / "data/raw/espn/summaries" / name).read_bytes(), body)
        self.assertTrue((self.root / "data/processed/gradient_boosting/inputs.jsonl").exists())
        self.assertTrue((self.root / "data/models/gradient_boosting/model.joblib").exists())
        self.assertEqual(publish_stage(self.root, self.stage)["dataset_id"], manifest["dataset_id"])
        with self.assertRaisesRegex(ValueError, "validators"):
            cleanup_dataset(self.root)
        result = cleanup_dataset(self.root, validated=True)
        self.assertEqual(result["status"], "complete")
        self.assertFalse((self.root / "data/processed/gradient_boosting").exists())
        self.assertEqual(Dataset(self.root).inputs(), self.expected["inputs"])
        self.assertEqual(Dataset(self.root).verify()["features"], 3)

    def test_publish_rolls_back_on_raw_move_failure(self):
        raw = self.root / "data/raw/summaries/a.json"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"raw evidence")
        original_rename = Path.rename
        def fail_raw(path, target):
            if path == raw.parent:
                raise OSError("simulated raw move failure")
            return original_rename(path, target)
        with patch.object(Path, "rename", fail_raw):
            with self.assertRaisesRegex(OSError, "simulated"):
                publish_stage(self.root, self.stage)
        self.assertEqual(raw.read_bytes(), b"raw evidence")
        self.assertFalse((self.root / "data/processed/manifest.json").exists())
        self.assertFalse((self.root / "data/processed/features.parquet").exists())
        self.assertTrue((self.root / "data/processed/gradient_boosting/inputs.jsonl").exists())
        self.assertEqual(json.loads((self.stage / "publish_journal.json").read_text())["status"], "rolled_back")
        publish_stage(self.root, self.stage)

    def test_cleanup_refuses_unknown_or_changed_files_before_deleting_anything(self):
        publish_stage(self.root, self.stage)
        old = self.root / "data/processed/gradient_boosting"
        extra = old / "unmapped.txt"
        extra.write_text("unique evidence")
        with self.assertRaisesRegex(ValueError, "coverage"):
            cleanup_dataset(self.root, validated=True)
        self.assertTrue((old / "inputs.jsonl").exists())
        extra.unlink()
        (old / "inputs.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Original changed"):
            cleanup_dataset(self.root, validated=True)
        self.assertTrue((old / "model.joblib").exists())

    def test_additive_publish_defers_relocation_and_cleanup_for_active_readers(self):
        raw = self.root / "data/raw/summaries/a.json"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"raw evidence")
        manifest = publish_stage(self.root, self.stage, relocate_raw=False)
        self.assertEqual(manifest["publication"]["raw_layout"], "original")
        self.assertTrue(raw.exists())
        self.assertEqual(Dataset(self.root).inputs(), self.expected["inputs"])
        with self.assertRaisesRegex(ValueError, "pending"):
            cleanup_dataset(self.root, validated=True)
        result = relocate_raw_sources(self.root)
        self.assertEqual(result["raw_layout"], "grouped")
        self.assertFalse(raw.exists())
        self.assertEqual((self.root / "data/raw/espn/summaries/a.json").read_bytes(), b"raw evidence")
        cleanup_dataset(self.root, validated=True)

    def test_cleanup_preserves_unrelated_experiment_in_model_parent(self):
        old = self.root / "data/processed/tabpfn/train-smoke/covered.txt"
        old.parent.mkdir(parents=True)
        old.write_text("represented in shared tables")
        active = self.root / "data/processed/tabpfn/adaptation/model.ckpt"
        active.parent.mkdir()
        active.write_bytes(b"unique ongoing experiment")
        manifest = json.loads((self.stage / "manifest.json").read_text())
        relative = str(old.relative_to(self.root))
        manifest["cleanup_coverage"][relative] = {"sha256": sha256(old), "replacement": "features.parquet"}
        manifest["source_sha256"][relative] = sha256(old)
        manifest["remove_candidates"].append(str(old.parent.relative_to(self.root)))
        write_json(self.stage / "manifest.json", manifest)
        publish_stage(self.root, self.stage)
        cleanup_dataset(self.root, validated=True)
        self.assertFalse(old.exists())
        self.assertEqual(active.read_bytes(), b"unique ongoing experiment")

    def test_cleanup_refuses_changed_surviving_raw_body_before_deleting_duplicates(self):
        source = self.root / "data/raw/summaries/a.json"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"original raw evidence")
        publish_stage(self.root, self.stage)
        surviving = self.root / "data/raw/espn/summaries/a.json"
        surviving.write_bytes(b"tampered raw evidence")
        with self.assertRaisesRegex(ValueError, "Surviving raw source"):
            cleanup_dataset(self.root, validated=True)
        self.assertTrue((self.root / "data/processed/gradient_boosting/inputs.jsonl").exists())

    def test_cleanup_refuses_missing_surviving_raw_sidecar_before_deleting_duplicates(self):
        source = self.root / "data/raw/summaries/a.meta.json"
        source.parent.mkdir(parents=True)
        source.write_bytes(b'{"metadata":"original"}')
        publish_stage(self.root, self.stage)
        (self.root / "data/raw/espn/summaries/a.meta.json").unlink()
        with self.assertRaisesRegex(ValueError, "Surviving raw source"):
            cleanup_dataset(self.root, validated=True)
        self.assertTrue((self.root / "data/processed/gradient_boosting/inputs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
