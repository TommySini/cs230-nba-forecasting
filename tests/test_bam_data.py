"""Temporal boundaries and immutable source/page snapshots for the new pipeline."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bam_data as bam


GAME = {"game_id": "espn:1", "game_date_et": "2024-01-01", "home_abbr": "BOS", "away_abbr": "DEN",
        "scheduled_start_utc": "2024-01-02T01:00:00Z", "cutoff_utc": "2024-01-02T00:00:00Z"}


def pdf_doc(root, body=b"exact archived bytes", capture=datetime(2024, 1, 1, 12, tzinfo=timezone.utc), team="BOS"):
    path = Path(root) / f"data/raw/{team}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"game_id": "espn:1", "team": team, "kind": "rolling_archive", "tier": "archive_proven",
            "url": f"http://www.nba.com/gamenotes/{team}.pdf", "cutoff_utc": GAME["cutoff_utc"],
            "replay_url": "https://web.archive.org/web/20240101120000id_/http://www.nba.com/gamenotes/x.pdf",
            "text_path": f"data/processed/notes_v2/texts/1-{team}.txt",
            "source": {"raw_path": f"data/raw/{team}.pdf", "sha256": bam.digest(body),
                       "memento_datetime": format_datetime(capture, usegmt=True)}}


def html_doc(root, modified="2024-01-01T15:00:00Z"):
    items = [{"permalink": "https://www.nba.com/knicks/news/other", "title": "Other", "contentExpanded": []},
             {"permalink": "https://www.nba.com/knicks/news/notes", "title": "Knicks Game Notes",
              "contentExpanded": [{"type": "paragraph", "text": "Opening line"}, {"type": "heading", "text": "SERIES NOTES"},
                                  {"type": "paragraph", "html": "<p>Boston leads <b>2-1</b></p>"}]}]
    path = Path(root) / "data/raw/api.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({"items": items}).encode())
    return {"game_id": "espn:1", "team": "NYK", "kind": "html_post", "tier": "site_timestamped",
            "url": items[1]["permalink"], "published_utc": "2024-01-01T14:00:00Z", "modified_utc": modified,
            "modified_after_cutoff": modified > GAME["cutoff_utc"], "cutoff_utc": GAME["cutoff_utc"],
            "text_path": "data/processed/notes_v2/texts/1-NYK-html.txt",
            "source": {"api_raw_path": "data/raw/api.json", "api_sha256": bam.digest(path.read_bytes())}}, items[1]


class NotesV2Tests(unittest.TestCase):
    def test_late_revised_html_and_unscheduled_games_are_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            pdf = pdf_doc(root)
            late, _ = html_doc(root, modified="2024-01-02T03:00:00Z")
            other = dict(pdf_doc(root, team="DEN"), game_id="espn:2")
            rows, bodies, excluded = bam.notes_v2_games([pdf, late, other], {"espn:1": GAME}, root)
            self.assertEqual(excluded, {"html_modified_after_cutoff": 1, "outside_numeric_history": 1})
            self.assertEqual(len(rows), 1)
            source = rows[0]["sources"][0]
            self.assertEqual((source["source_kind"], source["evidence_tier"]), ("pdf", "archive_proven"))
            self.assertEqual(source["archive_time"], "2024-01-01T12:00:00Z")
            self.assertEqual(bodies[source["sha256"]], b"exact archived bytes")
            self.assertNotIn("home_win", rows[0])

    def test_source_after_cutoff_or_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            late = pdf_doc(root, capture=datetime(2024, 1, 2, 0, 1, tzinfo=timezone.utc))
            with self.assertRaisesRegex(ValueError, "cutoff"):
                bam.notes_v2_games([late], {"espn:1": GAME}, root)
            doc = pdf_doc(root)
            (Path(root) / doc["source"]["raw_path"]).write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                bam.notes_v2_games([doc], {"espn:1": GAME}, root)

    def test_html_post_becomes_sections_with_title_and_heading_context(self):
        with tempfile.TemporaryDirectory() as root:
            doc, item = html_doc(root)
            rows, bodies, _ = bam.notes_v2_games([doc], {"espn:1": GAME}, root)
            source = rows[0]["sources"][0]
            self.assertEqual(source["sha256"], bam.digest(bam.canonical(item)))
            self.assertEqual((source["source_kind"], source["archive_time"]), ("html", "2024-01-01T15:00:00Z"))
            _, pages = bam.extract_pages(("html", self.write(root, bodies[source["sha256"]]), source["sha256"]))
            self.assertEqual([p["label"] for p in pages], ["title", "paragraph", "heading", "paragraph"])
            self.assertEqual(" ".join(pages[3]["text"].split()), "Boston leads 2-1")
            self.assertEqual([c["label"] for c in pages[3]["context_sections"]], ["title", "heading"])
            self.assertEqual(bam.pages_with_lines(pages), pages)

    def write(self, root, body):
        path = Path(root) / "raw.bin"
        path.write_bytes(body)
        return str(path)


class BoostingTests(unittest.TestCase):
    def setUp(self):
        self.history = {f"g{i}": {"game_id": f"g{i}", "season": season, "game_date_et": date,
                                  "scheduled_start_utc": date + "T23:00:00Z"}
                        for i, (season, date) in enumerate([(2015, "2016-01-01"), (2023, "2024-01-01"), (2024, "2025-01-01"), (2025, "2026-01-01")])}
        self.splits = {2015: "train", 2023: "train", 2024: "dev", 2025: "test"}

    def rows(self):
        return [{"game_id": g, "game_date_et": game["game_date_et"], "sources": []} for g, game in self.history.items()]

    def test_split_follows_season_and_only_predicted_games_are_eligible(self):
        predictions = {"g1": {"split": "train", "game_date_et": "2024-01-01", "p_boost": 0.6, "produced_by": "prior_seasons"},
                       "g2": {"split": "dev", "game_date_et": "2025-01-01", "p_boost": 0.4, "produced_by": "train_fit"}}
        excluded = {"g0": {"reason": "fewer_than_min_box_scores"}}
        rows = bam.attach_boosting(self.rows(), self.history, self.splits, predictions, {"g2": {"x": 1.0}}, excluded)
        self.assertEqual([r["split"] for r in rows], ["train", "train", "dev", "test"])
        self.assertEqual([r["p_boost"] for r in rows], [None, 0.6, 0.4, None])
        self.assertEqual([r["boosting_produced_by"] for r in rows], [None, "prior_seasons", "train_fit", None])
        self.assertEqual(rows[0]["forecast_ineligibility_reason"], "fewer_than_min_box_scores")
        self.assertEqual(rows[3]["forecast_ineligibility_reason"], "no_out_of_sample_boosting_prediction")
        self.assertEqual(rows[2]["features"], {"x": 1.0})

    def test_prediction_from_another_split_or_unconfigured_season_rejected(self):
        wrong = {"g2": {"split": "train", "game_date_et": "2025-01-01", "p_boost": 0.4, "produced_by": "prior_seasons"}}
        with self.assertRaisesRegex(ValueError, "split"):
            bam.attach_boosting(self.rows(), self.history, self.splits, wrong, {}, {})
        # A DEV game must be forecast by the TRAIN-fitted model, not a TRAIN-season model.
        wrong = {"g2": {"split": "dev", "game_date_et": "2025-01-01", "p_boost": 0.4, "produced_by": "prior_seasons"}}
        with self.assertRaisesRegex(ValueError, "split"):
            bam.attach_boosting(self.rows(), self.history, self.splits, wrong, {}, {})
        with self.assertRaisesRegex(ValueError, "no configured split"):
            bam.attach_boosting(self.rows(), self.history, {2015: "train"}, {}, {}, {})


class SnapshotTests(unittest.TestCase):
    def fixture(self, directory):
        directory = Path(directory)
        raw = b"raw bytes"
        sha = bam.digest(raw)
        page = {"page": 1, "text": " heading\ncolumns\r\n\nlast", "text_sha256": bam.digest(b" heading\ncolumns\r\n\nlast")}
        pages = bam.pages_with_lines([page])
        bam.write_json(directory / "sources/a.json", {"source_sha256": sha, "pages": pages})
        (directory / "raw").mkdir()
        (directory / "raw/a.bin").write_bytes(raw)
        row = {**{k: v for k, v in GAME.items() if k in bam.METADATA}, "split": "train", "season": 2023,
               "scheduled_start_utc": "2024-01-02T01:00:00Z", "forecast_eligible": False, "p_boost": None,
               "boosting_produced_by": None, "sources": [{"source_id": "sha256:" + sha, "sha256": sha, "raw_path": "raw/a.bin",
                            "pages_path": "sources/a.json", "pages_sha256": bam.digest((directory / "sources/a.json").read_bytes()),
                            "active_for_research": True, "archive_time": "2024-01-01T12:00:00Z"}]}
        bam.write_rows(directory / "inputs.jsonl", [row])
        bam.write_rows(directory / "labels.jsonl", [{"game_id": row["game_id"], "home_win": 1}])
        manifest = {"snapshot_id": "test", "season_splits": {"2023": "train", "2024": "dev", "2025": "test"},
                    "prediction_model_cutoff_date": "2024-09-25", "boosting_fit_game_ids": [], "summary": {"games": 1},
                    "artifact_sha256": {str(p.relative_to(directory)): bam.digest(p.read_bytes()) for p in directory.rglob("*") if p.is_file()}}
        bam.write_json(directory / "manifest.json", manifest)
        return row

    def test_exact_page_line_reconstruction_and_outcome_blind_load(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.fixture(directory)
            self.assertEqual(bam.validate_snapshot(directory), {"games": 1})
            # Loader reads no label or postmortem file.
            (Path(directory) / "labels.jsonl").unlink()
            game = bam.load_snapshot_game(directory, row["game_id"])
            self.assertNotIn("home_win", game)
            page = game["sources"][0]["pages"][0]
            self.assertEqual("".join(line["text"] for line in page["lines"]), page["text"])
            self.assertEqual([line["line"] for line in page["lines"]], [1, 2, 3, 4])

    def test_source_and_artifact_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            (Path(directory) / "raw/a.bin").write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "checksum"):
                bam.validate_snapshot(directory)

    def test_cutoff_and_outcome_leakage_rejected_even_with_new_artifact_hash(self):
        for change in ({"home_win": 1}, {"cutoff_utc": "2024-01-02T00:01:00Z"}, {"split": "dev"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                row = self.fixture(directory)
                row.update(change)
                path = Path(directory)
                bam.write_rows(path / "inputs.jsonl", [row])
                manifest = json.loads((path / "manifest.json").read_text())
                manifest["artifact_sha256"]["inputs.jsonl"] = bam.digest((path / "inputs.jsonl").read_bytes())
                bam.write_json(path / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    bam.validate_snapshot(directory)


class SharedNotesTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        old = root / "data/processed/bam_pipeline/bam_notes_v2"
        old.mkdir(parents=True)
        row = SnapshotTests().fixture(old)
        raw_path = root / "data/raw/notes/example.pdf"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes((old / "raw/a.bin").read_bytes())
        row.update(features={"x": 1.25}, forecast_eligible=True, p_boost=0.4,
                   boosting_produced_by="prior_seasons", forecast_ineligibility_reason=None)
        row["sources"][0].update(original_raw_path="data/raw/notes/example.pdf", source_kind="pdf")
        bam.write_rows(old / "inputs.jsonl", [row])
        manifest = json.loads((old / "manifest.json").read_text())
        manifest.update(identity={"config": {}}, artifact_sha256={
            str(path.relative_to(old)): bam.digest(path.read_bytes())
            for path in old.rglob("*") if path.is_file() and path.name != "manifest.json"})
        bam.write_json(old / "manifest.json", manifest)
        reviewed = root / "data/processed/notes_v2"
        reviewed.mkdir()
        for name in ("candidates", "documents", "approved", "html_posts"):
            bam.write_rows(reviewed / (name + ".jsonl"), [])
        bam.write_json(reviewed / "review_summary.json", {})
        stage = root / "stage"
        stage.mkdir()
        bam.migrate_notes(root, stage)
        dataset = Mock()
        dataset.inputs.return_value = [{"game_id": row["game_id"], "features": row["features"]}]
        dataset.predictions.return_value = [{**{key: row[key] for key in ("game_id", "season", "game_date_et", "split", "p_boost")},
                                            "produced_by": row["boosting_produced_by"]}]
        dataset.labels.return_value = [{"game_id": row["game_id"], "home_win": 1}]
        return stage, row, dataset

    def test_shared_load_preserves_citations_without_labels_or_raw_copies(self):
        with tempfile.TemporaryDirectory() as root:
            stage, row, dataset = self.fixture(root)
            with patch("nba_dataset.Dataset", return_value=dataset):
                loaded = bam.load_snapshot_game(stage, row["game_id"], root=root)
                self.assertEqual(loaded["features"], {"x": 1.25})
                self.assertEqual(loaded["p_boost"], 0.4)
                self.assertNotIn("home_win", loaded)
                dataset.labels.assert_not_called()
                page = loaded["sources"][0]["pages"][0]
                self.assertEqual("".join(line["text"] for line in page["lines"]), page["text"])
                self.assertNotIn("plain_text", page)  # Arrow optional fields must not alter packet identity.
                self.assertFalse((stage / "raw").exists())
                self.assertFalse((stage / "inputs.jsonl").exists())
                self.assertEqual(bam.read_snapshot_inputs(stage, root=root, legacy_paths=True), [row])
                self.assertEqual(bam.validate_snapshot(stage, root=root, label_splits=()), {"games": 1})
                dataset.labels.assert_not_called()
                dataset.history.assert_not_called()
                dataset.box_scores.assert_not_called()
                self.assertEqual(bam.validate_snapshot(stage, root=root, label_splits=("train",)), {"games": 1})
                dataset.labels.assert_called_once_with(("train",), eligible_only=False)
                self.assertEqual(bam.validate_snapshot(stage, root=root), {"games": 1})

    def test_shared_pages_and_original_source_mutations_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            stage, row, dataset = self.fixture(root)
            with patch("nba_dataset.Dataset", return_value=dataset):
                (Path(root) / row["sources"][0]["original_raw_path"]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source checksum"):
                    bam.validate_snapshot(stage, root=root)
                with (stage / "note_pages.parquet").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    bam.load_snapshot_game(stage, row["game_id"], root=root)

    def test_postgame_opens_only_train_labels_and_train_history(self):
        import bam_postgame
        with tempfile.TemporaryDirectory() as root:
            stage, row, dataset = self.fixture(root)
            dataset.history.side_effect = RuntimeError("stop after TRAIN history boundary")
            with patch("nba_dataset.Dataset", return_value=dataset), patch.object(bam_postgame, "ROOT", Path(root)):
                with self.assertRaisesRegex(RuntimeError, "TRAIN history boundary"):
                    bam_postgame.prepare(stage, Path(root) / "teacher")
            dataset.labels.assert_called_once_with(("train",), eligible_only=False)
            dataset.history.assert_called_once_with(seasons=[2023])
            dataset.box_scores.assert_not_called()

    def test_cleanup_coverage_accounts_for_all_files_and_rejects_unknowns(self):
        with tempfile.TemporaryDirectory() as root:
            stage, row, dataset = self.fixture(root)
            root = Path(root)
            manifest = json.loads((stage / "bam_manifest.json").read_text())
            originals = [root / "data/processed/bam_pipeline/bam_notes_v2", root / "data/processed/notes_v2"]
            expected = {str(path.relative_to(root)) for directory in originals
                        for path in directory.rglob("*") if path.is_file()}
            self.assertEqual(set(manifest["cleanup_coverage"]), expected)
            self.assertEqual(set(manifest["source_sha256"]), expected)
            (originals[1] / "unmapped.csv").write_text("important,new,data\\n")
            old_hash = bam.digest((stage / "notes.parquet").read_bytes())
            with self.assertRaisesRegex(ValueError, "Unmapped notes cleanup"):
                bam.migrate_notes(root, stage)
            self.assertEqual(bam.digest((stage / "notes.parquet").read_bytes()), old_hash)

    def test_other_model_forecasts_do_not_change_bam_but_changed_boosting_is_rejected(self):
        from nba_dataset import write_table, write_json, SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as root:
            stage, row, dataset = self.fixture(root)
            forecast = {**dataset.predictions.return_value[0], "model_id": "gradient_boosting", "stage": "baseline",
                        "run_id": "frozen", "prediction_index": 0, "p_home_win": 0.4}
            forecast.pop("p_boost")
            metadata = write_table(stage / "predictions.parquet", [forecast])
            manifest = {"schema_version": SCHEMA_VERSION, "tables": {"predictions": metadata}}
            write_json(stage / "manifest.json", manifest)
            bam._verify_shared_files(stage, root=root)
            other = {**forecast, "model_id": "new_model", "p_home_win": 0.9}
            manifest["tables"]["predictions"] = write_table(stage / "predictions.parquet", [forecast, other])
            write_json(stage / "manifest.json", manifest)
            bam._verify_shared_files(stage, root=root)
            manifest["tables"]["predictions"] = write_table(stage / "predictions.parquet", [dict(forecast, p_home_win=0.7), other])
            write_json(stage / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "gradient forecast"):
                bam._verify_shared_files(stage, root=root)


if __name__ == "__main__":
    unittest.main()
