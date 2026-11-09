"""Temporal boundaries and immutable source/page snapshots for the new pipeline."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
