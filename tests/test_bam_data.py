"""Temporal boundaries and immutable source/page snapshots for the new pipeline."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bam_data as bam


def note(gid="espn:1", sha="a" * 64, capture="20240101120000", url="https://nba.com/notes.pdf"):
    time = datetime.strptime(capture, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return {"game_id": gid, "game_date_et": "2024-01-01", "cutoff_utc": "2024-01-02T00:00:00Z",
            "home_abbr": "BOS", "away_abbr": "DEN", "team": "BOS", "publisher_url": url,
            "review": {"decision": "approved_template_checked_excerpt", "approved_pdf_pages": [1]},
            "source": {"sha256": sha, "raw_path": "data/raw/source.pdf", "status": 200,
                       "final_url": "https://web.archive.org/web/" + capture + "id_/" + url,
                       "memento_datetime": format_datetime(time, usegmt=True)}}


class MergeTests(unittest.TestCase):
    def test_multi_packet_dedup_and_versions_are_order_independent(self):
        first = note()
        duplicate = deepcopy(first)
        duplicate["source"]["raw_path"] = "data/raw/duplicate.pdf"
        later = note(sha="b" * 64, capture="20240101130000")
        opponent = note(sha="c" * 64, url="https://nba.com/denver.pdf")
        opponent["team"] = "DEN"
        rows = [first, duplicate, later, opponent]
        game = bam.merge_notes(rows)[0]
        self.assertEqual(bam.merge_notes(rows), bam.merge_notes(list(reversed(rows))))
        self.assertEqual(len(game["sources"]), 3)
        self.assertEqual(sum(s["active_for_research"] for s in game["sources"]), 2)
        self.assertFalse(next(s for s in game["sources"] if s["sha256"] == "a" * 64)["active_for_research"])
        self.assertEqual(next(s for s in game["sources"] if s["sha256"] == "a" * 64)["source_version_count"], 2)

    def test_conflicting_game_and_capture_rejected(self):
        first = note()
        mismatch = deepcopy(first)
        mismatch["away_abbr"] = "CLE"
        with self.assertRaisesRegex(ValueError, "metadata"):
            bam.merge_notes([first, mismatch])
        with self.assertRaisesRegex(ValueError, "archive capture"):
            bam.merge_notes([first, note(sha="b" * 64)])

    def test_repeated_identical_bytes_preserve_latest_capture_chronology(self):
        notes = [note(capture="20240101120000"),
                 note(sha="b" * 64, capture="20240101130000"),
                 note(capture="20240101140000")]
        game = bam.merge_notes(notes)[0]
        latest = next(s for s in game["sources"] if s["active_for_research"])
        self.assertEqual(latest["sha256"], "a" * 64)
        self.assertEqual(latest["archive_time"], "2024-01-01T14:00:00Z")
        self.assertEqual(latest["aliases"][0]["archive_time"], "2024-01-01T12:00:00Z")
        self.assertEqual(bam.merge_notes(notes), bam.merge_notes(list(reversed(notes))))

    def test_conflicting_capture_cannot_hide_in_duplicate_hash_alias(self):
        with self.assertRaisesRegex(ValueError, "archive capture"):
            bam.merge_notes([note(capture="20240101120000"), note(capture="20240101140000"),
                             note(sha="b" * 64, capture="20240101140000")])

    def test_version_selection_checks_all_publisher_url_aliases(self):
        notes = [note(url="https://nba.com/a.pdf", capture="20240101140000"),
                 note(url="https://nba.com/b.pdf", capture="20240101120000"),
                 note(sha="b" * 64, url="https://nba.com/b.pdf", capture="20240101130000")]
        game = bam.merge_notes(notes)[0]
        self.assertEqual(sum(s["active_for_research"] for s in game["sources"]), 2)
        source = next(s for s in game["sources"] if s["sha256"] == "a" * 64)
        group = next(g for g in source["source_version_groups"] if g["url"].endswith("/b.pdf"))
        self.assertEqual(group["selected_sha256"], "b" * 64)

    def test_verified_bytes_exact_hash_and_pre_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data/raw/source.pdf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"exact archived bytes")
            source = note(sha=bam.digest(path.read_bytes()))
            self.assertEqual(bam.verified_bytes(source, root), path.read_bytes())
            source["cutoff_utc"] = "2024-01-01T11:59:59Z"
            with self.assertRaisesRegex(ValueError, "cutoff"):
                bam.verified_bytes(source, root)
            source["cutoff_utc"] = "2024-01-02T00:00:00Z"
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                bam.verified_bytes(source, root)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"game_id": str(i), "game_date_et": date, "forecast_eligible": True}
                     for i, date in enumerate(["2024-01-01", "2024-10-01", "2025-04-01", "2025-10-01", "2025-10-01", "2026-01-01", "2026-02-01"])]
        self.policy = {"prediction_model_cutoff_date": "2024-09-18", "dev_start_date": "2025-10-01",
                       "test_start_date": "2026-02-01", "minimum_split_games": {"train": 1, "dev": 1, "test": 1}}

    def test_old_and_booster_fit_games_train_only_whole_dates(self):
        result = bam.assign_splits(self.rows, {"1", "2"}, self.policy)
        self.assertEqual([r["split"] for r in result], ["train"] * 3 + ["dev"] * 3 + ["test"])
        with self.assertRaisesRegex(ValueError, "boundaries"):
            bam.assign_splits(self.rows, {"3"}, self.policy)
        self.rows[-1]["forecast_eligible"] = False
        with self.assertRaisesRegex(ValueError, "boundaries"):
            bam.assign_splits(self.rows, set(), self.policy)

    def test_automatic_split_is_chronological(self):
        policy = {k: v for k, v in self.policy.items() if k not in ("dev_start_date", "test_start_date")}
        result = bam.assign_splits(self.rows, {"2"}, policy)
        self.assertEqual([r["split"] for r in result[:3]], ["train"] * 3)
        self.assertEqual(result[3]["split"], result[4]["split"])

    def test_monthly_booster_membership_and_research_only_games(self):
        numeric = [{"game_id": "jan1", "game_date_et": "2024-01-01"},
                   {"game_id": "jan2", "game_date_et": "2024-01-02"},
                   {"game_id": "feb", "game_date_et": "2024-02-02"},
                   {"game_id": "future", "game_date_et": "2024-03-02"}]
        targets = numeric + [{"game_id": "warmup", "game_date_et": "2023-10-24"}]
        plans, selected = bam.prior_probability_plan(targets, numeric, {"jan1", "jan2", "feb"}, 2)
        self.assertEqual(plans["2024-02-01"]["fit_game_ids"], ["jan1", "jan2"])
        self.assertEqual(selected["feb"]["fold"], "2024-02-01")
        self.assertEqual(selected["future"]["fold"], "original_frozen")
        self.assertIsNone(selected["jan2"]["fold"])
        self.assertEqual(selected["warmup"]["reason"], "baseline_warmup")


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
        row = {**{k: v for k, v in note().items() if k in bam.METADATA}, "split": "train",
               "scheduled_start_utc": "2024-01-02T01:00:00Z", "forecast_eligible": False, "p_boost": None,
               "sources": [{"source_id": "sha256:" + sha, "sha256": sha, "raw_path": "raw/a.bin",
                            "pages_path": "sources/a.json", "pages_sha256": bam.digest((directory / "sources/a.json").read_bytes()),
                            "active_for_research": True, "archive_time": "2024-01-01T12:00:00Z"}]}
        bam.write_rows(directory / "inputs.jsonl", [row])
        bam.write_rows(directory / "labels.jsonl", [{"game_id": row["game_id"], "home_win": 1}])
        manifest = {"snapshot_id": "test", "policy": {"prediction_model_cutoff_date": "2024-09-18"},
                    "original_fit_game_ids": [], "summary": {"games": 1},
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
        for change in ({"home_win": 1}, {"cutoff_utc": "2024-01-02T00:01:00Z"}):
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
