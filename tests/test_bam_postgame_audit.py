from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bam_postgame_audit import audit_recap_fold_times


class RecapFoldTimeTests(unittest.TestCase):
    def fixture(self):
        inputs = [{"game_id": "a", "game_date_et": "2024-01-01", "cutoff_utc": "2024-01-01T23:00:00Z", "split": "train", "forecast_eligible": False},
                  {"game_id": "b", "game_date_et": "2024-01-01", "cutoff_utc": "2024-01-01T22:00:00Z", "split": "train"},
                  {"game_id": "c", "game_date_et": "2024-01-02", "cutoff_utc": "2024-01-02T21:00:00Z", "split": "train"},
                  {"game_id": "d", "game_date_et": "2024-01-03", "cutoff_utc": "2024-01-03T23:00:00Z", "split": "train"},
                  {"game_id": "e", "game_date_et": "2024-01-04", "cutoff_utc": "2024-01-04T23:00:00Z", "split": "dev"}]
        source = {"source_id": "recap:a", "content_types": ["recap"], "retrieved_at_utc": "2026-11-01T00:00:00Z",
                  "pages": [{"page": 1, "lines": [
                      {"line": 1, "source_pointer": "/article/published", "text": "Recap published: 2024-01-02T02:00:00Z"},
                      {"line": 2, "source_pointer": "/article/lastModified", "text": "Recap lastModified: 2024-01-02T21:00:00Z"}]}]}
        records = [{"game_id": r["game_id"], "split": "train", "normalized": True,
                    "sources": [source] if r["game_id"] == "a" else []} for r in inputs if r["split"] == "train"]
        return inputs, records

    def test_whole_date_research_folds_include_research_only_and_ignore_retrieval_time(self):
        inputs, records = self.fixture()
        audit = audit_recap_fold_times(inputs, records, folds=3)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual([f["first_prediction_date"] for f in audit["folds"]],
                         ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
        self.assertEqual([f["training_games"] for f in audit["folds"]], [0, 2, 3, 4])
        self.assertEqual(audit["folds"][0]["first_prediction_cutoff"], "2024-01-01T22:00:00+00:00")
        self.assertIn("not archive proof", audit["limitation"])

    def test_late_modified_and_missing_published_fail_earlier_fold(self):
        inputs, records = self.fixture()
        lines = records[0]["sources"][0]["pages"][0]["lines"]
        lines.pop(0)
        lines[0]["text"] = "Recap lastModified: 2024-01-02T21:00:01Z"
        audit = audit_recap_fold_times(inputs, records)
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["folds"][1]["missing_date_fields"][0]["field"], "published")
        self.assertEqual(audit["folds"][1]["post_boundary_recap_fields"][0]["field"], "lastModified")
        self.assertEqual(audit["folds"][2]["post_boundary_recap_fields"], [])
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            audit_recap_fold_times(inputs, [{**records[0], "split": "test"}])


if __name__ == "__main__":
    unittest.main()
