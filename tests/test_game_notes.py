import gzip
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_game_notes import archive_time_verified, decode_archive_body


class GameNoteArchiveTests(unittest.TestCase):
    def setUp(self):
        self.url = "https://example.org/notes.pdf"
        self.cutoff = "2025-02-21T23:00:00Z"
        self.source = {"status": 200, "final_url": "https://web.archive.org/web/20250221213743id_/" + self.url,
                       "memento_datetime": "Fri, 21 Feb 2025 21:37:43 GMT"}

    def test_actual_replay_timestamp_is_verified(self):
        self.assertTrue(archive_time_verified(self.source, self.url, self.cutoff))

    def test_redirect_to_future_or_other_document_is_rejected(self):
        future = dict(self.source, final_url="https://web.archive.org/web/20250222010000id_/" + self.url,
                      memento_datetime="Sat, 22 Feb 2025 01:00:00 GMT")
        self.assertFalse(archive_time_verified(future, self.url, self.cutoff))
        other = dict(self.source, final_url=self.source["final_url"] + ".different")
        self.assertFalse(archive_time_verified(other, self.url, self.cutoff))

    def test_missing_or_conflicting_capture_evidence_is_rejected(self):
        for value in (None, "invalid", "Fri, 21 Feb 2025 22:37:43 GMT"):
            self.assertFalse(archive_time_verified(dict(self.source, memento_datetime=value), self.url, self.cutoff))
        bad = dict(self.source, final_url="https://web.archive.org/web/20251321213743id_/" + self.url)
        self.assertFalse(archive_time_verified(bad, self.url, self.cutoff))

    def test_gzip_extraction_does_not_replace_preserved_source_bytes(self):
        body = b"<html>Archived article only</html>"
        packed = gzip.compress(body)
        self.assertEqual(decode_archive_body(packed), body)
        self.assertTrue(packed.startswith(b"\x1f\x8b"))
        self.assertEqual(decode_archive_body(body), body)


if __name__ == "__main__":
    unittest.main()
