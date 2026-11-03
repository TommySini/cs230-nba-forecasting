"""Check full-document coverage, source citations and research input boundaries."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_full_game_notes import digest, research_prompt, select_page_text, split_page, validate_citations, validate_page_coverage


class FullGameNotesTests(unittest.TestCase):
    def test_sparse_layout_cannot_discard_text_from_plain_extraction(self):
        plain = "HEADER\n" + "Full rotated table and player information. " * 40
        selected, method, flags = select_page_text(plain, "HEADER\n" + "\n" * 200)
        self.assertEqual(selected, plain)
        self.assertEqual(method, "pypdf_plain_layout_omitted_text")
        self.assertIn("layout_omitted_text_recovered_from_plain", flags)

    def test_reviewed_transcription_recovers_garbled_page_without_destroying_original_inputs(self):
        plain, layout = "3ULRU\x03WR", "3ULRU\x03WR"
        text, method, flags = select_page_text(plain, layout, {"text": "Prior to signing with the team, the player appeared in 34 games."})
        self.assertIn("Prior to signing", text)
        self.assertEqual(method, "reviewed_visual_transcription")
        self.assertNotIn("suspicious_glyphs_need_visual_review", flags)
        self.assertIn("\x03", plain)

    def example(self):
        texts = ["First page.\n" * 9, "", "Last page has a late-packet injury update."]
        pages = [{"page": i, "text": t, "text_sha256": digest(t.encode())} for i, t in enumerate(texts, 1)]
        chunks = [{"page": p["page"], **c} for p in pages for c in split_page(p["text"], 25)]
        return pages, chunks

    def test_includes_last_page_blank_page_and_every_character(self):
        pages, chunks = self.example()
        validate_page_coverage(pages, chunks, 3)
        self.assertEqual("".join(c["text"] for c in chunks), "".join(p["text"] for p in pages))

    def test_missing_late_page_rejected(self):
        pages, chunks = self.example()
        with self.assertRaises(ValueError):
            validate_page_coverage(pages[:-1], chunks, 3)

    def test_truncation_overlap_and_changed_text_rejected(self):
        pages, chunks = self.example()
        variants = [chunks[:-1], [chunks[0], *chunks], copy.deepcopy(chunks)]
        variants[-1][0]["text"] = "fabricated source"
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_page_coverage(pages, variant, 3)

    def test_research_prompt_excludes_labels_booster_and_other_unselected_fields(self):
        meta = {"away_abbr": "MIL", "home_abbr": "CLE", "game_date_et": "2024-11-04",
                "publisher": "CLE", "cutoff_utc": "2024-11-04T23:00:00Z", "home_win": "PRIVATE_OUTCOME",
                "p_boost": "PRIVATE_PROBABILITY", "raw_espn": "PRIVATE_RECAP"}
        prompt = research_prompt(meta, {"page": 20, "start": 0, "end": 19, "text": "Player availability"})
        self.assertIn("page 20", prompt)
        self.assertIn("Player availability", prompt)
        self.assertNotIn("PRIVATE_", prompt)

    def test_citations_require_correct_source_page_and_verbatim_evidence(self):
        packet = {"source": {"sha256": "source-hash"}, "pages": [{"page": 9, "text": "Player A missed\nthe previous game."}]}
        cite = {"source_sha256": "source-hash", "page": 9, "quote": "Player A missed the previous game."}
        validate_citations(packet, [{"citations": [cite]}])
        for changed in ({**cite, "page": 1}, {**cite, "quote": "Player A is out tonight."}, {**cite, "source_sha256": "other"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_citations(packet, [{"citations": [changed]}])
        with self.assertRaises(ValueError):
            validate_citations(packet, [{"claim": "Uncited fact"}])


if __name__ == "__main__":
    unittest.main()
