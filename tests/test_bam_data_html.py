from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bam_data import canonical, digest
from bam_data_html import sections_as_pages, verify_content_review


class HtmlCitationBoundaryTests(unittest.TestCase):
    def section(self, number=1):
        text = "Heading\nExact source text.\n"
        return {"section": number, "label": "paragraph", "text": text, "text_sha256": digest(text.encode()),
                "lines": [{"line": 1, "text": "Heading\n"}, {"line": 2, "text": "Exact source text.\n"}]}

    def test_article_citation_units_are_explicit_sections_not_pdf_pages(self):
        pages = sections_as_pages([self.section(), self.section(2)])
        self.assertEqual([p["section"] for p in pages], [1, 2])
        self.assertTrue(all(p["page_kind"] == "article_section" for p in pages))
        self.assertTrue(all("PDF" not in p["method"] for p in pages))
        self.assertEqual("".join(line["text"] for line in pages[0]["lines"]), pages[0]["text"])

    def test_reordered_or_changed_source_text_fails_closed(self):
        with self.assertRaises(ValueError):
            sections_as_pages([self.section(2)])
        changed = self.section()
        changed["text"] += "Hidden extra statement"
        with self.assertRaises(ValueError):
            sections_as_pages([changed])
        changed = self.section()
        changed["lines"][1]["text"] = "Different quote.\n"
        with self.assertRaises(ValueError):
            sections_as_pages([changed])

    def test_prior_game_heading_context_keeps_original_citation_ids(self):
        sections = [self.section(), self.section(2), self.section(3)]
        sections[0]["label"] = "title"
        sections[1]["label"] = "heading"
        pages = sections_as_pages(sections)
        self.assertEqual([item["section"] for item in pages[2]["context_sections"]], [1, 2])
        self.assertEqual(pages[2]["context_sections"][1]["lines"], sections[1]["lines"])
        self.assertEqual(pages[2]["lines"], sections[2]["lines"])
        self.assertNotIn("context_sections", sections_as_pages(sections, with_context=False)[2])

    def test_quarantined_or_partial_review_cannot_be_claimed_as_approved_manifest(self):
        sections = [self.section(), self.section(2)]
        good = {"decision": "approved_source_for_pregame_research", "game_id": "fixture:1", "source_sha256": "a" * 64,
                "sections_sha256": digest(canonical(sections)), "reviewed_sections": [1, 2]}
        verify_content_review(good, "fixture:1", "a" * 64, sections)
        for change in ({"decision": "quarantined_source"}, {"reviewed_sections": [1]},
                       {"game_id": "fixture:2"}, {"sections_sha256": "b" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_content_review({**good, **change}, "fixture:1", "a" * 64, sections)


if __name__ == "__main__":
    unittest.main()
