"""Regression checks for article revision leakage and page-wide text pollution."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_narrative_sources import parse_article, article_checks
from collect_pilot import stamp


class NarrativeSourceTests(unittest.TestCase):
    def setUp(self):
        self.url = "https://www.nba.com/knicks/news/example"
        self.game = {"game_date_et": "2025-01-21", "home_team": "Brooklyn Nets", "away_team": "New York Knicks"}
        self.cutoff = stamp("2025-01-21T23:30:00Z")
        self.article = {"title": "Knicks vs Nets January 21, 2025", "cms_title": "Knicks vs Nets January 21, 2025",
                        "published": "2025-01-21T18:00:00Z", "modified": "2025-01-21T18:00:00Z",
                        "cms_published": "2025-01-21T18:00:00Z", "cms_modified": "2025-01-21T18:00:00Z",
                        "url": self.url, "cms_url": self.url, "opening": "Knicks visit the Nets.",
                        "header": "Knicks visit the Nets.", "text": "Pregame article."}

    def test_postgame_revision_disqualifies_pregame_publication(self):
        self.article["modified"] = self.article["cms_modified"] = "2025-03-31T15:15:22Z"
        self.assertIn("after_cutoff", article_checks(self.article, self.game, self.cutoff, self.url))

    def test_conflicting_cms_timestamp_is_not_ignored(self):
        self.article["cms_modified"] = "2025-01-22T03:00:00Z"
        reasons = article_checks(self.article, self.game, self.cutoff, self.url)
        self.assertIn("timestamp_metadata_disagrees", reasons)
        self.assertIn("after_cutoff", reasons)

    def test_wrong_opening_matchup_disqualifies_matching_title(self):
        self.article["opening"] = "The Knicks host the Atlanta Hawks."
        self.assertIn("home_team_missing_from_title_or_opening",
                      article_checks(self.article, self.game, self.cutoff, self.url))

    def test_missing_timezone_is_not_accepted(self):
        self.article["modified"] = "2025-01-21T18:00:00"
        self.assertIn("missing_or_invalid_modified", article_checks(self.article, self.game, self.cutoff, self.url))

    def test_pre_cutoff_metadata_passes_only_automated_checks(self):
        self.assertEqual(article_checks(self.article, self.game, self.cutoff, self.url), [])

    def test_only_article_blocks_are_extracted(self):
        ld = {"@type": "NewsArticle", "headline": self.article["title"],
              "mainEntityOfPage": {"@id": self.url}}
        page = {"type": "article", "title": self.article["title"], "permalink": self.url,
                "contentStructured": [{"type": "paragraph", "html": "<p>Pregame text.</p>"},
                                      {"type": "list", "html": "<ul><li>One.</li><li>Two.</li></ul>"}],
                "related": [{"text": "Target team won the game."}]}
        hydration = {"props": {"pageProps": {"pageObject": page, "layout": {"text": "Current news."}}}}
        body = ('<nav>Current news.</nav><script type="application/ld+json">' + json.dumps(ld)
                + '</script><script id="__NEXT_DATA__">' + json.dumps(hydration) + '</script>').encode()
        article = parse_article(body)
        self.assertEqual(article["text"], "Pregame text.\n\nOne. Two.")
        self.assertNotIn("Target team", article["text"])


if __name__ == "__main__":
    unittest.main()
