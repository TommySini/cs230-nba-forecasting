from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bam_postgame import normalize_document, normalize_optional_document, text_only


class PostgameTests(unittest.TestCase):
    def setUp(self):
        self.game = {"game_id": "espn:1", "split": "train", "home_abbr": "BOS", "away_abbr": "DEN",
                     "home_score": 100, "away_score": 99, "scheduled_start_utc": "2024-01-02T01:00:00Z",
                     "cutoff_utc": "2024-01-02T00:00:00Z"}
        self.header = {"id": "1", "competitions": [{"date": "2024-01-02T01:00:00Z", "status": {"type": {"completed": True}},
                       "competitors": [{"homeAway": "home", "score": "100", "team": {"abbreviation": "BOS"}},
                                       {"homeAway": "away", "score": "99", "team": {"abbreviation": "DEN"}}]}]}
        self.document = {"header": self.header, "news": "CURRENT NEWS NEVER INCLUDE", "winprobability": "LEAKED MODEL",
                         "article": {"type": "Recap", "gameId": "1", "story": "<p>Boston won.</p><script>TRACKER</script>"},
                         "boxscore": {"teams": [{"team": {"abbreviation": "BOS"}, "statistics": [{"label": "FG", "displayValue": "40-80"}]}],
                                      "players": [{"team": {"abbreviation": "BOS"}, "statistics": [{"labels": ["PTS"],
                                                   "athletes": [{"athlete": {"displayName": "Player Name"}, "stats": ["25"], "starter": True}]}]}]}}
        self.meta = {"status": 200, "sha256": "a" * 64, "url": "https://espn.com/api", "retrieved_at_utc": "2024-01-02T05:00:00Z"}

    def test_whitelisted_fields_and_stable_citations_only(self):
        result = normalize_document(self.game, self.document, self.meta, "summary")
        self.assertEqual(result["content_types"], ["final_score", "schedule", "recap", "boxscore"])
        text = json.dumps(result)
        self.assertIn("Player Name PTS: 25", text)
        self.assertIn("Boston won.", text)
        self.assertNotIn("CURRENT NEWS", text)
        self.assertNotIn("LEAKED MODEL", text)
        self.assertNotIn("TRACKER", text)
        lines = result["pages"][0]["lines"]
        self.assertEqual([line["line"] for line in lines], list(range(1, len(lines) + 1)))
        self.assertTrue(all(line["source_pointer"].startswith("/") for line in lines))

    def test_nontraining_game_and_mismatched_results_rejected(self):
        for change in ({"split": "dev"}, {"split": "test"}, {"game_id": "espn:2"},
                       {"home_score": 101}, {"home_abbr": "CLE"}, {"scheduled_start_utc": "2024-01-02T02:00:00Z"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_document({**self.game, **change}, self.document, self.meta, "summary")

    def test_incomplete_game_and_misaligned_table_rejected(self):
        incomplete = deepcopy(self.document)
        incomplete["header"]["competitions"][0]["status"]["type"]["completed"] = False
        with self.assertRaisesRegex(ValueError, "completed"):
            normalize_document(self.game, incomplete, self.meta, "summary")
        broken = deepcopy(self.document)
        broken["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["stats"] = []
        with self.assertRaisesRegex(ValueError, "align"):
            normalize_document(self.game, broken, self.meta, "summary")

    def test_other_game_article_is_never_copied(self):
        self.document["article"]["gameId"] = "2"
        result = normalize_document(self.game, self.document, self.meta, "summary")
        self.assertNotIn("recap", result["content_types"])
        self.assertNotIn("Boston won.", json.dumps(result))

    def test_html_paragraphs_preserved(self):
        self.assertEqual(text_only("<p>One &amp; two.</p><p>Three.</p>"), "One & two.\nThree.")

    def test_espn_dnp_empty_vector_preserves_status_without_fake_zeros(self):
        # Shape reproduced from the first fetched 2023-10-24 ESPN summary.
        row = {"athlete": {"displayName": "Justin Holiday"}, "starter": False,
               "didNotPlay": True, "reason": "COACH'S DECISION", "stats": []}
        group = self.document["boxscore"]["players"][0]["statistics"][0]
        group["labels"] = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
        group["athletes"] = [row]
        source, error = normalize_optional_document(self.game, json.dumps(self.document), self.meta, "summary")
        self.assertIsNone(error)
        text = "\n".join(line["text"] for line in source["pages"][0]["lines"])
        self.assertIn("Justin Holiday didNotPlay: True", text)
        self.assertIn("Justin Holiday reason: COACH'S DECISION", text)
        self.assertNotIn("Justin Holiday PTS", text)
        self.assertNotIn("Justin Holiday MIN", text)

    def test_malformed_played_stats_quarantines_whole_optional_source(self):
        group = self.document["boxscore"]["players"][0]["statistics"][0]
        group["athletes"][0]["stats"] = []
        source, error = normalize_optional_document(self.game, json.dumps(self.document), self.meta, "summary")
        self.assertIsNone(source)
        self.assertEqual(error["decision"], "quarantined_not_teacher_input")
        self.assertIn("align", error["reason"])
        self.assertEqual(error["sha256"], self.meta["sha256"])
        # The independent checked scoreboard remains usable for this game.
        scoreboard = {"events": [self.header]}
        fallback = normalize_document(self.game, scoreboard, self.meta, "scoreboard")
        self.assertEqual(fallback["content_types"], ["final_score", "schedule"])


if __name__ == "__main__":
    unittest.main()
