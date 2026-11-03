import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bam_html_notes import parse_html_note
from bam_html_por import parse_por_html_note


class PortlandArticleTests(unittest.TestCase):
    def fixture(self, change=None, heading=True, capture="20240325190000", modified="2024-03-25T18:00:00Z"):
        url = "https://www.nba.com/blazers/news/game-notes-portland-trail-blazers-at-houston-rockets"
        game = {"game_id": "fixture:por-hou", "home_abbr": "HOU", "away_abbr": "POR", "season": 2023,
                "game_date_et": "2024-03-25", "scheduled_start_utc": "2024-03-26T00:00:00Z", "cutoff_utc": "2024-03-25T23:00:00Z"}
        opening = "Portland Trail Blazers (19-52, 8-25 away) @ Houston Rockets (35-35, 25-11 home)\nMonday, Mar. 25, 2024 • 5:00 pm PST • Toyota Center"
        if change:
            opening = opening.replace(*change)
        page = {"type": "article", "title": "Game Notes: Trail Blazers (19-52) at Rockets (35-35)", "permalink": url,
                "date": "2024-03-25T18:00:00Z", "modified": modified,
                "contentStructured": [{"type": "heading" if heading else "paragraph", "text": opening},
                                      {"type": "paragraph", "text": "SERIES HISTORY"},
                                      {"type": "paragraph", "text": "Historical matchup facts remain unchanged."}]}
        raw = ('<script id="__NEXT_DATA__">' + json.dumps({"props": {"pageProps": {"pageObject": page}}}) + '</script>').encode()
        replay = "https://web.archive.org/web/" + capture + "id_/" + url
        from datetime import datetime, timezone
        from email.utils import format_datetime
        when = datetime.strptime(capture, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        source = {"publisher": "POR", "publisher_url": url, "status": 200, "url": replay, "final_url": replay,
                  "memento_datetime": format_datetime(when, usegmt=True), "sha256": hashlib.sha256(raw).hexdigest()}
        return raw, source, game, {"POR": [19, 52], "HOU": [35, 35]}

    def test_explicit_heading_or_paragraph_checks_preserve_all_canonical_source_sections(self):
        for heading in (True, False):
            with self.subTest(heading=heading):
                args = self.fixture(heading=heading)
                parsed = parse_por_html_note(*args)
                original = parse_html_note(*args[:3])
                self.assertEqual(parsed["sections"], original["sections"])
                self.assertEqual(parsed["sha256"], original["sha256"])
                self.assertEqual(parsed["eligibility"]["status"], "verified_pregame")
                self.assertIn("literal_PST_differs_from_verified_schedule_PDT", parsed["publisher_adapter"]["warnings"])
                self.assertIn("never the article timezone", parsed["publisher_adapter"]["cutoff_source"])

    def test_wrong_identity_records_date_weekday_or_wallclock_remain_quarantined(self):
        for change in [("Rockets", "Nuggets"), ("19-52", "20-51"), ("Monday", "Tuesday"),
                       ("Mar. 25", "Mar. 26"), ("2024", "2023"), ("5:00 pm", "6:00 pm"),
                       (" away)", " home)"), ("Monday", "Mon")]:
            with self.subTest(change=change):
                self.assertEqual(parse_por_html_note(*self.fixture(change))["eligibility"]["status"], "quarantined")

    def test_actual_replay_after_official_cutoff_or_modification_after_capture_rejected(self):
        for kwargs in ({"capture": "20240325230100"}, {"modified": "2024-03-25T20:00:00Z"}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(parse_por_html_note(*self.fixture(**kwargs))["eligibility"]["status"], "quarantined")

    def test_literal_extra_parenthesis_is_preserved_and_warned(self):
        parsed = parse_por_html_note(*self.fixture(("away)", "away))")))
        self.assertEqual(parsed["eligibility"]["status"], "verified_pregame")
        self.assertIn("extra_closing_parenthesis_in_literal_header", parsed["publisher_adapter"]["warnings"])
        self.assertIn("away))", parsed["sections"][1]["text"])

    def test_historical_recap_body_carries_actual_dated_header_context(self):
        raw, source, game, records = self.fixture()
        hydration = json.loads(raw.split(b">", 1)[1].rsplit(b"</script>", 1)[0])
        hydration["props"]["pageProps"]["pageObject"]["contentStructured"] += [
            {"type": "paragraph", "text": "LAST FIVE GAMES"},
            {"type": "paragraph", "text": "NUGGETS 114, TRAIL BLAZERS 111 - MARCH 23, 2024 - GAME #71 (19-52, 11-27 HOME)"},
            {"type": "paragraph", "text": "A player scored24 in that prior game."}]
        raw = ('<script id="__NEXT_DATA__">' + json.dumps(hydration) + '</script>').encode()
        source["sha256"] = hashlib.sha256(raw).hexdigest()
        parsed = parse_por_html_note(raw, source, game, records)
        self.assertEqual(parsed["eligibility"]["status"], "verified_pregame")
        self.assertEqual(parsed["publisher_adapter"]["context_sections"]["7"], [1, 2, 5, 6])
        self.assertIn("MARCH 23, 2024", parsed["sections"][5]["text"])


if __name__ == "__main__":
    unittest.main()
