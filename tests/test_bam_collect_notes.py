from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bam_collect_notes as collect


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.game = {"game_id": "espn:1", "game_date_et": "2024-01-01", "scheduled_start_utc": "2024-01-02T01:00:00Z",
                     "home_abbr": "LAL", "away_abbr": "DEN"}
        self.config = {"minimum_game_date": "2023-10-01", "maximum_game_date": "2026-04-10",
                       "rolling_max_age_hours": 36, "previous_tip_buffer_hours": 2.5}

    def record(self, **kwargs):
        return {"timestamp": "20240101230000", "original": "https://www.nba.com/gamenotes/lakers.pdf?ls=footer_nav",
                "mimetype": "application/pdf", "statuscode": "200", "digest": "DIGEST", **kwargs}

    def test_query_variants_before_utc_next_day_tip_retained(self):
        result = collect.index_candidates([self.record()], [self.game], {"LAL": "lakers", "DEN": "nuggets"}, self.config)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["publisher_url"].endswith("?ls=footer_nav"))
        self.assertEqual(result[0]["cutoff_utc"], "2024-01-02T00:00:00Z")

    def test_stale_digest_revisit_does_not_become_fresh_packet(self):
        records = [self.record(timestamp="20231230120000"), self.record(mimetype="warc/revisit", statuscode="-")]
        self.assertEqual(collect.index_candidates(records, [self.game], {"LAL": "lakers"}, self.config), [])

    def test_after_cutoff_or_previous_game_running_rejected(self):
        late = self.record(timestamp="20240102000100")
        self.assertEqual(collect.index_candidates([late], [self.game], {"LAL": "lakers"}, self.config), [])
        previous = {**self.game, "game_id": "espn:0", "game_date_et": "2023-12-31", "scheduled_start_utc": "2024-01-01T22:00:00Z"}
        self.assertEqual(collect.index_candidates([self.record()], [previous, self.game], {"LAL": "lakers"}, self.config), [])

    def test_cross_team_placeholder_and_html_block_page_dropped(self):
        records = [self.record(), self.record(original="https://www.nba.com/gamenotes/nuggets.pdf")]
        self.assertEqual(collect.index_candidates(records, [self.game], {"LAL": "lakers", "DEN": "nuggets"}, self.config), [])
        self.assertEqual(collect.index_candidates([self.record(mimetype="text/html")], [self.game], {"LAL": "lakers"}, self.config), [])
        self.assertEqual(collect.index_candidates([self.record(original="https://other.example/gamenotes/lakers.pdf")], [self.game], {"LAL": "lakers"}, self.config), [])
        with self.assertRaises(ValueError):
            collect.parse_cdx(b"<html>Temporarily Offline</html>")

    def test_exact_knicks_leads_require_date_matchup_and_regular_season(self):
        game = {**self.game, "home_abbr": "NYK"}
        row = "| 2024-01-01 | DEN@NYK | RS | [Knicks Game Notes](https://www.nba.com/knicks/news/exact-note) | 2024-01-01T12:00 | 2024-01-01T12:01 | 10 | before | 20240101130000 | |"
        result = collect.knicks_leads(row, [game])
        self.assertEqual(result[0]["source_kind"], "html")
        self.assertEqual(result[0]["archive_timestamp"], "20240101130000")
        self.assertEqual(collect.knicks_leads(row.replace("| RS |", "| PO |"), [game]), [])
        self.assertEqual(collect.knicks_leads(row.replace("DEN@NYK", "NYK@DEN"), [game]), [])

    def test_capture_aliases_survive_approved_manifest_deduplication(self):
        note = {"game_id": "espn:1", "game_date_et": "2024-01-01", "team": "LAL",
                "source": {"sha256": "a" * 64, "final_url": "https://web.archive.org/web/20240101120000id_/https://nba.com/a.pdf"}}
        later = deepcopy(note)
        later["source"]["final_url"] = later["source"]["final_url"].replace("120000", "140000")
        result = collect.deduplicate_approved([note, later, note])
        self.assertEqual(len(result), 2)
        self.assertEqual(collect.deduplicate_approved([later, note]), result)

    def test_rejected_closest_capture_redirect_does_not_reblock_requested_lead(self):
        requested = "https://web.archive.org/web/20240223224016id_/https://nba.com/bucks.pdf"
        actual = requested.replace("20240223224016", "20240224230750")
        older = requested.replace("20240223224016", "20240222220604")
        finished = collect.replay_urls([{"url": requested, "final_url": actual}])
        self.assertIn(requested, finished)
        self.assertIn(actual, finished)
        self.assertNotIn(older, finished)

    def test_portland_collection_retains_strict_date_and_opponent_quarantine(self):
        url = "https://www.nba.com/blazers/news/game-notes-lakers-at-trail-blazers"
        game = {**self.game, "home_abbr": "POR", "away_abbr": "LAL"}
        page = {"type": "article", "title": "Game Notes: Lakers at Trail Blazers", "permalink": url,
                "date": "2024-01-01T18:00:00Z", "modified": "2024-01-01T18:00:00Z",
                "contentStructured": [
                    {"type": "paragraph", "html": "<p>The Trail Blazers host the Lakers on January 1, 2024.</p>"},
                    {"type": "heading", "html": "<h2>Last game played vs Grizzlies</h2>"}]}
        body = ("<script id=\"__NEXT_DATA__\" type=\"application/json\">" +
                json.dumps({"props": {"pageProps": {"pageObject": page}}}) + "</script>").encode()
        replay = "https://web.archive.org/web/20240101220000id_/" + url
        source = {"status": 200, "url": replay, "final_url": replay,
                  "memento_datetime": "Mon, 01 Jan 2024 22:00:00 GMT", "sha256": collect.digest(body)}
        result = collect.assess_html(body, source, game, url, team="POR", history=[])
        self.assertEqual(result["parsed_article"]["publisher"], "POR")
        self.assertIn("game_date_missing_from_title", result["reasons"])
        self.assertIn("last_meeting_heading_wrong_opponent", result["reasons"])
        self.assertEqual(result["decision"], "quarantined_html_metadata_or_content")


class CacheTests(unittest.TestCase):
    def test_http200_block_page_is_preserved_but_not_reusable_success(self):
        class Response:
            status = 200
            url = "https://web.archive.org/cdx/search/cdx?x"
            headers = {}
            def read(self):
                return b"<html>Temporarily Offline</html>"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        config = {"namespace": "bam_notes_expansion/test", "max_network_requests": 1,
                  "archive_gap_seconds": 0, "publisher_gap_seconds": 0, "timeout_seconds": 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(collect, "ROOT", Path(directory)), patch.object(collect.urllib.request, "urlopen", return_value=Response()):
            cache = collect.BoundedCache(config)
            body, source = cache.get(Response.url, "indexes")
            self.assertIsNone(body)
            self.assertEqual(source["status"], 200)
            self.assertFalse(source["cache_usable"])
            self.assertTrue(cache.archive_throttled)
            self.assertTrue((Path(directory) / source["raw_path"]).exists())
            body, meta = collect.BoundedCache(config, offline=True).get(Response.url, "indexes")
            self.assertIsNone(body)


class OfficialRecordHistoryTests(unittest.TestCase):
    def test_both_finalists_records_before_and_after_each_cup_final(self):
        from review_game_notes import prior_record
        cases = [("espn:401607495", "2023-12-09", 2023, "LAL", "IND"),
                 ("espn:401734908", "2024-12-17", 2024, "OKC", "MIL"),
                 ("espn:401809839", "2025-12-16", 2025, "NYK", "SAS")]
        for gid, day, season, home, away in cases:
            with self.subTest(game_id=gid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                final_date = datetime.fromisoformat(day)
                history = []
                for number, days_before in enumerate((3, 2)):
                    history.append({"game_id": str(number), "game_date_et": (final_date - timedelta(days=days_before)).date().isoformat(),
                                    "season": season, "home_abbr": home, "away_abbr": away,
                                    "home_score": 100 if number == 0 else 90, "away_score": 90 if number == 0 else 100})
                raw = root / "data/raw/scoreboard.json"
                raw.parent.mkdir(parents=True)
                raw.write_text(json.dumps({"events": [{"id": gid.split(":")[1], "competitions": [{"type": {"id": "39", "abbreviation": "CC"},
                                         "notes": [{"headline": "NBA Cup Championship"}]}]}]}))
                final = {"game_id": gid, "game_date_et": day, "season": season, "home_abbr": home, "away_abbr": away,
                         "home_score": 101, "away_score": 99, "schedule_source": "data/raw/scoreboard.json"}
                history.append(final)
                rule = {**{k: final[k] for k in ("game_id", "game_date_et", "season", "home_abbr", "away_abbr", "schedule_source")},
                        "scoreboard_sha256": collect.digest(raw.read_bytes())}
                policy = {"excluded_only_for_official_record_and_header_validation": [rule]}
                official, excluded = collect.official_record_history(history, policy, root)
                self.assertEqual(excluded, {gid})
                for delta in (-1, 0, 1):
                    target = {"season": season, "game_date_et": (final_date + timedelta(days=delta)).date().isoformat()}
                    for team in (home, away):
                        self.assertEqual(prior_record(target, team, official), (1, 1))
                after = {"season": season, "game_date_et": (final_date + timedelta(days=1)).date().isoformat()}
                self.assertEqual(prior_record(after, home, history), (2, 1))
                self.assertEqual(prior_record(after, away, history), (1, 2))
                # Actual feature history remains unmodified and Cup final stays.
                self.assertEqual(len(history), 3)
                self.assertEqual(len(official), 2)

    def test_regular_game_cannot_be_excluded_by_an_unverified_id_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scoreboard.json"
            path.write_text(json.dumps({"events": [{"id": "1", "competitions": [{"type": {"id": "1", "abbreviation": "STD"}, "notes": []}]}]}))
            game = {"game_id": "espn:1", "game_date_et": "2024-01-01", "season": 2023, "home_abbr": "LAL", "away_abbr": "IND", "schedule_source": "scoreboard.json"}
            rule = {**game, "scoreboard_sha256": collect.digest(path.read_bytes())}
            with self.assertRaisesRegex(ValueError, "explicitly identified"):
                collect.official_record_history([game], {"excluded_only_for_official_record_and_header_validation": [rule]}, root)


if __name__ == "__main__":
    unittest.main()
