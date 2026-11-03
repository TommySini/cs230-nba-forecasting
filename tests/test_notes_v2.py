"""Checks for game-notes v2 matching: team slugs, stale versions, windows, dated files and header checks."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_notes_v2 as notes
from collect_pilot import stamp

CONFIG = json.loads(notes.CONFIG_PATH.read_text())


def game(game_id, start, home, away, season=2024, home_score=110, away_score=100, date=None):
    return {"game_id": game_id, "scheduled_start_utc": start, "game_date_et": date or start[:10], "season": season,
            "home_abbr": home, "away_abbr": away, "home_team": {"BOS": "Boston Celtics", "NYK": "New York Knicks",
            "PHI": "Philadelphia 76ers", "CLE": "Cleveland Cavaliers"}[home],
            "away_team": {"BOS": "Boston Celtics", "NYK": "New York Knicks", "PHI": "Philadelphia 76ers",
                          "CLE": "Cleveland Cavaliers"}[away], "home_score": home_score, "away_score": away_score}


def capture(ts, original, digest, status="200"):
    return {"timestamp": ts, "original": original, "digest": digest, "statuscode": status, "cdx_source": "x"}


class NotesV2Tests(unittest.TestCase):
    def test_slugs_follow_franchise_history_and_query_variants(self):
        when = stamp("2013-01-05T12:00:00Z")
        self.assertEqual(notes.slug_team("http://www.nba.com/gamenotes/hornets.pdf", when, CONFIG), "NOP")
        self.assertEqual(notes.slug_team("https://www.nba.com/gamenotes/hornets.pdf", stamp("2016-01-05T12:00:00Z"), CONFIG), "CHA")
        self.assertIsNone(notes.slug_team("https://www.nba.com/gamenotes/hornets.pdf", stamp("2014-01-05T12:00:00Z"), CONFIG))
        self.assertEqual(notes.slug_team("https://www.nba.com/gamenotes/lakers.pdf?ls=footer_nav", when, CONFIG), "LAL")
        self.assertEqual(notes.slug_team("http://www.nba.com/gamenotes//warriors.pdf", when, CONFIG), "GSW")

    def test_placeholder_shared_across_teams_is_dropped(self):
        caps = [capture("20250101120000", "https://www.nba.com/gamenotes/celtics.pdf", "EMPTY"),
                capture("20250101120500", "https://www.nba.com/gamenotes/knicks.pdf", "EMPTY"),
                capture("20250101121000", "https://www.nba.com/gamenotes/knicks.pdf", "REAL")]
        self.assertEqual([v["digest"] for v in notes.rolling_versions(caps, CONFIG)], ["REAL"])

    def test_stale_previous_game_packet_is_not_matched(self):
        history = [game("g1", "2025-01-02T00:30:00Z", "BOS", "NYK", date="2025-01-01"),
                   game("g2", "2025-01-04T00:30:00Z", "BOS", "PHI", date="2025-01-03")]
        caps = [capture("20250101150000", "https://www.nba.com/gamenotes/celtics.pdf", "D1"),   # notes for g1
                capture("20250102150000", "https://www.nba.com/gamenotes/celtics.pdf", "D1")]   # same file, after g1
        matched = notes.match_rolling(notes.rolling_versions(caps, CONFIG), history, CONFIG)
        self.assertEqual([(m["game_id"], m["digest"]) for m in matched], [("g1", "D1")])

    def test_latest_new_version_before_cutoff_wins_and_after_cutoff_is_ignored(self):
        history = [game("g1", "2025-01-02T00:30:00Z", "BOS", "NYK", date="2025-01-01")]
        caps = [capture("20250101140000", "https://www.nba.com/gamenotes/celtics.pdf", "A"),
                capture("20250101210000", "https://www.nba.com/gamenotes/celtics.pdf", "B"),
                capture("20250101233500", "https://www.nba.com/gamenotes/celtics.pdf", "C")]  # after 23:30 cutoff
        matched = notes.match_rolling(notes.rolling_versions(caps, CONFIG), history, CONFIG)
        self.assertEqual([(m["digest"], m["alternates"]) for m in matched if m["team"] == "BOS"], [("B", 1)])

    def test_dated_file_maps_to_exact_game(self):
        history = [game("g9", "2025-02-22T00:30:00Z", "CLE", "NYK", date="2025-02-21")]
        caps = [capture("20250221150000", "https://cdn.nba.com/teams/uploads/sites/1610612739/2025/02/250221-nykcle-notes.pdf", "X")]
        archived, live = notes.match_dated(caps, history, CONFIG)
        self.assertEqual([(c["game_id"], c["tier"]) for c in archived], [("g9", "archive_proven")])
        self.assertTrue(live[0]["url"].endswith("2025/02/250221-nykcle-notes.pdf"))

    def test_header_checks_accept_matching_and_reject_contradicting_records(self):
        g = game("g1", "2025-01-02T00:30:00Z", "BOS", "NYK", date="2025-01-01")
        before = {("g1", "home"): (20, 10), ("g1", "away"): (18, 12)}
        front = "BOSTON CELTICS (20-10) vs NEW YORK KNICKS (18-12) January 1, 2025 TD Garden " + "word " * 120
        self.assertEqual(notes.header_checks(front, g, "BOS", before, CONFIG)[0], [])
        stale = "BOSTON CELTICS (19-10) vs NEW YORK KNICKS (18-12) January 1, 2025 " + "word " * 120
        self.assertIn("home_record_contradicts_history", notes.header_checks(stale, g, "BOS", before, CONFIG)[0])
        wrong_day = "BOSTON CELTICS vs NEW YORK KNICKS December 30, 2024 " + "word " * 120
        self.assertIn("header_date_missing", notes.header_checks(wrong_day, g, "BOS", before, CONFIG)[0])
        undated = "BOSTON CELTICS (20-10) vs NEW YORK KNICKS (18-12) Game #31 TD Garden " + "word " * 120
        self.assertEqual(notes.header_checks(undated, g, "BOS", before, CONFIG)[0], [])
        undated_stale = "BOSTON CELTICS (19-10) vs NEW YORK KNICKS (18-12) Game #30 " + "word " * 120
        self.assertIn("header_date_missing", notes.header_checks(undated_stale, g, "BOS", before, CONFIG)[0])

    def test_one_document_per_team_game_prefers_archive_proof(self):
        site = {"game_id": "g1", "team": "CLE", "tier": "site_timestamped", "reasons": [], "words": 900}
        archive = {"game_id": "g1", "team": "CLE", "tier": "archive_proven", "reasons": [], "words": 500,
                   "capture_timestamp": "20250101120000"}
        rejected = {"game_id": "g1", "team": "CLE", "tier": "archive_proven", "reasons": ["postgame_content"],
                    "capture_timestamp": "20250101220000"}
        self.assertIs(notes.choose([site, rejected, archive])[0], archive)

    def test_post_title_date_maps_to_game(self):
        history = [game("g5", "2026-10-31T00:00:00Z", "NYK", "PHI", season=2026, date="2026-10-30")]
        post = {"team": "NYK", "title": "Knicks vs 76ers Game Notes | October 30, 2026"}
        self.assertEqual(notes.post_game(post, history, CONFIG)["game_id"], "g5")
        post = {"team": "NYK", "title": "Game Notes: Knicks at 76ers (Oct. 30, 2026)"}
        self.assertEqual(notes.post_game(post, history, CONFIG)["game_id"], "g5")

    def test_undated_post_maps_to_next_game_against_named_opponent(self):
        history = [game("g5", "2026-10-31T00:00:00Z", "NYK", "PHI", season=2026, date="2026-10-30")]
        post = {"team": "NYK", "title": "Game Notes: Knicks (3-2) vs. 76ers (2-3)", "published_utc": "2026-10-30T15:00:00Z"}
        self.assertEqual(notes.post_game(post, history, CONFIG)["game_id"], "g5")
        self.assertIsNone(notes.post_game(dict(post, title="Game Notes: Knicks vs. Celtics"), history, CONFIG))
        self.assertIsNone(notes.post_game(dict(post, published_utc="2026-10-31T01:00:00Z"), history, CONFIG))


if __name__ == "__main__":
    unittest.main()
