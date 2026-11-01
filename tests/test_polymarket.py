"""Regression checks for market matching and cutoff-price leakage."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_polymarket import (OUTCOME_FIELDS, candidate_markets, choose_market, discover_markets, index_markets,
                                list_markets, market_record, outcome_side, price_at)
from collect_pilot import stamp

CONFIG = {"cutoff_minutes": 60, "max_staleness_minutes": 60, "max_start_difference_minutes": 180,
          "nba_tag_id": 745, "listing_window_days": 7, "listing_page_size": 2, "listing_padding_days": 1}


def market(slug, outcomes, start="2025-01-15 00:30:00+00", market_id="1", question=None):
    return {"id": market_id, "slug": slug, "conditionId": "0xabc", "outcomes": json.dumps(outcomes),
            "question": question or f"{outcomes[0]} vs. {outcomes[1]}",
            "clobTokenIds": json.dumps(["token-" + outcomes[0], "token-" + outcomes[1]]), "gameStartTime": start,
            "volume": "1000", "outcomePrices": '["1", "0"]', "umaResolutionStatus": "resolved"}


class PolymarketTests(unittest.TestCase):
    def setUp(self):
        self.game = {"game_id": "espn:1", "season": 2024, "game_date_et": "2025-01-14",
                     "scheduled_start_utc": "2025-01-15T00:30:00Z", "home_team": "Brooklyn Nets",
                     "away_team": "Charlotte Hornets", "home_abbr": "BKN", "away_abbr": "CHA",
                     "home_score": 99, "away_score": 101}
        self.cutoff = stamp("2025-01-14T23:30:00Z")

    def test_team_suffix_does_not_confuse_nets_and_hornets(self):
        self.assertEqual(outcome_side("Nets", self.game), "home")
        self.assertEqual(outcome_side("Hornets", self.game), "away")
        self.assertIsNone(outcome_side("Pistons", self.game))

    def test_home_token_follows_outcome_order_not_title_order(self):
        found = candidate_markets(self.game, index_markets([market("nba-cha-bkn-2025-01-14", ["Nets", "Hornets"])]), 180)
        self.assertEqual(found[0]["home_index"], 0)
        found = candidate_markets(self.game, index_markets([market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"])]), 180)
        self.assertEqual(found[0]["home_index"], 1)

    def test_primary_slug_preferred_over_moneyline_duplicate(self):
        markets = [market("nba-cha-bkn-2025-01-14-moneyline", ["Hornets", "Nets"], market_id="2"),
                   market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"], market_id="3")]
        chosen, decision = choose_market(candidate_markets(self.game, index_markets(markets), 180), "2025-01-14")
        self.assertEqual((chosen["slug"], decision), ("nba-cha-bkn-2025-01-14", "matched"))

    def test_rescheduled_or_rematch_market_is_not_matched(self):
        markets = [market("nba-cha-bkn-2025-01-16", ["Hornets", "Nets"], start="2025-01-17 00:30:00+00")]
        self.assertEqual(choose_market(candidate_markets(self.game, index_markets(markets), 180), "2025-01-14"), (None, "no_matching_market"))

    def test_et_dated_slug_preferred_over_utc_dated_parallel_market(self):
        markets = [market("nba-cha-bkn-2025-01-15", ["Hornets", "Nets"], market_id="6"),
                   market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"], market_id="7")]
        chosen, _ = choose_market(candidate_markets(self.game, index_markets(markets), 180), "2025-01-14")
        self.assertEqual(chosen["id"], "7")

    def test_handicap_or_total_lines_are_not_moneylines(self):
        self.assertEqual(index_markets([market("nba-cha-bkn-2025-01-14-spread", ["Hornets", "Nets"],
                                               question="Spread: Nets (-5.5)")]), {})
        thunder = dict(self.game, home_team="Oklahoma City Thunder")
        found = candidate_markets(thunder, index_markets([market("nba-cha-okc-2025-01-14", ["Hornets", "Thunder"])]), 180)
        self.assertEqual(len(found), 1)

    def test_slug_lookup_finds_game_market_with_wrong_type_label(self):
        class FakeCache:
            def __init__(self):
                self.urls = []

            def get(self, url, kind, suffix):
                self.urls.append(url)
                found = [dict(market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"]), sportsMarketType="totals")] if "slug=" in url else []
                return json.dumps(found).encode(), {"status": 200}
        cache = FakeCache()
        markets, failures = discover_markets(cache, CONFIG, [self.game])
        self.assertTrue(any("slug=nba-cha-bkn-2025-01-14" in url for url in cache.urls))
        self.assertEqual(choose_market(candidate_markets(self.game, index_markets(markets), 180), "2025-01-14")[1], "matched")
        self.assertEqual(failures, [])

    def test_two_primary_markets_are_ambiguous(self):
        markets = [market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"], market_id="4"),
                   market("nba-bkn-cha-2025-01-14", ["Nets", "Hornets"], market_id="5")]
        self.assertEqual(choose_market(candidate_markets(self.game, index_markets(markets), 180), "2025-01-14")[1], "ambiguous_markets")

    def test_price_after_cutoff_never_used(self):
        cutoff = int(self.cutoff.timestamp())
        points = [{"t": cutoff - 600, "p": 0.40}, {"t": cutoff, "p": 0.45}, {"t": cutoff + 60, "p": 0.99}]
        self.assertEqual(price_at(points, self.cutoff)["p"], 0.45)
        found = candidate_markets(self.game, index_markets([market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"])]), 180)[0]
        record, decision = market_record(self.game, found, points, {"url": "x"}, CONFIG)
        self.assertEqual((record["home_probability"], record["tipoff_home_probability"], decision), (0.45, 0.99, "matched_price_at_cutoff"))
        self.assertEqual(record["home_token_id"], "token-Nets")

    def test_stale_or_missing_pregame_price_is_rejected(self):
        found = candidate_markets(self.game, index_markets([market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"])]), 180)[0]
        cutoff = int(self.cutoff.timestamp())
        self.assertEqual(market_record(self.game, found, [{"t": cutoff - 7200, "p": 0.5}], {}, CONFIG)[1], "stale_price")
        self.assertEqual(market_record(self.game, found, [{"t": cutoff + 60, "p": 0.5}], {}, CONFIG)[1], "no_price_before_cutoff")

    def test_market_record_contains_no_outcome_or_resolution_fields(self):
        found = candidate_markets(self.game, index_markets([market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"])]), 180)[0]
        record, _ = market_record(self.game, found, [{"t": int(self.cutoff.timestamp()), "p": 0.5}], {}, CONFIG)
        self.assertFalse(OUTCOME_FIELDS.intersection(record))

    def test_listing_pages_until_short_page_and_stops_on_failure(self):
        class FakeCache:
            def __init__(self, pages):
                self.pages, self.urls = pages, []

            def get(self, url, kind, suffix):
                self.urls.append(url)
                page = self.pages.pop(0) if self.pages else []
                return (None, {"url": url, "status": "network_error"}) if page is None else (json.dumps(page).encode(), {"status": 200})
        games = [self.game]
        cache = FakeCache([[{"id": "1"}, {"id": "2"}], [{"id": "3"}]])
        markets, failures = list_markets(cache, CONFIG, games)
        self.assertEqual([m["id"] for m in markets], ["1", "2", "3"])
        self.assertIn("offset=2", cache.urls[1])
        self.assertEqual(failures, [])
        markets, failures = list_markets(FakeCache([None]), CONFIG, games)
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()
