"""Regression checks for market matching and cutoff-price leakage."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_polymarket import (OUTCOME_FIELDS, candidate_markets, choose_market, discover_markets, index_markets,
                                list_markets, market_record, outcome_side, price_at, write_market_tables,
                                load_market_records, load_horizon_records, load_market_audit,
                                load_price_observations, migrate_market_data)
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


class MarketStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.game = {"game_id": "espn:1", "season": 2024, "game_date_et": "2025-01-14",
                     "scheduled_start_utc": "2025-01-15T00:30:00Z", "home_team": "Brooklyn Nets",
                     "away_team": "Charlotte Hornets", "home_abbr": "BKN", "away_abbr": "CHA"}
        chosen = candidate_markets(self.game, index_markets([market("nba-cha-bkn-2025-01-14", ["Hornets", "Nets"])]), 180)[0]
        cutoff = int(stamp("2025-01-14T23:30:00Z").timestamp())
        self.record, _ = market_record(self.game, chosen, [{"t": cutoff - 1, "p": 0.41}, {"t": cutoff + 60, "p": 0.99}],
                                      {"url": "https://example.com/prices", "raw_path": "data/raw/polymarket_prices/a.json",
                                       "sha256": "a" * 64, "status": 200}, CONFIG)
        self.audit = [{"game_id": "espn:1", "season": 2024, "decision": "matched_price_at_cutoff",
                       "candidate_slugs": [self.record["market_slug"]],
                       "market": {"slug": self.record["market_slug"], "id": "1", "type_label": "moneyline",
                                  "question": "Hornets vs Nets", "volume": "1000", "outcomes": ["Hornets", "Nets"],
                                  "outcome_prices": '["0", "1"]', "resolution": "resolved", "closed_time": None},
                       "price_source": self.record["source"]},
                      {"game_id": "espn:2", "season": 2024, "decision": "no_matching_market", "candidate_slugs": []}]
        self.horizons = [{**{k: self.record[k] for k in ("game_id", "season", "game_date_et", "scheduled_start_utc",
                                                       "home_abbr", "away_abbr", "market_slug", "condition_id", "home_token_id")},
                          "trade_pages": [{"url": "https://example.com/trades", "sha256": "b" * 64,
                                           "raw_path": "data/raw/polymarket_trades/b.json"}],
                          "horizons": [{"hours_before_tipoff": 8, "anchor_utc": "2025-01-14T16:30:00Z",
                                        "history_probability": None, "history_point_utc": None, "history_age_minutes": None,
                                        "window_trades": 0, "window_cash_usd": 0, "usable": False,
                                        "reasons": ["no_recent_history_point", "too_few_trades", "too_little_cash"]}]}]

    def test_typed_tables_round_trip_optional_fields_and_exclude_reference_from_pregame(self):
        import pyarrow.parquet as pq
        from nba_dataset import remap_paths
        write_market_tables(self.directory, [self.record], self.audit, self.horizons)
        self.assertEqual(load_market_records(processed_dir=self.directory), remap_paths([self.record]))
        self.assertEqual(load_market_audit(processed_dir=self.directory), remap_paths(self.audit))
        self.assertEqual(load_horizon_records(processed_dir=self.directory), remap_paths(self.horizons))
        observations = load_price_observations(processed_dir=self.directory)
        self.assertEqual({r["record_kind"] for r in observations}, {"cutoff", "horizon"})
        self.assertNotIn(0.99, [r["home_probability"] for r in observations])
        self.assertTrue(all(not OUTCOME_FIELDS.intersection(r) for r in observations))
        schema = pq.read_schema(self.directory / "market_prices.parquet")
        self.assertEqual(str(schema.field("cutoff_utc").type), "timestamp[us, tz=UTC]")
        self.assertEqual(str(schema.field("home_probability").type), "double")

    def test_price_after_cutoff_is_rejected_when_loading(self):
        late = dict(self.record, price_time_utc="2025-01-14T23:30:01Z")
        write_market_tables(self.directory, [late], self.audit)
        with self.assertRaisesRegex(ValueError, "later than its cutoff"):
            load_price_observations(processed_dir=self.directory)

    def test_failed_download_audit_preserves_text_status_and_error(self):
        failure = {"game_id": "espn:3", "season": 2024, "decision": "price_fetch_failed", "candidate_slugs": [],
                   "price_source": {"url": "https://example.com/failure", "status": "network_error", "error": "timeout"}}
        write_market_tables(self.directory, [self.record], [failure])
        self.assertEqual(load_market_audit(processed_dir=self.directory), [failure])

    def test_outcomes_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Outcome fields"):
            write_market_tables(self.directory, [dict(self.record, home_win=1)], self.audit)
        with self.assertRaisesRegex(ValueError, "Duplicate market game"):
            write_market_tables(self.directory, [self.record, self.record], self.audit)

    def test_migration_preserves_all_legacy_fields_and_returns_source_hashes(self):
        legacy = self.directory / "data/processed/polymarket"
        legacy.mkdir(parents=True)
        for name, records in (("market_probs", [self.record]), ("horizon_prices", self.horizons), ("audit", self.audit)):
            (legacy / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in records))
        (legacy / "comparison_test.csv").write_bytes(b"game_id,p\ng1,0.5\n")
        log = b"[1/1] markets walked\r\ncomplete\n"
        (legacy / "trades_fetch.log").write_bytes(log)
        result = migrate_market_data(self.directory, self.directory / "staging")
        self.assertEqual(result["equality"], {"market_records": 1, "horizon_records": 1,
                                             "horizon_observations": 1, "market_audit": 2})
        self.assertEqual(len(result["source_sha256"]), 5)
        self.assertEqual(set(result["source_sha256"]), set(result["cleanup_coverage"]))
        report = self.directory / "staging/exports/reports/polymarket/collection.json"
        self.assertEqual(json.loads(report.read_text())["trades_fetch_log"].encode(), log)
        self.assertTrue((legacy / "market_probs.jsonl").exists())
        (legacy / "unrecognized.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unrecognized market migration inputs"):
            migrate_market_data(self.directory, self.directory / "other_staging")
        self.assertFalse((self.directory / "other_staging").exists())


if __name__ == "__main__":
    unittest.main()
