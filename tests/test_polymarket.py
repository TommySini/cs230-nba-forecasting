"""Regression checks for market matching and cutoff-price leakage."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_polymarket import (OUTCOME_FIELDS, candidate_markets, choose_market, discover_markets, index_markets,
                                list_markets, market_record, outcome_side, price_at)
from collect_pilot import stamp
import evaluate_polymarket as evaluation

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

    def test_evaluation_uses_collected_records_without_a_report_coverage_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data/processed/polymarket"
            data.mkdir(parents=True)
            config = {**CONFIG, "scope": "fixture", "seasons": [2024, 2025],
                      "comparison": {"boosting_metrics": "baseline.json", "boosting_predictions": "predictions.jsonl",
                                     "bootstrap_replicates": 10, "seed": 230, "calibration_bins": 2,
                                     "liquidity_threshold_usd": 50000}}
            (root / "config.json").write_text(json.dumps(config))
            (root / "baseline.json").write_text(json.dumps({"train_home_win_rate": .5}))
            games, predictions, markets, audit = [], [], [], []
            for i in range(4):
                home_win = i % 2
                games.append({**self.game, "game_id": str(i), "season": 2024 if i < 2 else 2025,
                              "game_date_et": f"2025-01-{i+1:02d}", "home_score": 110 if home_win else 90})
                predictions.append({"game_id": str(i), "split": "dev" if i < 2 else "test", "p_boost": .45 + .1 * home_win})
                markets.append({"game_id": str(i), "home_probability": .4 + .2 * home_win,
                                "tipoff_home_probability": .4 + .2 * home_win, "minutes_before_cutoff": 10})
                audit.append({"game_id": str(i), "season": games[-1]["season"], "decision": "matched_price_at_cutoff",
                              "market": {"volume": "100000", "resolution": "resolved", "outcomes": ["Nets", "Hornets"],
                                         "outcome_prices": json.dumps([home_win, 1-home_win])}})
            for path, rows in [(root / "predictions.jsonl", predictions), (data / "market_probs.jsonl", markets),
                               (data / "audit.jsonl", audit)]:
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result_path = root / "results/polymarket.json"
            with patch.multiple(evaluation, ROOT=root, DATA_DIR=data, CONFIG_PATH=root / "config.json", METRICS_PATH=result_path), \
                    patch.object(evaluation, "load_games", return_value=games), patch("sys.argv", ["evaluate_polymarket.py"]), \
                    redirect_stdout(io.StringIO()):
                evaluation.main()
                result = json.loads(result_path.read_text())
                self.assertEqual(result["coverage"]["markets_with_price"], 4)
                self.assertEqual(result["splits"]["test"]["matched_games"], 2)
                self.assertEqual(result["resolution_consistency"], {"agree": 4, "disagree": [], "unresolved": 0})
                self.assertIn("test_calibration", result)
                self.assertFalse((root / "reports").exists())
                original = result_path.read_bytes()
                (data / "listing_failures.json").write_text('[{"error": "incomplete listing"}]')
                with self.assertRaisesRegex(ValueError, "Incomplete market collection"):
                    evaluation.main()
                self.assertEqual(result_path.read_bytes(), original)

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
