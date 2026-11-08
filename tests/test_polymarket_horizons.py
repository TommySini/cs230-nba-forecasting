"""Checks that horizon prices use only trades at or before each horizon and reject thin markets."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_pilot import stamp
from collect_polymarket_trades import horizon_row

RULES = {"trade_window_hours": 2, "max_history_age_minutes": 60, "min_window_trades": 2, "min_window_cash_usd": 20,
         "max_last_trade_age_minutes": 60, "max_history_vwap_gap": 0.05}
MARKET = {"scheduled_start_utc": "2025-01-15T00:30:00Z", "condition_id": "0xabc", "home_token_id": "home"}
ANCHOR = stamp("2025-01-14T16:30:00Z").timestamp()  # eight hours before tipoff


def trade(minutes_before_anchor, price, token="home", size=100):
    return {"timestamp": int(ANCHOR - 60 * minutes_before_anchor), "price": price, "token_id": token,
            "size": size, "condition_id": "0xabc", "transaction_hash": "0x1"}


class HorizonTests(unittest.TestCase):
    def test_later_trades_and_points_are_ignored_and_away_trades_are_converted(self):
        points = [{"t": int(ANCHOR - 600), "p": 0.6}, {"t": int(ANCHOR + 600), "p": 0.9}]
        trades = [trade(30, 0.6), trade(10, 0.42, token="away"), trade(-5, 0.95), trade(-300, 0.999)]
        row = horizon_row(MARKET, points, trades, 8, RULES)
        self.assertEqual(row["history_probability"], 0.6)
        self.assertEqual(row["window_trades"], 2)
        self.assertAlmostEqual(row["last_trade_home_probability"], 0.58)
        self.assertAlmostEqual(row["last_trade_age_minutes"], 10)
        self.assertTrue(row["usable"], row["reasons"])

    def test_thin_or_stale_or_disagreeing_markets_are_rejected(self):
        points = [{"t": int(ANCHOR - 600), "p": 0.6}]
        self.assertIn("too_few_trades", horizon_row(MARKET, points, [trade(30, 0.6)], 8, RULES)["reasons"])
        self.assertIn("last_trade_too_old", horizon_row(MARKET, points, [trade(100, 0.6), trade(90, 0.6)], 8, RULES)["reasons"])
        self.assertIn("history_disagrees_with_trades",
                      horizon_row(MARKET, points, [trade(20, 0.7), trade(10, 0.7)], 8, RULES)["reasons"])
        self.assertIn("no_recent_history_point", horizon_row(MARKET, [], [trade(20, 0.6), trade(10, 0.6)], 8, RULES)["reasons"])


if __name__ == "__main__":
    unittest.main()
