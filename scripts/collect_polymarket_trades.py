"""Market prices at fixed hours before tipoff, checked against the trades behind them.

Game notes are typically published about eight hours before tipoff, so a fair market
reference is the price at that moment, not one hour before. For every matched market and
every horizon this records the CLOB price-history point at or before tipoff minus the
horizon, plus the actual trades in the preceding window: count, cash volume, age of the
last trade and the volume-weighted price. A price is usable only when real, recent
trading supports it. Trade pages are walked newest first (the Data API cannot filter a
market by time), so raw pages include in-game and settlement trades; only trades at or
before each horizon reach the processed rows.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import statistics
import time
import urllib.parse

from collect_pilot import ROOT, UTC, Cache, iso, stamp, write_json, write_jsonl
from collect_polymarket import CONFIG_PATH, DATA_DIR, OUTCOME_FIELDS, REPORT_DIR, price_at, price_points, read_rows

TRADES = "https://data-api.polymarket.com/v2/trades?"


def fetch(cache, url, attempts=6):
    for attempt in range(attempts):
        body, meta = cache.get(url, "polymarket_trades", ".json")
        if body is not None or meta.get("status") in (400, 403, 404, 410, "not_cached"):
            return body, meta
        time.sleep(5 * 2 ** attempt)  # 429 / 5xx / network: back off, then retry
    return body, meta


def walk_trades(cache, market, stop_ts, max_pages):
    """Newest-first trade pages for one market until a page reaches back past stop_ts."""
    query = {"condition": market["condition_id"], "limit": 1000}
    trades, pages = [], []
    for _ in range(max_pages):
        body, meta = fetch(cache, TRADES + urllib.parse.urlencode(query))
        if body is None:
            return None, pages + [meta]
        page = json.loads(body)
        pages.append({"url": meta["url"], "sha256": meta["sha256"], "raw_path": meta["raw_path"]})
        trades += page["data"]
        cursor = page["pagination"].get("next_cursor")
        if not cursor or not page["data"] or page["data"][-1]["timestamp"] < stop_ts:
            return trades, pages
        query = {"condition": market["condition_id"], "cursor": cursor}
    return None, pages + [{"error": "max_pages_reached"}]


def home_price(trade, market):
    if trade["condition_id"] != market["condition_id"]:
        raise ValueError(f"Trade from another market: {trade['transaction_hash']}")
    price = float(trade["price"])
    return price if trade["token_id"] == market["home_token_id"] else 1 - price


def horizon_row(market, points, trades, hours, rules):
    """Price-history point at tipoff minus `hours`, and the trades in the window before it."""
    anchor = stamp(market["scheduled_start_utc"]) - timedelta(hours=hours)
    point = price_at(points, anchor)
    window = [t for t in trades if anchor.timestamp() - rules["trade_window_hours"] * 3600 < t["timestamp"] <= anchor.timestamp()]
    cash = [float(t["size"]) * float(t["price"]) for t in window]
    row = {"hours_before_tipoff": hours, "anchor_utc": iso(anchor),
           "history_probability": float(point["p"]) if point else None,
           "history_point_utc": iso(datetime.fromtimestamp(point["t"], UTC)) if point else None,
           "history_age_minutes": (anchor.timestamp() - point["t"]) / 60 if point else None,
           "window_trades": len(window), "window_cash_usd": sum(cash)}
    if window:
        last = max(window, key=lambda t: t["timestamp"])
        row.update(last_trade_home_probability=home_price(last, market),
                   last_trade_age_minutes=(anchor.timestamp() - last["timestamp"]) / 60,
                   vwap_home_probability=sum(c * home_price(t, market) for c, t in zip(cash, window)) / sum(cash) if sum(cash) else None)
    reasons = []
    if point is None or row["history_age_minutes"] > rules["max_history_age_minutes"]:
        reasons.append("no_recent_history_point")
    if row["window_trades"] < rules["min_window_trades"]:
        reasons.append("too_few_trades")
    if row["window_cash_usd"] < rules["min_window_cash_usd"]:
        reasons.append("too_little_cash")
    if window and row["last_trade_age_minutes"] > rules["max_last_trade_age_minutes"]:
        reasons.append("last_trade_too_old")
    if point and row.get("vwap_home_probability") is not None and \
            abs(row["history_probability"] - row["vwap_home_probability"]) > rules["max_history_vwap_gap"]:
        reasons.append("history_disagrees_with_trades")
    row.update(usable=not reasons, reasons=reasons)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Rebuild entirely from saved trade pages")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    rules = config["horizons"]
    cache = Cache(offline=args.offline, delay=0.25)
    markets = read_rows(DATA_DIR / "market_probs.jsonl")
    reach = max(rules["hours_before_tipoff"]) + rules["trade_window_hours"]
    stop = lambda m: (stamp(m["scheduled_start_utc"]) - timedelta(hours=reach)).timestamp()
    work = lambda m: walk_trades(cache, m, stop(m), rules["max_pages"])
    with ThreadPoolExecutor(max_workers=1 if args.offline else args.workers) as pool:
        walked = []
        for i, result in enumerate(pool.map(work, markets), 1):
            walked.append(result)
            if i % 200 == 0:
                print(f"[{i}/{len(markets)}] markets walked", flush=True)
    rows, failures = [], []
    for market, (trades, pages) in zip(markets, walked):
        if trades is None:
            failures.append({"game_id": market["game_id"], "pages": pages})
            continue
        points = price_points((ROOT / market["source"]["raw_path"]).read_text())
        rows.append({**{k: market[k] for k in ("game_id", "season", "game_date_et", "scheduled_start_utc", "home_abbr",
                                                "away_abbr", "market_slug", "condition_id", "home_token_id")},
                     "trade_pages": pages,
                     "horizons": [horizon_row(market, points, trades, h, rules) for h in rules["hours_before_tipoff"]]})
    for row in rows:
        if OUTCOME_FIELDS.intersection(row) or any(OUTCOME_FIELDS.intersection(h) for h in row["horizons"]):
            raise ValueError("Outcome fields leaked into horizon prices")
    write_jsonl(DATA_DIR / "horizon_prices.jsonl", rows)
    summary = {"markets": len(markets), "walked": len(rows), "failures": failures, "rules": rules, "by_horizon": {}}
    for hours in rules["hours_before_tipoff"]:
        hs = [next(h for h in r["horizons"] if h["hours_before_tipoff"] == hours) for r in rows]
        reasons = {}
        for h in hs:
            for reason in h["reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        quantile = lambda xs, q: sorted(xs)[int(q * (len(xs) - 1))] if xs else None
        cash = [h["window_cash_usd"] for h in hs]
        age = [h["last_trade_age_minutes"] for h in hs if h.get("last_trade_age_minutes") is not None]
        summary["by_horizon"][str(hours)] = {
            "usable": sum(h["usable"] for h in hs), "reasons": reasons,
            "window_trades_median": statistics.median(h["window_trades"] for h in hs),
            "window_cash_usd": {"p10": quantile(cash, .1), "median": quantile(cash, .5), "p90": quantile(cash, .9)},
            "last_trade_age_minutes": {"median": quantile(age, .5), "p90": quantile(age, .9)}}
    write_json(REPORT_DIR / "horizon_coverage.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("failures", "rules")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
