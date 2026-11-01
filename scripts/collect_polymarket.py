"""Collect Polymarket NBA moneyline prices at each game's prediction cutoff.

A market price is a pregame reference forecast, never a label. Only CLOB price
points timestamped at or before the cutoff become the market probability;
resolution fields and lifetime volume stay in the audit file.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import re
import statistics
import urllib.parse

from collect_pilot import ROOT, UTC, Cache, iso, stamp, write_json, write_jsonl

CONFIG_PATH = ROOT / "config/polymarket.json"
DATA_DIR = ROOT / "data/processed/polymarket"
REPORT_DIR = ROOT / "reports/polymarket"
MARKETS = "https://gamma-api.polymarket.com/markets?"
PRICES = "https://clob.polymarket.com/prices-history?"
# Primary game markets use nba-{away}-{home}-{date}; some games also have a
# low-volume duplicate with a "-moneyline" suffix.
CANONICAL_SLUG = re.compile(r"nba-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}")
# Word boundaries matter: "Thunder" contains "under".
LINE_PATTERN = re.compile(r"\d\.5|O/U|\bover\b|\bunder\b|spread|\([+-]\d", re.I)
OUTCOME_FIELDS = {"home_win", "home_score", "away_score", "outcome_prices", "resolution", "volume"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_games(config):
    games = [g for g in read_rows(ROOT / config["history_path"]) if g["season"] in config["seasons"]]
    if len({g["game_id"] for g in games}) != len(games):
        raise ValueError("Duplicate historical game IDs")
    return sorted(games, key=lambda g: (g["scheduled_start_utc"], g["game_id"]))


def list_markets(cache, config, games):
    """Page through closed NBA moneyline markets, one end-date window at a time."""
    markets, failures = {}, []
    for season in sorted({g["season"] for g in games}):
        starts = [stamp(g["scheduled_start_utc"]) for g in games if g["season"] == season]
        # A market's endDate can be its resolution deadline, up to a week after tipoff.
        day = min(starts).replace(hour=0, minute=0, second=0) - timedelta(days=1)
        last = max(starts) + timedelta(days=config["listing_padding_days"])
        while day <= last:
            end = day + timedelta(days=config["listing_window_days"])
            offset = 0
            while True:
                query = {"tag_id": config["nba_tag_id"], "sports_market_types": "moneyline", "closed": "true",
                         "end_date_min": iso(day), "end_date_max": iso(end),
                         "limit": config["listing_page_size"], "offset": offset}
                body, meta = cache.get(MARKETS + urllib.parse.urlencode(query), "polymarket_markets", ".json")
                if body is None:
                    failures.append(meta)
                    break
                page = json.loads(body)
                markets.update((m["id"], m) for m in page)
                if len(page) < config["listing_page_size"]:
                    break
                offset += len(page)
            day = end
    return sorted(markets.values(), key=lambda m: m["id"]), failures


def discover_markets(cache, config, games):
    """Type-filtered listing, then a primary-slug lookup for games it missed.

    Some 2024-25 game markets are labelled "totals" although their outcomes are
    the two teams; the slug lookup finds them without listing every prop market.
    """
    markets, failures = list_markets(cache, config, games)
    index = index_markets(markets)
    known = {m["id"] for m in markets}
    for game in games:
        if candidate_markets(game, index, config["max_start_difference_minutes"]):
            continue
        slug = f'nba-{game["away_abbr"]}-{game["home_abbr"]}-{game["game_date_et"]}'.lower()
        # Gamma omits closed markets unless asked for them explicitly.
        body, meta = cache.get(MARKETS + urllib.parse.urlencode({"slug": slug, "closed": "true"}), "polymarket_markets", ".json")
        if body is None:
            failures.append(meta)
            continue
        markets.extend(m for m in json.loads(body) if m["id"] not in known)
        known.update(m["id"] for m in markets)
    return sorted(markets, key=lambda m: m["id"]), failures


def parse_market(market):
    try:
        outcomes, tokens = json.loads(market["outcomes"]), json.loads(market["clobTokenIds"])
        start = stamp(market["gameStartTime"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(outcomes) != 2 or len(tokens) != 2 or start.tzinfo is None:
        return None
    # Team-vs-team outcomes with a handicap or total line are not moneylines.
    if LINE_PATTERN.search(market.get("question") or ""):
        return None
    return {"id": market["id"], "slug": market["slug"], "condition_id": market.get("conditionId"),
            "outcomes": outcomes, "tokens": tokens, "start": start, "raw": market}


def outcome_side(name, game):
    # Whole-word suffix match: "Nets" must not match "Charlotte Hornets".
    name = name.lower()
    hits = [side for side in ("home", "away")
            if game[side + "_team"].lower() == name or game[side + "_team"].lower().endswith(" " + name)]
    return hits[0] if len(hits) == 1 else None


def index_markets(markets):
    index = defaultdict(list)
    for market in filter(None, map(parse_market, markets)):
        index[market["start"].date()].append(market)
    return index


def candidate_markets(game, index, max_minutes):
    start = stamp(game["scheduled_start_utc"])
    found = []
    for offset in (-1, 0, 1):
        for market in index.get((start + timedelta(days=offset)).date(), []):
            sides = [outcome_side(name, game) for name in market["outcomes"]]
            if sorted(map(str, sides)) == ["away", "home"] and abs(market["start"] - start) <= timedelta(minutes=max_minutes):
                found.append(dict(market, home_index=sides.index("home")))
    return sorted(found, key=lambda m: m["id"])


def choose_market(candidates, game_date_et):
    """Prefer the primary slug dated on the ET game date.

    Never use volume or prices: both include post-cutoff trading. A few games
    have parallel primary markets whose slugs carry the UTC date instead.
    """
    if not candidates:
        return None, "no_matching_market"
    pool = [m for m in candidates if CANONICAL_SLUG.fullmatch(m["slug"])] or candidates
    pool = [m for m in pool if m["slug"].endswith(game_date_et)] or pool
    if len(pool) > 1:
        return None, "ambiguous_markets"
    return pool[0], "matched"


def price_points(body):
    points = json.loads(body)["history"]
    for point in points:
        if not isinstance(point.get("t"), int) or not 0 <= float(point.get("p", -1)) <= 1:
            raise ValueError("Unexpected price-history point")
    return points


def price_at(points, moment):
    eligible = [p for p in points if p["t"] <= moment.timestamp()]
    return max(eligible, key=lambda p: p["t"]) if eligible else None


def price_url(game, market, config):
    start = stamp(game["scheduled_start_utc"])
    cutoff = start - timedelta(minutes=config["cutoff_minutes"])
    # Never request points after the scheduled tipoff.
    query = {"market": market["tokens"][market["home_index"]],
             "startTs": int((cutoff - timedelta(hours=config["price_lookback_hours"])).timestamp()),
             "endTs": int(start.timestamp()), "fidelity": config["price_fidelity_minutes"]}
    return PRICES + urllib.parse.urlencode(query)


def market_record(game, market, points, source, config):
    """Pregame-only row: no resolution, lifetime volume, scores or outcome."""
    start = stamp(game["scheduled_start_utc"])
    cutoff = start - timedelta(minutes=config["cutoff_minutes"])
    at_cutoff, at_tipoff = price_at(points, cutoff), price_at(points, start)
    if at_cutoff is None:
        return None, "no_price_before_cutoff"
    staleness = (cutoff.timestamp() - at_cutoff["t"]) / 60
    if staleness > config["max_staleness_minutes"]:
        return None, "stale_price"
    record = {key: game[key] for key in ("game_id", "season", "game_date_et", "scheduled_start_utc", "home_abbr", "away_abbr")}
    record.update(cutoff_utc=iso(cutoff), market_slug=market["slug"], market_id=market["id"],
                  condition_id=market["condition_id"], home_outcome=market["outcomes"][market["home_index"]],
                  home_token_id=market["tokens"][market["home_index"]],
                  market_start_utc=iso(market["start"]),
                  home_probability=float(at_cutoff["p"]), price_time_utc=iso(datetime.fromtimestamp(at_cutoff["t"], UTC)),
                  minutes_before_cutoff=staleness,
                  tipoff_home_probability=float(at_tipoff["p"]),
                  tipoff_price_time_utc=iso(datetime.fromtimestamp(at_tipoff["t"], UTC)),
                  source=source)
    return record, "matched_price_at_cutoff"



def summarize(records, audit, games, failures, config):
    decisions = Counter(row["decision"] for row in audit)
    by_season = {}
    for season in config["seasons"]:
        rows = [row for row in audit if row["season"] == season]
        by_season[str(season)] = {"games": len(rows), "decisions": dict(Counter(row["decision"] for row in rows))}
    staleness = [r["minutes_before_cutoff"] for r in records]
    volumes = sorted(float(row["market"]["volume"] or 0) for row in audit if row.get("market"))
    return {"scope": config["scope"], "games": len(games), "markets_with_price": len(records),
            "decisions": dict(decisions), "by_season": by_season, "listing_failures": len(failures),
            "median_minutes_before_cutoff": statistics.median(staleness) if staleness else None,
            "max_minutes_before_cutoff": max(staleness) if staleness else None,
            "matched_market_lifetime_volume_usd": {
                "median": statistics.median(volumes) if volumes else None,
                "p10": volumes[len(volumes) // 10] if volumes else None,
                "note": "Lifetime volume includes post-cutoff and in-game trading; descriptive only"},
            "price_definition": "Last Polymarket CLOB price-history point for the home outcome token at or before the cutoff",
            "config": config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Rebuild entirely from saved source responses")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent price-history downloads (online only)")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    cache = Cache(offline=args.offline)
    games = load_games(config)
    markets, failures = discover_markets(cache, config, games)
    if failures:
        write_json(REPORT_DIR / "listing_failures.json", failures)
        raise RuntimeError("Incomplete market listing; retry before matching games")
    index = index_markets(markets)
    print(f"Listed {len(markets)} NBA moneyline markets for {len(games)} games", flush=True)
    if args.workers > 1 and not args.offline:
        # Warm the response cache concurrently; the loop below reads it in order.
        urls = []
        for game in games:
            market, _ = choose_market(candidate_markets(game, index, config["max_start_difference_minutes"]), game["game_date_et"])
            if market:
                urls.append(price_url(game, market, config))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda url: cache.get(url, "polymarket_prices", ".json"), urls))
    records, audit = [], []
    for i, game in enumerate(games, 1):
        candidates = candidate_markets(game, index, config["max_start_difference_minutes"])
        market, decision = choose_market(candidates, game["game_date_et"])
        row = {"game_id": game["game_id"], "season": game["season"], "decision": decision,
               "candidate_slugs": [m["slug"] for m in candidates]}
        if market:
            raw = market["raw"]
            row["market"] = {"slug": market["slug"], "id": market["id"], "type_label": raw.get("sportsMarketType"),
                             "question": raw.get("question"), "volume": raw.get("volume"),
                             "outcomes": market["outcomes"], "outcome_prices": raw.get("outcomePrices"),
                             "resolution": raw.get("umaResolutionStatus"), "closed_time": raw.get("closedTime")}
            body, meta = cache.get(price_url(game, market, config), "polymarket_prices", ".json")
            row["price_source"] = meta
            if body is None:
                row["decision"] = "price_fetch_failed"
            else:
                record, row["decision"] = market_record(game, market, price_points(body), meta, config)
                if record:
                    records.append(record)
        audit.append(row)
        if i % 200 == 0:
            print(f"[{i}/{len(games)}] {len(records)} cutoff prices", flush=True)
    for record in records:
        if OUTCOME_FIELDS.intersection(record):
            raise ValueError("Outcome fields leaked into market probabilities")
    write_jsonl(DATA_DIR / "market_probs.jsonl", records)
    write_jsonl(DATA_DIR / "audit.jsonl", audit)
    (REPORT_DIR / "listing_failures.json").unlink(missing_ok=True)
    summary = summarize(records, audit, games, failures, config)
    write_json(REPORT_DIR / "coverage.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
