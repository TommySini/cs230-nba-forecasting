"""Collect Polymarket NBA moneyline prices at each game's prediction cutoff.

A market price is a pregame reference forecast, never a label. Only CLOB price
points timestamped at or before the cutoff become the market probability;
resolution fields and lifetime volume stay in the audit file.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import urllib.parse

from collect_pilot import ROOT, UTC, Cache, iso, stamp, write_json

CONFIG_PATH = ROOT / "config/polymarket.json"
DATA_DIR = ROOT / "data/processed"
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
    from nba_dataset import read_rows as read_dataset_rows
    return read_dataset_rows(path)


def load_games(config, *, root=ROOT, processed_dir=None):
    from nba_dataset import Dataset
    directory = Path(processed_dir) if processed_dir is not None else Path(root) / config["dataset_dir"]
    games = Dataset(root, directory).history(config["seasons"], include_non_franchise=True)
    if len({g["game_id"] for g in games}) != len(games):
        raise ValueError("Duplicate historical game IDs")
    return sorted(games, key=lambda g: (g["scheduled_start_utc"], g["game_id"]))


MARKET_COMMON = ("game_id", "season", "game_date_et", "scheduled_start_utc", "home_abbr", "away_abbr",
                 "market_slug", "condition_id", "home_token_id")
SOURCE_FIELDS = ("url", "retrieved_at_utc", "status", "content_type", "last_modified", "sha256", "bytes", "raw_path", "error")
HORIZON_FIELDS = ("hours_before_tipoff", "anchor_utc", "history_probability", "history_point_utc",
                  "history_age_minutes", "window_trades", "window_cash_usd", "last_trade_home_probability",
                  "last_trade_age_minutes", "vwap_home_probability", "usable", "reasons")
TIME_FIELDS = ("scheduled_start_utc", "market_start_utc", "cutoff_utc", "price_time_utc")


def market_schemas():
    """Explicit types keep prices numeric and audit-only outcomes out of price rows."""
    import pyarrow as pa
    source = pa.struct([(name, pa.int64() if name == "bytes" else pa.string())
                        for name in SOURCE_FIELDS if name != "status"] +
                       [("status_code", pa.int64()), ("status_text", pa.string())])
    page = pa.struct([(name, pa.string()) for name in ("url", "sha256", "raw_path")])
    fields = [("row_order", pa.int64()), ("horizon_order", pa.int32()), ("record_kind", pa.string()),
              ("price_method", pa.string()), ("game_id", pa.string()), ("season", pa.int32()),
              ("game_date_et", pa.date32()), ("scheduled_start_utc", pa.timestamp("us", tz="UTC")),
              *[(k, pa.string()) for k in ("home_abbr", "away_abbr", "market_slug", "condition_id", "home_token_id",
                                           "market_id", "home_outcome")],
              *[(k, pa.timestamp("us", tz="UTC")) for k in ("market_start_utc", "cutoff_utc", "price_time_utc")],
              ("home_probability", pa.float64()), ("minutes_before_cutoff", pa.float64()),
              ("hours_before_tipoff", pa.float64()), ("source", source), ("source_fields", pa.list_(pa.string())),
              ("trade_pages", pa.list_(page)), ("horizon_fields", pa.list_(pa.string())),
              ("window_trades", pa.int64()), ("window_cash_usd", pa.float64()),
              ("last_trade_home_probability", pa.float64()), ("last_trade_age_minutes", pa.float64()),
              ("vwap_home_probability", pa.float64()), ("usable", pa.bool_()), ("reasons", pa.list_(pa.string()))]
    audit_market = pa.struct([(k, pa.list_(pa.string()) if k == "outcomes" else pa.string()) for k in
                             ("slug", "id", "type_label", "question", "volume", "outcomes", "outcome_prices",
                              "resolution", "closed_time")])
    audit = pa.schema([("row_order", pa.int64()), ("game_id", pa.string()), ("season", pa.int32()),
                       ("decision", pa.string()), ("candidate_slugs", pa.list_(pa.string())),
                       ("market", audit_market), ("price_source", source),
                       ("source_fields", pa.list_(pa.string()))])
    return pa.schema(fields), audit


def _encode_source(source):
    from nba_dataset import remap_paths
    unknown = set(source) - set(SOURCE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown source metadata fields: {sorted(unknown)}")
    encoded = remap_paths({k: v for k, v in source.items() if k != "status"})
    if "status" in source:
        field = "status_code" if isinstance(source["status"], int) else "status_text"
        encoded[field] = source["status"]
    return encoded


def _decode_source(source, fields):
    return {k: (source["status_code"] if source["status_code"] is not None else source["status_text"])
            if k == "status" else source[k] for k in fields}


def _typed_market_row(row):
    row = dict(row)
    for key in TIME_FIELDS:
        if isinstance(row.get(key), str):
            row[key] = stamp(row[key])
    if isinstance(row.get("game_date_et"), str):
        row["game_date_et"] = datetime.fromisoformat(row["game_date_et"]).date()
    return row


def _plain_market_row(row):
    row = dict(row)
    for key in TIME_FIELDS:
        if isinstance(row.get(key), datetime):
            row[key] = iso(row[key])
    if row.get("game_date_et") is not None and not isinstance(row["game_date_et"], str):
        row["game_date_et"] = row["game_date_et"].isoformat()
    return row


def _price_rows(records, horizons):
    """One observation per row; tipoff references can never masquerade as cutoff prices."""
    from nba_dataset import remap_paths
    rows = []
    by_id = {}
    for i, record in enumerate(records):
        if record["game_id"] in by_id:
            raise ValueError("Duplicate market game ID")
        if OUTCOME_FIELDS.intersection(record):
            raise ValueError("Outcome fields in market price records")
        by_id[record["game_id"]] = record
        common = {k: record[k] for k in MARKET_COMMON}
        common.update({k: record[k] for k in ("market_id", "home_outcome", "market_start_utc")})
        common.update(row_order=i, price_method="last_clob_point", source=_encode_source(record["source"]),
                      source_fields=list(record["source"]))
        cutoff = {**common, "record_kind": "cutoff", "cutoff_utc": record["cutoff_utc"],
                  **{k: record[k] for k in ("home_probability", "price_time_utc", "minutes_before_cutoff")}}
        tipoff = {**common, "record_kind": "tipoff_reference", "cutoff_utc": record["scheduled_start_utc"],
                  "home_probability": record["tipoff_home_probability"], "price_time_utc": record["tipoff_price_time_utc"]}
        rows.extend((_typed_market_row(cutoff), _typed_market_row(tipoff)))
    seen = set()
    for i, record in enumerate(horizons):
        if record["game_id"] in seen or record["game_id"] not in by_id:
            raise ValueError("Duplicate or unmatched horizon game ID")
        seen.add(record["game_id"])
        cutoff = by_id[record["game_id"]]
        common = {k: record[k] for k in MARKET_COMMON}
        if any(common[k] != cutoff[k] for k in MARKET_COMMON):
            raise ValueError("Market and horizon game metadata differ")
        for j, h in enumerate(record["horizons"]):
            unknown = set(h) - set(HORIZON_FIELDS)
            if unknown:
                raise ValueError(f"Unknown horizon fields: {sorted(unknown)}")
            row = {**common, "row_order": i, "horizon_order": j, "record_kind": "horizon",
                   "price_method": "trade_validated_clob_point", "source": _encode_source(cutoff["source"]),
                   "source_fields": list(cutoff["source"]), "trade_pages": remap_paths(record["trade_pages"]),
                   "horizon_fields": list(h), "cutoff_utc": h["anchor_utc"],
                   "price_time_utc": h["history_point_utc"], "home_probability": h["history_probability"],
                   "minutes_before_cutoff": h["history_age_minutes"],
                   **{k: v for k, v in h.items() if k not in
                      ("anchor_utc", "history_point_utc", "history_probability", "history_age_minutes")}}
            rows.append(_typed_market_row(row))
    return rows


def write_market_tables(directory, records, audit, horizons=()):
    """Write shared derived tables; original response files are never changed."""
    from nba_dataset import remap_paths, write_table
    directory = Path(directory)
    prices_schema, audit_schema = market_schemas()
    result = {"market_prices.parquet": write_table(directory / "market_prices.parquet",
                                                   _price_rows(records, horizons), prices_schema)}
    audit_rows = [{**remap_paths(row), "row_order": i, "source_fields": list(row.get("price_source", {}))}
                  for i, row in enumerate(audit)]
    for row in audit_rows:
        if "price_source" in row:
            row["price_source"] = _encode_source(row["price_source"])
    if len({r["game_id"] for r in audit_rows}) != len(audit_rows):
        raise ValueError("Duplicate audit game ID")
    result["market_audit.parquet"] = write_table(directory / "market_audit.parquet", audit_rows, audit_schema)
    return result


def load_price_observations(*, root=ROOT, processed_dir=None, include_tipoff=False):
    """Pregame view excludes later tipoff references unless requested explicitly."""
    directory = Path(processed_dir) if processed_dir is not None else Path(root) / "data/processed"
    rows = [_plain_market_row(r) for r in read_rows(directory / "market_prices.parquet")]
    keys = [(r["record_kind"], r["game_id"], r["cutoff_utc"], r["price_method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate market price observation")
    for row in rows:
        if row["record_kind"] not in ("cutoff", "tipoff_reference", "horizon"):
            raise ValueError("Unknown market observation kind")
        cutoff, tipoff = stamp(row["cutoff_utc"]), stamp(row["scheduled_start_utc"])
        if row["record_kind"] == "tipoff_reference":
            if cutoff != tipoff:
                raise ValueError("Tipoff reference has another anchor")
        elif cutoff >= tipoff:
            raise ValueError("Pregame market cutoff must precede tipoff")
        if row["record_kind"] == "horizon" and (
                row["hours_before_tipoff"] <= 0 or cutoff != tipoff - timedelta(hours=row["hours_before_tipoff"])):
            raise ValueError("Market horizon disagrees with its cutoff")
        if row["price_time_utc"] is not None and stamp(row["price_time_utc"]) > stamp(row["cutoff_utc"]):
            raise ValueError("Market observation is later than its cutoff")
        if row["home_probability"] is not None and not 0 <= row["home_probability"] <= 1:
            raise ValueError("Invalid market probability")
    return [r for r in rows if include_tipoff or r["record_kind"] != "tipoff_reference"]


def load_market_records(*, root=ROOT, processed_dir=None):
    """Reconstruct the evaluation record, including its explicitly named tipoff reference."""
    rows = load_price_observations(root=root, processed_dir=processed_dir, include_tipoff=True)
    tips = {r["game_id"]: r for r in rows if r["record_kind"] == "tipoff_reference"}
    records = []
    for row in sorted((r for r in rows if r["record_kind"] == "cutoff"), key=lambda r: r["row_order"]):
        tip = tips.pop(row["game_id"], None)
        if tip is None:
            raise ValueError("Missing tipoff reference")
        record = {k: row[k] for k in MARKET_COMMON + ("market_id", "home_outcome", "market_start_utc", "cutoff_utc",
                                                     "home_probability", "price_time_utc", "minutes_before_cutoff")}
        record.update(tipoff_home_probability=tip["home_probability"], tipoff_price_time_utc=tip["price_time_utc"],
                      source=_decode_source(row["source"], row["source_fields"]))
        records.append(record)
    if tips:
        raise ValueError("Tipoff reference without matching cutoff record")
    return records


def load_horizon_records(*, root=ROOT, processed_dir=None):
    rows = load_price_observations(root=root, processed_dir=processed_dir)
    grouped = {}
    rename = {"anchor_utc": "cutoff_utc", "history_point_utc": "price_time_utc",
              "history_probability": "home_probability", "history_age_minutes": "minutes_before_cutoff"}
    for row in sorted((r for r in rows if r["record_kind"] == "horizon"),
                      key=lambda r: (r["row_order"], r["horizon_order"])):
        record = grouped.setdefault(row["game_id"], {**{k: row[k] for k in MARKET_COMMON},
                                                     "trade_pages": row["trade_pages"], "horizons": []})
        h = {k: row[rename.get(k, k)] for k in row["horizon_fields"]}
        if h["hours_before_tipoff"].is_integer():
            h["hours_before_tipoff"] = int(h["hours_before_tipoff"])
        record["horizons"].append(h)
    return list(grouped.values())


def load_market_audit(*, root=ROOT, processed_dir=None):
    directory = Path(processed_dir) if processed_dir is not None else Path(root) / "data/processed"
    records = []
    for row in sorted(read_rows(directory / "market_audit.parquet"), key=lambda r: r["row_order"]):
        record = {k: row[k] for k in ("game_id", "season", "decision", "candidate_slugs")}
        if row["market"] is not None:
            record["market"] = row["market"]
        if row["price_source"] is not None:
            record["price_source"] = _decode_source(row["price_source"], row["source_fields"])
        records.append(record)
    return records


def migrate_market_data(root, stage_dir):
    """Lossless conversion hook; reads frozen files and only writes the staging directory."""
    from nba_dataset import remap_paths
    root, stage_dir = Path(root), Path(stage_dir)
    legacy = root / "data/processed/polymarket"
    names = ("market_probs.jsonl", "horizon_prices.jsonl", "audit.jsonl")
    known = {*names, "comparison_test.csv", "trades_fetch.log"}
    unknown = [p.name for p in legacy.iterdir() if p.name not in known or not p.is_file() or p.is_symlink()]
    if unknown:
        raise ValueError(f"Unrecognized market migration inputs: {sorted(unknown)}")
    paths = [legacy / name for name in names]
    records, horizons, audit = [read_rows(p) for p in paths]
    tables = write_market_tables(stage_dir, records, audit, horizons)
    rebuilt = (load_market_records(root=root, processed_dir=stage_dir),
               load_horizon_records(root=root, processed_dir=stage_dir),
               load_market_audit(root=root, processed_dir=stage_dir))
    for name, before, after in zip(names, (records, horizons, audit), rebuilt):
        if remap_paths(before) != after:
            raise ValueError(f"Market migration does not round trip: {name}")
    moves = {}
    replacements = {"market_probs.jsonl": "market_prices.parquet: cutoff and tipoff_reference records, exact round trip",
                    "horizon_prices.jsonl": "market_prices.parquet: horizon records and trade-page references, exact round trip",
                    "audit.jsonl": "market_audit.parquet: complete selection/resolution metadata, exact round trip"}
    comparison = legacy / "comparison_test.csv"
    if comparison.exists():
        target = "reports/polymarket/comparison_test.csv"
        export = stage_dir / "exports" / target
        export.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(comparison, export)
        if export.read_bytes() != comparison.read_bytes():
            raise ValueError("Market comparison export differs from the original")
        paths.append(comparison)
        moves[str(export.relative_to(stage_dir))] = target
        replacements[comparison.name] = target + ": byte-identical report export"
    log = legacy / "trades_fetch.log"
    if log.exists():
        body = log.read_bytes()
        target = "reports/polymarket/collection.json"
        export = stage_dir / "exports" / target
        write_json(export, {"trades_fetch_log": body.decode("utf-8"),
                            "source_path": str(log.relative_to(root)), "source_sha256": hashlib.sha256(body).hexdigest()})
        if json.loads(export.read_text())["trades_fetch_log"].encode("utf-8") != body:
            raise ValueError("Market collection log does not round trip")
        paths.append(log)
        moves[str(export.relative_to(stage_dir))] = target
        replacements[log.name] = target + ": trades_fetch_log retains the complete original text"
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"tables": list(tables),
            "source_sha256": hashes,
            "cleanup_coverage": {str(p.relative_to(root)): {"sha256": hashes[str(p.relative_to(root))],
                                                           "replacement": replacements[p.name]} for p in paths},
            "equality": {"market_records": len(records), "horizon_records": len(horizons),
                         "horizon_observations": sum(len(r["horizons"]) for r in horizons), "market_audit": len(audit)},
            "install_files": moves, "remove_candidates": ["data/processed/polymarket"]}


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
    parser.add_argument("--output", type=Path, default=DATA_DIR, help="Unpublished directory for shared market tables")
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise ValueError("The shared dataset is frozen; collect into a new staged dataset before publishing changes")
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
    write_market_tables(args.output, records, audit)
    (REPORT_DIR / "listing_failures.json").unlink(missing_ok=True)
    summary = summarize(records, audit, games, failures, config)
    write_json(REPORT_DIR / "coverage.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
