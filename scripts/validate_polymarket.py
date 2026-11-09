"""Validate Polymarket market selection, cutoff prices and provenance from cached sources."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

from collect_pilot import ROOT, Cache, stamp
from collect_polymarket import (CONFIG_PATH, OUTCOME_FIELDS, candidate_markets, choose_market,
                                discover_markets, index_markets, load_games, market_record, price_points,
                                load_market_records, load_horizon_records, load_market_audit)
from collect_polymarket_trades import horizon_row
from nba_dataset import Dataset, resolve_source_path


def validate(processed_dir=None):
    config = json.loads(CONFIG_PATH.read_text())
    dataset = Dataset(ROOT, processed_dir)
    dataset.verify()
    games = load_games(config, processed_dir=processed_dir)
    records = load_market_records(processed_dir=processed_dir)
    audit = load_market_audit(processed_dir=processed_dir)
    assert [r["game_id"] for r in audit] == [g["game_id"] for g in games], "Audit must cover every configured game once"
    ids = [r["game_id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate market rows"
    with redirect_stdout(io.StringIO()):
        markets, failures = discover_markets(Cache(offline=True), config, games)
    assert not failures, "Market listing is not fully cached"
    index = index_markets(markets)
    by_id = {g["game_id"]: g for g in games}
    audit_by_id = {r["game_id"]: r for r in audit}
    for record in records:
        game = by_id[record["game_id"]]
        assert not OUTCOME_FIELDS.intersection(record), "Outcome or resolution fields in market probabilities"
        market, decision = choose_market(candidate_markets(game, index, config["max_start_difference_minutes"]), game["game_date_et"])
        assert decision == "matched" and market["slug"] == record["market_slug"], record["game_id"]
        assert market["tokens"][market["home_index"]] == record["home_token_id"]
        assert audit_by_id[record["game_id"]]["decision"] == "matched_price_at_cutoff"
        body = resolve_source_path(ROOT, record["source"]["raw_path"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == record["source"]["sha256"], "Price-history checksum mismatch"
        rebuilt, _ = market_record(game, market, price_points(body), record["source"], config)
        assert rebuilt == record, f"Cutoff price does not reproduce: {record['game_id']}"
        assert stamp(record["price_time_utc"]) <= stamp(record["cutoff_utc"])
        assert stamp(record["tipoff_price_time_utc"]) <= stamp(record["scheduled_start_utc"])
        assert 0 <= record["minutes_before_cutoff"] <= config["max_staleness_minutes"]
    matched = sum(r["decision"] == "matched_price_at_cutoff" for r in audit)
    assert matched == len(records)
    records_by_id = {r["game_id"]: r for r in records}
    horizons = load_horizon_records(processed_dir=processed_dir)
    for record in horizons:
        market = records_by_id[record["game_id"]]
        assert not OUTCOME_FIELDS.intersection(record)
        points = price_points(resolve_source_path(ROOT, market["source"]["raw_path"]).read_bytes())
        trades = []
        for page in record["trade_pages"]:
            body = resolve_source_path(ROOT, page["raw_path"]).read_bytes()
            assert hashlib.sha256(body).hexdigest() == page["sha256"], "Trade-page checksum mismatch"
            trades.extend(json.loads(body)["data"])
        for h in record["horizons"]:
            assert not OUTCOME_FIELDS.intersection(h)
            assert h == horizon_row(market, points, trades, h["hours_before_tipoff"], config["horizons"]), \
                f"Horizon does not reproduce from source trades: {record['game_id']} at {h['hours_before_tipoff']}h"
            if h["history_point_utc"] is not None:
                assert stamp(h["history_point_utc"]) <= stamp(h["anchor_utc"])
    print(f"PASS: {len(audit)} games audited; {len(records)} cutoff prices reproduce from checksummed raw responses; "
          f"{sum(len(r['horizons']) for r in horizons)} horizons reproduce from checksummed trade pages; "
          "market selection reproduces offline; no outcome fields; no price after cutoff.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Staged or published processed dataset directory")
    args = parser.parse_args()
    validate(args.dataset)


if __name__ == "__main__":
    main()
