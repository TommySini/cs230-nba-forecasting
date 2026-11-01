"""Validate Polymarket market selection, cutoff prices and provenance from cached sources."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json

from collect_pilot import ROOT, Cache, stamp
from collect_polymarket import (CONFIG_PATH, DATA_DIR, OUTCOME_FIELDS, candidate_markets, choose_market,
                                discover_markets, index_markets, load_games, market_record, price_points, read_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    games = load_games(config)
    records = read_rows(DATA_DIR / "market_probs.jsonl")
    audit = read_rows(DATA_DIR / "audit.jsonl")
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
        body = (ROOT / record["source"]["raw_path"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == record["source"]["sha256"], "Price-history checksum mismatch"
        rebuilt, _ = market_record(game, market, price_points(body), record["source"], config)
        assert rebuilt == record, f"Cutoff price does not reproduce: {record['game_id']}"
        assert stamp(record["price_time_utc"]) <= stamp(record["cutoff_utc"])
        assert stamp(record["tipoff_price_time_utc"]) <= stamp(record["scheduled_start_utc"])
        assert 0 <= record["minutes_before_cutoff"] <= config["max_staleness_minutes"]
    matched = sum(r["decision"] == "matched_price_at_cutoff" for r in audit)
    assert matched == len(records)
    print(f"PASS: {len(audit)} games audited; {len(records)} cutoff prices reproduce from checksummed raw responses; "
          "market selection reproduces offline; no outcome fields; no price after cutoff.")


if __name__ == "__main__":
    main()
