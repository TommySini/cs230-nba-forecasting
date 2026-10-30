"""Validate the completed pilot's joins, provenance and temporal boundaries."""
import hashlib
import json
from pathlib import Path

from collect_pilot import ROOT, prior_features, stamp, parse_injury


def rows(name):
    return [json.loads(line) for line in (ROOT / "data/processed" / name).read_text().splitlines()]


def main():
    inputs, labels, history, audit, manifest = [rows(name) for name in (
        "inputs.jsonl", "labels.jsonl", "game_history.jsonl", "audit.jsonl", "sample_manifest.jsonl")]
    config = json.loads((ROOT / "config/pilot.json").read_text())
    expected = sum(s["sample_size"] for s in config["seasons"])
    assert len(inputs) == expected, (len(inputs), expected)
    ids = [r["game_id"] for r in inputs]
    assert len(ids) == len(set(ids))
    for group in (labels, audit, manifest):
        assert [r["game_id"] for r in group] == ids
    game_by_id = {g["game_id"]: g for g in history}
    report_count = 0
    for record, label in zip(inputs, labels):
        game = game_by_id[record["game_id"]]
        assert label["home_win"] == int(game["home_score"] > game["away_score"])
        assert label["home_score"] == game["home_score"]
        assert label["away_score"] == game["away_score"]
        assert not {"home_win", "home_score", "away_score"}.intersection(record)
        assert record["narrative_text"] is None
        assert (stamp(record["scheduled_start_utc"]) - stamp(record["cutoff_utc"])).total_seconds() == config["cutoff_minutes"] * 60
        for side in ("home", "away"):
            assert record[side + "_features"] == prior_features(game, history, side)
            for prior_id in record[side + "_features"]["prior_game_ids"]:
                assert game_by_id[prior_id]["game_date_et"] < record["game_date_et"]
                assert prior_id != record["game_id"]
        injury = record["injury_report"]
        if injury:
            report_count += 1
            assert stamp(injury["report_time_utc"]) <= stamp(record["cutoff_utc"])
            assert injury["game_date_et"] == record["game_date_et"]
            assert injury["matchup"] == record["away_abbr"] + "@" + record["home_abbr"]
            body = (ROOT / injury["source"]["raw_path"]).read_bytes()
            assert hashlib.sha256(body).hexdigest() == injury["source"]["sha256"]
            report_time, blocks = parse_injury(body)
            assert report_time == injury["report_time_utc"]
            matches = [b for b in blocks if b["game_date_et"] == record["game_date_et"] and b["matchup"] == injury["matchup"]]
            assert len(matches) == 1 and matches[0]["text"] == injury["text"]
    for row in audit:
        assert not row["preview"]["included_in_inputs"]
        assert not row["team_notes"]["included_in_inputs"]
    print(f"PASS: {len(inputs)} unique games; aligned inputs/labels/audit; prior-only features; {report_count} source-verified injury excerpts; no unapproved narratives.")


if __name__ == "__main__":
    main()
