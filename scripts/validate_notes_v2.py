"""Validate game-notes v2 from saved sources: checksums, pre-cutoff evidence, no results, no outcome fields."""
import argparse
from datetime import timedelta
from email.utils import parsedate_to_datetime
import hashlib
import json

from collect_notes_v2 import DATA, CONFIG_PATH, REPLAY, ROOT, load_history, read_rows, records_before, utc
from collect_pilot import stamp
from postgame_screen import postgame_flags

OUTCOME_FIELDS = {"home_score", "away_score", "home_win", "winner", "margin", "result"}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    history = load_history(config)
    games, before = {g["game_id"]: g for g in history}, records_before(history)
    approved = read_rows(DATA / "approved.jsonl")
    keys = [(d["game_id"], d["team"]) for d in approved]
    assert len(keys) == len(set(keys)), "More than one approved document per team-game"
    for d in approved:
        assert not d["reasons"], d
        assert not OUTCOME_FIELDS.intersection(d), f"Outcome field in document record: {d['game_id']}"
        game, cutoff = games[d["game_id"]], stamp(d["cutoff_utc"])
        assert cutoff == stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"])
        if d["kind"] == "html_post":
            assert stamp(d["published_utc"]) <= cutoff, f"Post published after cutoff: {d['url']}"
            api = ROOT / d["source"]["api_raw_path"]
            assert hashlib.sha256(api.read_bytes()).hexdigest() == d["source"]["api_sha256"], "API response checksum changed"
        else:
            source = d["source"]
            assert hashlib.sha256((ROOT / source["raw_path"]).read_bytes()).hexdigest() == source["sha256"], \
                f"Raw document checksum changed: {source['raw_path']}"
            if d["tier"] == "archive_proven":
                match = REPLAY.fullmatch(source["replay_url"].replace("http://", "https://", 1))
                assert match and utc(match[1]) == parsedate_to_datetime(source["memento_datetime"]) <= cutoff, \
                    f"Archive replay not before cutoff: {source['replay_url']}"
            else:
                assert parsedate_to_datetime(source["last_modified"]) <= cutoff, f"CDN file modified after cutoff: {source['url']}"
        text = (ROOT / d["text_path"]).read_text()
        flags = postgame_flags(text, dict(game, home_record_before=before[d["game_id"], "home"],
                                          away_record_before=before[d["game_id"], "away"]))
        assert not flags, f"Result of the game found in {d['text_path']}: {flags}"
    tiers = {t: sum(d["tier"] == t for d in approved) for t in ("archive_proven", "site_timestamped")}
    print(f"PASS: {len(approved)} team-game documents ({len({d['game_id'] for d in approved})} games; {tiers}); "
          "checksums reproduce, evidence predates every cutoff, no game results in text, no outcome fields.")


if __name__ == "__main__":
    main()
