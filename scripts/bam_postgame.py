"""Normalize postgame evidence for TRAIN hindsight teachers, never predictors.

Only explicit score, schedule, recap article and boxscore fields are rendered.
Raw ESPN responses/news/recommendations/win-probability objects are excluded.
"""
import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import tempfile

from bam_data import ROOT, digest, read_rows, stamp, validate_snapshot, write_json, write_rows, read_snapshot_inputs, snapshot_manifest_path


class ArticleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("p", "br", "div") and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag in ("p", "div") and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


def text_only(value):
    parser = ArticleText()
    parser.feed(value)
    return "\n".join(" ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip())


def completed_header(document, game_id):
    package = document.get("gamepackageJSON", document)
    if "events" in document:
        found = [event for event in document["events"] if "espn:" + str(event.get("id")) == game_id]
        if len(found) != 1:
            raise ValueError("Scoreboard does not identify exactly one target event")
        header = found[0]
        prefix = "/events/" + str(document["events"].index(header))
    else:
        header = package.get("header", {})
        prefix = "/gamepackageJSON/header" if "gamepackageJSON" in document else "/header"
    if "espn:" + str(header.get("id")) != game_id:
        raise ValueError("Postgame document identifies a different event")
    competitions = header.get("competitions", [])
    if len(competitions) != 1:
        raise ValueError("Ambiguous competition")
    competition = competitions[0]
    status = competition.get("status", header.get("status", {}))
    if status.get("type", {}).get("completed") is not True:
        raise ValueError("Postgame evidence is not a completed game")
    return package, header, competition, prefix


def normalize_document(game, document, provenance, kind):
    """Return citations over normalized, explicitly whitelisted source fields."""
    if game.get("split") != "train":
        raise ValueError("Postgame access is allowed only for training games")
    if provenance.get("status") != 200 or not provenance.get("sha256"):
        raise ValueError("Postgame source lacks verified raw-byte provenance")
    if stamp(provenance["retrieved_at_utc"]) <= stamp(game["cutoff_utc"]):
        raise ValueError("Postgame response was retrieved before prediction cutoff")
    package, header, competition, prefix = completed_header(document, game["game_id"])
    scheduled = competition.get("date", header.get("date"))
    if stamp(scheduled) != stamp(game["scheduled_start_utc"]):
        raise ValueError("Postgame schedule differs from snapshot")
    lines = []
    def add(pointer, value, field):
        if value is None or isinstance(value, (dict, list)):
            return
        lines.append({"line": len(lines) + 1, "text": field + ": " + str(value), "source_pointer": pointer})
    add(prefix + "/id", header["id"], "ESPN game ID")
    add(prefix + "/competitions/0/date", scheduled, "Scheduled start")
    competitors = competition.get("competitors", [])
    if len(competitors) != 2 or {c.get("homeAway") for c in competitors} != {"home", "away"}:
        raise ValueError("Expected exactly one home and away competitor")
    scores = {}
    for index, competitor in enumerate(competitors):
        side = competitor["homeAway"]
        score = competitor.get("score")
        score = score.get("value", score.get("displayValue")) if isinstance(score, dict) else score
        scores[side] = int(score)
        abbr = competitor.get("team", {}).get("abbreviation")
        from collect_pilot import ABBR
        abbr = ABBR.get(abbr, abbr)
        if abbr != game[side + "_abbr"]:
            raise ValueError("Postgame competitor differs from snapshot")
        if side + "_score" in game and scores[side] != game[side + "_score"]:
            raise ValueError("Postgame score differs from verified numerical history")
        path = prefix + "/competitions/0/competitors/" + str(index)
        add(path + "/team/abbreviation", abbr, side + " team")
        add(path + "/score", scores[side], side + " final score")
    if scores["home"] == scores["away"]:
        raise ValueError("Tied NBA result")
    content_types = ["final_score", "schedule"]
    article = package.get("article", {})
    if isinstance(article, dict) and article.get("type", "").lower() == "recap":
        event_ids = {str(c.get("eventId", c.get("event", {}).get("id"))) for c in article.get("categories", []) if isinstance(c, dict)}
        event_ids.add(str(article.get("gameId")))
        if game["game_id"].split(":")[-1] in event_ids and isinstance(article.get("story"), str):
            article_prefix = "/gamepackageJSON/article" if "gamepackageJSON" in document else "/article"
            for field in ("headline", "published", "lastModified"):
                add(article_prefix + "/" + field, article.get(field), "Recap " + field)
            for paragraph in text_only(article["story"]).splitlines():
                add(article_prefix + "/story", paragraph, "Recap paragraph")
            content_types.append("recap")
    box = package.get("boxscore", {})
    box_prefix = "/gamepackageJSON/boxscore" if "gamepackageJSON" in document else "/boxscore"
    if isinstance(box, dict):
        for index, team in enumerate(box.get("teams", [])):
            path = box_prefix + "/teams/" + str(index)
            add(path + "/team/abbreviation", team.get("team", {}).get("abbreviation"), "Boxscore team")
            for number, stat in enumerate(team.get("statistics", [])):
                label = stat.get("label", stat.get("name"))
                if label is not None:
                    add(path + "/statistics/" + str(number) + "/displayValue", stat.get("displayValue"), "Team " + str(label))
        for team_number, team in enumerate(box.get("players", [])):
            path = box_prefix + "/players/" + str(team_number)
            add(path + "/team/abbreviation", team.get("team", {}).get("abbreviation"), "Player boxscore team")
            for group_number, group in enumerate(team.get("statistics", [])):
                labels = group.get("labels", [])
                for player_number, player in enumerate(group.get("athletes", [])):
                    player_path = path + "/statistics/" + str(group_number) + "/athletes/" + str(player_number)
                    name = player.get("athlete", {}).get("displayName", "unknown")
                    add(player_path + "/athlete/displayName", name, "Player")
                    for field in ("starter", "didNotPlay", "reason"):
                        add(player_path + "/" + field, player.get(field), str(name) + " " + field)
                    stats = player.get("stats", [])
                    if player.get("didNotPlay") is True and stats == []:
                        # ESPN encodes DNP rows without a statistics vector.
                        # Keep the explicit DNP/reason above; never fill zeros.
                        continue
                    if len(stats) != len(labels):
                        raise ValueError("Player boxscore labels and values do not align")
                    for stat_number, (label, value) in enumerate(zip(labels, stats)):
                        add(player_path + "/stats/" + str(stat_number), value, str(name) + " " + str(label))
        if box.get("teams") or box.get("players"):
            content_types.append("boxscore")
    return {"source_id": "sha256:" + provenance["sha256"] + ":postgame:" + game["game_id"], "publisher": "ESPN", "url": provenance["url"],
            "sha256": provenance["sha256"], "archive_time": provenance["retrieved_at_utc"],
            "retrieved_at_utc": provenance["retrieved_at_utc"], "raw_path": provenance.get("raw_path"),
            "time_semantics": "postgame retrieval timestamp; not pregame archive evidence",
            "source_kind": kind, "content_types": content_types,
            "pages": [{"page": 1, "lines": lines}],
            "normalization": "Explicit JSON field whitelist with JSON-pointer citations; raw ESPN objects excluded"}


def normalize_optional_document(game, body, provenance, kind):
    """Quarantine a malformed optional rich source, preserving score fallback."""
    try:
        return normalize_document(game, json.loads(body), provenance, kind), None
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return None, {"game_id": game["game_id"], "source_kind": kind,
                      "url": provenance.get("url"), "sha256": provenance.get("sha256"),
                      "raw_path": provenance.get("raw_path"), "decision": "quarantined_not_teacher_input",
                      "error_type": type(error).__name__, "reason": str(error)}


def verified_json(path, root=ROOT):
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    body = path.read_bytes()
    if digest(body) != meta.get("sha256") or meta.get("status") != 200:
        raise ValueError("Postgame raw cache checksum/status mismatch")
    from nba_dataset import resolve_source_path
    if resolve_source_path(root, meta["raw_path"]).resolve() != path.resolve():
        raise ValueError("Postgame cache provenance path mismatch")
    return json.loads(body), meta


def prepare(snapshot, output, fetch_missing=False):
    from collect_pilot import Cache
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    validate_snapshot(snapshot, root=ROOT, label_splits=("train",))
    games = read_snapshot_inputs(snapshot, root=ROOT, splits=("train",))
    manifest = json.loads(snapshot_manifest_path(snapshot).read_text())
    from nba_dataset import Dataset, resolve_source_path
    if (snapshot / "bam_manifest.json").exists():
        train_seasons = [int(season) for season, split in manifest["season_splits"].items() if split == "train"]
        history = {g["game_id"]: g for g in Dataset(ROOT, processed_dir=snapshot).history(seasons=train_seasons)}
    else:
        history_path = ROOT / "data/processed/seasonal/game_history.jsonl"
        history = {g["game_id"]: g for g in read_rows(history_path)}
        history_hash = manifest["identity"]["provenance_sha256"].get("data/processed/seasonal/game_history.jsonl")
        if digest(history_path.read_bytes()) != history_hash:
            raise ValueError("Numerical history differs from input snapshot")
    if output.exists():
        raise FileExistsError("Postgame outputs are immutable; choose another output directory")
    cache = Cache(offline=not fetch_missing)
    records, missing, quarantined = [], [], []
    for game in games:
        target = {**game, "home_score": history[game["game_id"]]["home_score"], "away_score": history[game["game_id"]]["away_score"]}
        scoreboard = resolve_source_path(ROOT, history[game["game_id"]]["schedule_source"])
        document, provenance = verified_json(scoreboard)
        sources = [normalize_document(target, document, provenance, "scoreboard")]
        espn_id = game["game_id"].split(":")[-1]
        candidates = [(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={espn_id}", "summaries"),
                      (f"https://cdn.espn.com/core/nba/preview?xhr=1&gameId={espn_id}", "previews")]
        for url, kind in candidates:
            # Offline lookups first. Only the explicit fetch flag permits an API
            # request; previews are always historical cached responses.
            body, meta = (cache if kind == "summaries" else Cache(offline=True)).get(url, kind, ".json")
            if body is None:
                continue
            source, error = normalize_optional_document(target, body, meta, kind)
            if error is not None:
                quarantined.append(error)
                continue
            if "recap" in source["content_types"] or "boxscore" in source["content_types"]:
                sources.append(source)
        rich = {kind for source in sources for kind in source["content_types"]}
        if "recap" not in rich or "boxscore" not in rich:
            missing.append({"game_id": game["game_id"], "missing": sorted({"recap", "boxscore"} - rich), "summary_url": candidates[0][0]})
        records.append({"game_id": game["game_id"], "split": "train", "normalized": True,
                        "purpose": "hindsight_teacher_only", "snapshot_id": manifest["snapshot_id"], "sources": sources})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preparing-postgame-", dir=output.parent))
    try:
        write_rows(staging / "records.jsonl", records)
        write_rows(staging / "missing_rich_sources.jsonl", missing)
        write_rows(staging / "quarantined_sources.jsonl", quarantined)
        report = {"snapshot_id": manifest["snapshot_id"], "snapshot_manifest_sha256": digest(snapshot_manifest_path(snapshot).read_bytes()),
                  "normalizer_sha256": digest(Path(__file__).read_bytes()), "training_games": len(records),
                  "missing_rich_sources": len(missing),
                  "quarantined_sources": len(quarantined),
                  "source_content_counts": dict(Counter(kind for record in records for source in record["sources"] for kind in source["content_types"])),
                  "artifact_sha256": {p.name: digest(p.read_bytes()) for p in staging.iterdir()}}
        write_json(staging / "manifest.json", report)
        staging.rename(output)
        print(json.dumps(report, indent=2))
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-missing", action="store_true", help="Explicitly permit fetching missing ESPN summary API responses")
    args = parser.parse_args()
    prepare(args.snapshot, args.output, args.fetch_missing)


if __name__ == "__main__":
    main()
