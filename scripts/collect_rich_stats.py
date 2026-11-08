"""Collect allowlisted historical box scores for prior-game numeric features.

The NBA Stats bulk endpoint timed out during the recorded access probe. This
collector therefore uses ESPN's public summary endpoint, with the same verified
game identities as the original baseline. Raw summaries are retained only as
provenance: recaps, current player profiles, news, odds and target availability
never enter the normalized output. These are retrospectively retrieved final
box scores; only earlier-date records may be used for any forecast.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata

from collect_pilot import ABBR, Cache, ROOT, iso, stamp, write_json, write_jsonl

COUNTERS = ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
            "reb", "ast", "tov", "stl", "blk", "pf", "pts")
PAIR_NAMES = {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
              "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
              "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta")}
TEAM_NAMES = {"offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
              "totalRebounds": "reb", "assists": "ast", "steals": "stl",
              "blocks": "blk", "fouls": "pf"}
PLAYER_NAMES = {"PTS": "pts", "REB": "reb", "AST": "ast", "TO": "tov",
                "STL": "stl", "BLK": "blk", "OREB": "oreb", "DREB": "dreb", "PF": "pf"}
DEFAULT_OUTPUT = ROOT / "data/processed/rich_stats_v1"
NBA_FRANCHISES = frozenset("ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS".split())
NBA_REPAIR_IDS = {"espn:401704652": "0022400086", "espn:401704937": "0022400371",
                  "espn:401705058": "0022400492", "espn:401809936": "0022500083",
                  "espn:401810469": "0022500614", "espn:401811030": "0022501175"}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def integer(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a box-score count")
    out = float(value)
    if not math.isfinite(out) or out < 0 or out != int(out):
        raise ValueError(f"Invalid nonnegative integer: {value!r}")
    return int(out)


def minutes(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a minute value")
    if isinstance(value, str) and ":" in value:
        first, second = value.split(":")
        seconds = integer(second)
        if seconds >= 60:
            raise ValueError("Invalid minute seconds")
        return integer(first) + seconds / 60
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Invalid minutes")
    return result


def made_attempted(value):
    parts = str(value).split("-")
    if len(parts) != 2:
        raise ValueError("Expected made-attempted pair")
    made, attempted = map(integer, parts)
    if made > attempted:
        raise ValueError("Made count exceeds attempts")
    return made, attempted


def check_counters(row):
    missing = sorted(set(COUNTERS) - row.keys())
    if missing:
        raise ValueError("Missing box-score counters: " + ",".join(missing))
    for name in COUNTERS:
        integer(row[name])
    for made, attempted in (("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")):
        if row[made] > row[attempted]:
            raise ValueError("Made count exceeds attempts")
    if row["fg3m"] > row["fgm"] or row["fg3a"] > row["fga"]:
        raise ValueError("Three-point counts exceed all field goals")
    if row["fgm"] - row["fg3m"] > row["fga"] - row["fg3a"]:
        raise ValueError("Two-point makes exceed two-point attempts")
    if 2 * row["fgm"] + row["fg3m"] + row["ftm"] != row["pts"]:
        raise ValueError("Points do not reconcile to shooting counts")
    if row["oreb"] + row["dreb"] != row["reb"]:
        raise ValueError("Rebound counts do not reconcile")


def normalize_players(group):
    players, issues, seen = [], [], set()
    for block in group.get("statistics", []):
        labels = block.get("labels", [])
        if len(labels) != len(set(labels)):
            raise ValueError("Repeated player statistic label")
        for athlete in block.get("athletes", []):
            # A present-day athlete.active/position/profile object is deliberately
            # ignored; only dated game participation and box-score fields count.
            if athlete.get("didNotPlay") is True:
                continue
            raw_identity = athlete.get("athlete", {}).get("id")
            if raw_identity is None or isinstance(raw_identity, bool):
                issues.append({"reason": "missing_participated_player_id"})
                continue
            identity = str(raw_identity)
            if not identity.isdigit() or identity in seen:
                issues.append({"player_id": identity, "reason": "invalid_or_duplicate_participated_player_id"})
                continue
            seen.add(identity)
            values = athlete.get("stats", [])
            if len(values) != len(labels):
                issues.append({"player_id": identity, "reason": "misaligned_or_missing_stat_vector"})
                continue
            data = dict(zip(labels, values))
            try:
                row = {"player_id": identity, "minutes": minutes(data["MIN"]),
                       "starter": bool(athlete.get("starter", False))}
                for label, (made, attempted) in {"FG": ("fgm", "fga"), "3PT": ("fg3m", "fg3a"),
                                                "FT": ("ftm", "fta")}.items():
                    row[made], row[attempted] = made_attempted(data[label])
                row.update({name: integer(data[label]) for label, name in PLAYER_NAMES.items()})
                check_counters(row)
                players.append(row)
            except (ValueError, KeyError, TypeError) as error:
                issues.append({"player_id": identity, "reason": str(error)})
    if not players:
        issues.append({"reason": "no_participating_player_rows"})
    return players, issues


def completed_duration(status, competitors, scores):
    """Derive elapsed game minutes independently of player-row completeness.

    ESPN summaries often omit status.period but provide ordered quarter score
    arrays. Both sides must have the same 4+ period count and reconcile to their
    final score. Validate explicit period indices when supplied, and reconcile
    a separately supplied status.period rather than choosing one arbitrarily.
    """
    explicit = status.get("period")
    if explicit is not None:
        explicit = integer(explicit)
        if not 4 <= explicit <= 20:
            raise ValueError("Invalid completed-game period count")
    line_arrays = [c.get("linescores") for c in competitors]
    has_lines = [bool(lines) for lines in line_arrays]
    line_count = None
    if any(has_lines):
        if not all(has_lines) or any(not isinstance(lines, list) for lines in line_arrays):
            raise ValueError("Both teams need ordered quarter linescores")
        counts = {len(lines) for lines in line_arrays}
        if len(counts) != 1 or not 4 <= next(iter(counts)) <= 20:
            raise ValueError("Quarter linescore counts disagree or are incomplete")
        line_count = next(iter(counts))
        for competitor, lines in zip(competitors, line_arrays):
            total = 0
            for index, line in enumerate(lines, 1):
                if not isinstance(line, dict):
                    raise ValueError("Malformed quarter linescore")
                if "period" in line and integer(line["period"]) != index:
                    raise ValueError("Quarter linescore indices are not chronological")
                values = [integer(line[key]) for key in ("value", "displayValue") if key in line]
                if not values or len(set(values)) != 1:
                    raise ValueError("Quarter linescore values are missing or inconsistent")
                total += values[0]
            if total != scores[competitor["homeAway"]]:
                raise ValueError("Quarter linescores do not reconcile to final score")
        if explicit is not None and explicit != line_count:
            raise ValueError("Status period disagrees with quarter linescores")
    periods = explicit if explicit is not None else line_count
    if periods is None:
        return None, "unknown"
    basis = "completed competition status.period" if line_count is None else "ordered completed quarter linescores reconciled to both final scores"
    if line_count is not None and explicit is not None:
        basis += "; matching status.period"
    return 48 + 5 * (periods - 4), basis


def normalize_summary(game, document, provenance):
    if any(ABBR.get(game[side + "_abbr"], game[side + "_abbr"]) not in NBA_FRANCHISES for side in ("home", "away")):
        raise ValueError("Historical matchup does not contain two NBA franchises")
    if provenance.get("status") != 200 or not provenance.get("sha256"):
        raise ValueError("Unverified source provenance")
    header = document.get("header", {})
    if str(header.get("id")) != str(game["espn_id"]):
        raise ValueError("Summary event ID does not match history")
    competitions = header.get("competitions", [])
    if len(competitions) != 1:
        raise ValueError("Expected one completed competition")
    comp = competitions[0]
    status = comp.get("status", header.get("status", {}))
    if status.get("type", {}).get("completed") is not True:
        raise ValueError("Historical box score is not completed")
    if stamp(comp.get("date", header.get("date"))) != stamp(game["scheduled_start_utc"]):
        raise ValueError("Summary date differs from verified history")
    competitors = comp.get("competitors", [])
    if len(competitors) != 2 or {c.get("homeAway") for c in competitors} != {"home", "away"}:
        raise ValueError("Invalid competitor identities")
    side_by_abbr, scores = {}, {}
    for competitor in competitors:
        side = competitor["homeAway"]
        abbr = competitor["team"]["abbreviation"]
        abbr = ABBR.get(abbr, abbr)
        if abbr != game[side + "_abbr"]:
            raise ValueError("Team does not match verified history")
        score = competitor["score"]
        if isinstance(score, dict):
            score = score.get("value", score.get("displayValue"))
        scores[side] = integer(score)
        if scores[side] != game[side + "_score"]:
            raise ValueError("Score does not match verified history")
        side_by_abbr[abbr] = side
    game_minutes, duration_basis = completed_duration(status, competitors, scores)
    box = document.get("boxscore", {})
    teams, player_groups = {}, {}
    for group in box.get("players", []):
        abbr = group.get("team", {}).get("abbreviation")
        abbr = ABBR.get(abbr, abbr)
        if abbr in player_groups or abbr not in side_by_abbr:
            raise ValueError("Unexpected or duplicate player-box team")
        player_groups[abbr] = group
    for team in box.get("teams", []):
        abbr = team.get("team", {}).get("abbreviation")
        abbr = ABBR.get(abbr, abbr)
        if abbr not in side_by_abbr or side_by_abbr[abbr] in teams:
            raise ValueError("Unexpected or duplicate team box score")
        side = side_by_abbr[abbr]
        row = {"team_abbr": abbr, "pts": scores[side]}
        stat_rows = team.get("statistics", [])
        stats = {s.get("name"): s.get("displayValue") for s in stat_rows}
        if len(stats) != len(stat_rows):
            raise ValueError("Repeated team statistic name")
        for label, (made, attempted) in PAIR_NAMES.items():
            row[made], row[attempted] = made_attempted(stats[label])
        row.update({name: integer(stats[label]) for label, name in TEAM_NAMES.items()})
        # ESPN separates individual turnovers and team turnovers. The team
        # possession denominator needs their total, not just player turnovers.
        if "totalTurnovers" in stats:
            row["tov"] = integer(stats["totalTurnovers"])
            if "turnovers" in stats and "teamTurnovers" in stats and row["tov"] != integer(stats["turnovers"]) + integer(stats["teamTurnovers"]):
                raise ValueError("Total turnovers do not reconcile")
            row["turnovers_definition"] = "ESPN totalTurnovers (player plus team)"
        elif "turnovers" in stats and "teamTurnovers" in stats:
            row["tov"] = integer(stats["turnovers"]) + integer(stats["teamTurnovers"])
            row["turnovers_definition"] = "ESPN turnovers plus teamTurnovers"
        else:
            raise ValueError("Missing total/team turnover distinction")
        check_counters(row)
        players, issues = normalize_players(player_groups.get(abbr, {}))
        row["players"] = players
        row["observed_player_minutes"] = sum(p["minutes"] for p in players)
        row["minutes"] = row["observed_player_minutes"]
        row["minutes_basis"] = "sum of parsed historical participant minutes"
        if game_minutes is not None:
            row["game_minutes"] = game_minutes
            row["minutes"] = 5 * game_minutes
            row["minutes_basis"] = "five players times completed game duration"
            if abs(row["observed_player_minutes"] - row["minutes"]) > 8:
                issues.append({"reason": "player_minutes_do_not_reconcile", "sum": row["observed_player_minutes"],
                               "expected": 5 * game_minutes})
        for field in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "ast", "stl", "blk"):
            total = sum(p[field] for p in players)
            if total != row[field]:
                issues.append({"reason": "player_total_mismatch", "field": field,
                               "player_sum": total, "team_total": row[field]})
        row.update(complete=True, missing_fields=[], players_complete=not issues, player_issues=issues)
        teams[side] = row
    if set(teams) != {"home", "away"}:
        raise ValueError("Both team box scores are required")
    return {"schema_version": "rich-stats-v1", "game_id": game["game_id"],
            "season": game["season"], "game_date_et": game["game_date_et"],
            "scheduled_start_utc": game["scheduled_start_utc"], "teams": teams,
            "game_minutes": game_minutes,
            "duration_basis": duration_basis,
            "complete": True, "players_complete": all(t["players_complete"] for t in teams.values()),
            "provenance": {"provider": "ESPN", "source_game_id": str(game["espn_id"]),
                           **{k: provenance.get(k) for k in ("url", "raw_path", "sha256", "retrieved_at_utc")}},
            "time_semantics": "Retrospectively retrieved final numerical box score; prior-date prediction use only; historical correction times unverified"}


def normalize_nba_page(game, body, provenance, espn_document, espn_provenance):
    """Read a dated NBA box score; ESPN supplies cross-provider player IDs only.

    The NBA page embeds other articles/charts/current profile data. Read only
    props.pageProps.game's historical box-score counters and explicit turnover
    totals; never render the raw page or whole game object into model inputs.
    """
    matches = re.findall(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', body.decode(), re.S)
    if len(matches) != 1:
        raise ValueError("NBA page must identify one structured data object")
    nba = json.loads(matches[0])["props"]["pageProps"]["game"]
    expected_id = NBA_REPAIR_IDS.get(game["game_id"])
    if not expected_id or nba.get("gameId") != expected_id or nba.get("gameStatus") != 3:
        raise ValueError("NBA repair game identity/status mismatch")
    if provenance["url"] != f"https://www.nba.com/game/{expected_id}/box-score":
        raise ValueError("NBA source URL mismatch")
    if stamp(nba["gameTimeUTC"]) != stamp(game["scheduled_start_utc"]):
        raise ValueError("NBA repair start time mismatch")
    if str(espn_document.get("header", {}).get("id")) != game["espn_id"]:
        raise ValueError("NBA player-ID mapping source is a different game")
    clean_name = lambda value: re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower())
    groups = {}
    for group in espn_document["boxscore"]["players"]:
        abbr = group["team"]["abbreviation"]
        abbr = ABBR.get(abbr, abbr)
        identities = {}
        for block in group["statistics"]:
            for athlete in block["athletes"]:
                if not athlete.get("athlete", {}).get("id") or not athlete.get("athlete", {}).get("displayName"):
                    continue
                name = clean_name(athlete["athlete"]["displayName"])
                if name in identities:
                    raise ValueError("Ambiguous cross-provider player identity")
                identities[name] = athlete
        groups[abbr] = identities
    nba_fields = dict(fgm="fieldGoalsMade", fga="fieldGoalsAttempted", fg3m="threePointersMade",
                      fg3a="threePointersAttempted", ftm="freeThrowsMade", fta="freeThrowsAttempted",
                      oreb="reboundsOffensive", dreb="reboundsDefensive", reb="reboundsTotal",
                      ast="assists", stl="steals", blk="blocks", pf="foulsPersonal", pts="points", tov="turnovers")
    competitors, teams, player_groups = [], [], []
    for side in ("home", "away"):
        team = nba[side + "Team"]
        abbr = team["teamTricode"]
        if abbr != game[side + "_abbr"] or integer(team["score"]) != game[side + "_score"]:
            raise ValueError("NBA repair team/score mismatch")
        counters = {name: integer(team["statistics"][field]) for name, field in nba_fields.items()}
        check_counters(counters)
        charts = nba["postgameCharts"][side + "Team"]
        if charts.get("teamId") != team["teamId"]:
            raise ValueError("NBA turnover totals team identity mismatch")
        turnovers = charts["statistics"]
        total_tov, team_tov = integer(turnovers["turnoversTotal"]), integer(turnovers["turnoversTeam"])
        if integer(turnovers["turnovers"]) != counters["tov"] or total_tov != counters["tov"] + team_tov:
            raise ValueError("NBA turnover totals do not reconcile")
        stat_values = {label: f'{counters[made]}-{counters[attempted]}' for label, (made, attempted) in PAIR_NAMES.items()}
        stat_values.update({label: str(counters[name]) for label, name in TEAM_NAMES.items()})
        stat_values.update(turnovers=str(counters["tov"]), teamTurnovers=str(team_tov), totalTurnovers=str(total_tov))
        teams.append({"team": {"abbreviation": abbr}, "statistics": [{"name": k, "displayValue": v} for k, v in stat_values.items()]})
        players = []
        for athlete in team["players"]:
            stats = athlete.get("statistics", {})
            duration = stats.get("minutes")
            if not duration and athlete.get("comment"):
                continue
            if not duration:
                raise ValueError("NBA player missing minutes without explicit nonparticipation comment")
            name = clean_name(athlete["firstName"] + " " + athlete["familyName"])
            canonical = groups[abbr].get(name, {})
            counts = {key: integer(stats[field]) for key, field in nba_fields.items()}
            check_counters(counts)
            vals = [duration, str(counts["pts"]), f'{counts["fgm"]}-{counts["fga"]}',
                    f'{counts["fg3m"]}-{counts["fg3a"]}', f'{counts["ftm"]}-{counts["fta"]}']
            vals += [str(counts[k]) for k in ("reb", "ast", "tov", "stl", "blk", "oreb", "dreb", "pf")]
            players.append({"athlete": {"id": canonical.get("athlete", {}).get("id")}, "starter": canonical.get("starter", False),
                            "didNotPlay": False, "stats": vals})
        player_groups.append({"team": {"abbreviation": abbr}, "statistics": [{
            "labels": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF"],
            "athletes": players}]})
        competitors.append({"homeAway": side, "team": {"abbreviation": abbr}, "score": team["score"]})
    adapter = {"header": {"id": game["espn_id"], "competitions": [{"date": nba["gameTimeUTC"],
                  "status": {"type": {"completed": True}, "period": integer(nba["period"])}, "competitors": competitors}]},
               "boxscore": {"teams": teams, "players": player_groups}}
    record = normalize_summary(game, adapter, provenance)
    record["provenance"].update(provider="NBA", source_game_id=expected_id,
        normalization="NBA historical box-score counters; exact normalized-name ESPN player-ID mapping from same historical game",
        identity_mapping_source={"provider": "ESPN", "source_game_id": game["espn_id"],
                                 **{k: espn_provenance[k] for k in ("url", "raw_path", "sha256", "retrieved_at_utc")}})
    try:
        normalize_summary(game, espn_document, espn_provenance)
        reason = "predeclared official NBA replacement"
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        reason = str(error)
    record["provenance"]["rejected_original_reason"] = reason
    for team in record["teams"].values():
        team["turnovers_definition"] = "NBA postgameCharts turnoversTotal (player plus team)"
    return record


def rebuild_from_record(game, record, root=ROOT):
    """Verify all raw hashes/metadata and reconstruct a record without network."""
    root = Path(root).resolve()

    def verified(source):
        path = (root / source["raw_path"]).resolve()
        if not path.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("Raw source escapes data/raw")
        body = path.read_bytes()
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if digest(body) != source["sha256"] or meta.get("status") != 200:
            raise ValueError("Raw source checksum/status mismatch")
        for key in ("sha256", "url", "retrieved_at_utc", "raw_path"):
            if meta.get(key) != source.get(key):
                raise ValueError("Raw source metadata differs from normalized provenance")
        return body, meta

    source = record["provenance"]
    body, meta = verified(source)
    if source["provider"] == "ESPN":
        return normalize_summary(game, json.loads(body), meta)
    if source["provider"] == "NBA":
        identity_body, identity_meta = verified(source["identity_mapping_source"])
        return normalize_nba_page(game, body, meta, json.loads(identity_body), identity_meta)
    raise ValueError("Unknown rich-stat source provider")


def collect(history_path, output, *, fetch_missing=False, workers=4, delay=0.35, retries=2, limit=None):
    history_path, output = Path(history_path), Path(output)
    history_body = history_path.read_bytes()
    games = [json.loads(line) for line in history_body.decode().splitlines()]
    if len({g["game_id"] for g in games}) != len(games):
        raise ValueError("Duplicate history game")
    if limit is not None:
        games = games[:limit]
    requested_games = len(games)
    excluded = [{"game_id": g["game_id"], "season": g["season"], "game_date_et": g["game_date_et"],
                 "home_abbr": g["home_abbr"], "away_abbr": g["away_abbr"],
                 "reason": "not_two_NBA_franchises: legacy ESPN season.type=2 history also contains All-Star exhibition games"}
                for g in games if any(ABBR.get(g[side + "_abbr"], g[side + "_abbr"]) not in NBA_FRANCHISES
                                      for side in ("home", "away"))]
    excluded_ids = {g["game_id"] for g in excluded}
    games = [g for g in games if g["game_id"] not in excluded_ids]
    output.mkdir(parents=True, exist_ok=True)
    started = iso(datetime.now(timezone.utc))

    def one(game):
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=" + game["espn_id"]
        for attempt in range(retries + 1):
            body, meta = Cache(offline=not fetch_missing, delay=delay).get(url, "summaries", ".json")
            if body is not None or not fetch_missing or meta.get("status") in (400, 403, 404, 410):
                break
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
        if body is None:
            return None, {"game_id": game["game_id"], "season": game["season"], "reason": "source_unavailable",
                          "source": meta}
        try:
            if game["game_id"] in NBA_REPAIR_IDS:
                nba_url = f'https://www.nba.com/game/{NBA_REPAIR_IDS[game["game_id"]]}/box-score'
                nba_body, nba_meta = Cache(offline=not fetch_missing, delay=delay).get(nba_url, "rich_stats_nba", ".html")
                if nba_body is None:
                    raise ValueError("Required NBA repair source unavailable")
                record = normalize_nba_page(game, nba_body, nba_meta, json.loads(body), meta)
            else:
                record = normalize_summary(game, json.loads(body), meta)
            return record, None
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return None, {"game_id": game["game_id"], "season": game["season"], "reason": str(error),
                          "source": {k: meta.get(k) for k in ("url", "raw_path", "sha256", "status")}}

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, g): g for g in games}
        for count, future in enumerate(as_completed(futures), 1):
            record, failure = future.result()
            if record is not None:
                records.append(record)
            else:
                failures.append(failure)
            if count % 100 == 0 or count == len(games):
                progress = dict(processed=count, requested=len(games), complete=len(records), failures=len(failures))
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    records.sort(key=lambda r: (r["scheduled_start_utc"], r["game_id"]))
    failures.sort(key=lambda r: r["game_id"])
    write_jsonl(output / "games.jsonl", records)
    write_jsonl(output / "failures.jsonl", failures)
    write_jsonl(output / "excluded_non_franchise.jsonl", excluded)
    incomplete_players = [{"game_id": r["game_id"], "teams": {s: t["player_issues"] for s, t in r["teams"].items()
                           if not t["players_complete"]}} for r in records if not r["players_complete"]]
    write_jsonl(output / "player_issues.jsonl", incomplete_players)
    repairs = [{"game_id": r["game_id"], "reason": r["provenance"]["rejected_original_reason"],
                "replacement_source": {k: v for k, v in r["provenance"].items() if k != "identity_mapping_source"},
                "rejected_numeric_source_retained_for_player_ids": r["provenance"]["identity_mapping_source"]}
               for r in records if r["provenance"]["provider"] == "NBA"]
    write_jsonl(output / "source_repairs.jsonl", repairs)
    provider_counts = dict(Counter(r["provenance"]["provider"] for r in records))
    manifest = {"schema_version": "rich-stats-v1", "provider": "mixed" if len(provider_counts) > 1 else next(iter(provider_counts), "none"), "started_at_utc": started,
                "finished_at_utc": iso(datetime.now(timezone.utc)), "requested_games": requested_games,
                "eligible_history_games": len(games), "excluded_non_franchise_games": len(excluded),
                "complete_team_games": len(records), "complete_player_games": len(records) - len(incomplete_players),
                "failed_games": len(failures), "season_counts": dict(Counter(r["season"] for r in records)),
                "provider_counts": provider_counts,
                "history_path": str(history_path), "history_sha256": digest(history_body),
                "normalizer_sha256": digest(Path(__file__).read_bytes()),
                "artifact_sha256": {name: digest((output / name).read_bytes()) for name in
                                    ("games.jsonl", "failures.jsonl", "excluded_non_franchise.jsonl", "player_issues.jsonl", "source_repairs.jsonl")},
                "normalization": "Numeric counters and historical participant IDs/minutes only. No raw ESPN, narratives, profiles, odds or current availability in outputs.",
                "limitation": "NBA Stats bulk access probe timed out. Sources are ESPN plus explicitly audited NBA.com historical box-score replacements where needed. Ratings/possessions derived downstream are estimates, not NBA official advanced ratings. Retrospective stat correction timestamps are unknown."}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return records, failures, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "data/processed/seasonal/game_history.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.delay < 0.35 or not 0 <= args.retries <= 3:
        parser.error("Use1–4workers, at least0.35s delay per worker, and0–3retries")
    collect(args.history, args.output, fetch_missing=args.fetch_missing, workers=args.workers,
            delay=args.delay, retries=args.retries, limit=args.limit)


if __name__ == "__main__":
    main()
