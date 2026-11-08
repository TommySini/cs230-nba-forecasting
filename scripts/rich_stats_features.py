"""Prior-date team and player box-score features for the richer boosting baseline.

Only allowlisted numeric, normalized box scores enter calculations. Canonical
scoreboard history determines each team's actual windows, so an absent box score
cannot silently turn the last five games into the last five *available* games.
The target game's box score, participating players and later games are never read.

Possessions use the explicitly approximate box-score formula
0.5 * sum_over_both_teams(FGA - OREB + TOV + 0.44 * FTA). These are not official
NBA offensive ratings or possession counts. Current retrieval timestamps do not
prove historical byte availability; prior ET dates provide the same conservative
result-availability assumption as the original baseline. Explicit result-available
timestamps, when supplied by a collector, additionally must precede the cutoff.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math


WINDOWS = (("season", None), ("last5", 5), ("last10", 10))
SIDES = ("home", "away", "difference")
TEAM_METRICS = (
    "estimated_offensive_rating", "estimated_defensive_rating", "estimated_net_rating",
    "estimated_pace", "effective_fg_pct", "three_point_attempt_rate", "three_point_pct",
    "free_throw_attempt_rate", "free_throw_pct", "turnovers_per100_estimated_possessions",
    "offensive_rebound_pct", "defensive_rebound_pct", "opponent_effective_fg_pct",
    "opponent_three_point_attempt_rate", "opponent_free_throw_attempt_rate",
    "opponent_turnovers_per100_estimated_possessions",
)
PLAYER_STAT_METRICS = (
    "max_player_minutes_per_game", "top5_minutes_share", "minutes_concentration",
    "top3_offensive_involvement_share", "rotation_players_10plus_minutes_per_game",
    "consecutive_game_minutes_overlap",
)
PLAYER_COVERAGE_METRICS = ("player_boxscore_coverage", "player_boxscore_missing")
PLAYER_METRICS = PLAYER_STAT_METRICS + PLAYER_COVERAGE_METRICS
TEAM_FEATURE_NAMES = tuple(f"{side}_{window}_{name}" for side in SIDES
                           for window, _ in WINDOWS for name in TEAM_METRICS)
PLAYER_FEATURE_NAMES = tuple(f"{side}_{window}_{name}" for side in SIDES
                             for window, _ in WINDOWS for name in PLAYER_METRICS)
RICH_FEATURE_NAMES = TEAM_FEATURE_NAMES + PLAYER_FEATURE_NAMES
TEAM_REQUIRED_FIELDS = ("minutes", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                        "oreb", "dreb", "tov", "pts")
PLAYER_REQUIRED_FIELDS = ("minutes", "fga", "fta", "tov")
FEATURE_DEFINITIONS = {
    "estimated_offensive_rating": "100 * summed team points / summed estimated possessions",
    "estimated_defensive_rating": "100 * summed opponent points / summed estimated possessions",
    "estimated_net_rating": "Estimated offensive rating minus estimated defensive rating",
    "estimated_pace": "48 * summed estimated possessions / summed game-duration minutes",
    "effective_fg_pct": "(sum FGM + 0.5 * sum 3PM) / sum FGA",
    "three_point_attempt_rate": "sum 3PA / sum FGA",
    "three_point_pct": "sum 3PM / sum 3PA",
    "free_throw_attempt_rate": "sum FTA / sum FGA",
    "free_throw_pct": "sum FTM / sum FTA",
    "turnovers_per100_estimated_possessions": "100 * sum TOV / summed estimated possessions",
    "offensive_rebound_pct": "sum OREB / (sum OREB + sum opponent DREB)",
    "defensive_rebound_pct": "sum DREB / (sum DREB + sum opponent OREB)",
    "opponent_effective_fg_pct": "Opponent (sum FGM + 0.5 * sum 3PM) / sum FGA",
    "opponent_three_point_attempt_rate": "Opponent sum 3PA / sum FGA",
    "opponent_free_throw_attempt_rate": "Opponent sum FTA / sum FGA",
    "opponent_turnovers_per100_estimated_possessions": "100 * sum opponent TOV / summed estimated possessions",
    "max_player_minutes_per_game": "Largest individual summed minutes divided by all team games in the window; absent appearances count as zero",
    "top5_minutes_share": "Share of all player minutes allocated to the five players with most minutes over the window",
    "minutes_concentration": "Sum of squared individual shares of all player minutes (Herfindahl index)",
    "top3_offensive_involvement_share": "Top three players' share of summed FGA + 0.44*FTA + TOV; an offensive-involvement proxy, not official usage percentage",
    "rotation_players_10plus_minutes_per_game": "Average number of players with at least ten minutes per game",
    "consecutive_game_minutes_overlap": "Mean sum of minimum individual minute shares across each consecutive pair of games in the window",
    "player_boxscore_coverage": "Fraction of actual team games in the window with complete, validated player minutes and counters",
    "player_boxscore_missing": "1 if any actual team game in the window has an incomplete player box score; its six player statistics are missing, not zero",
}
AVAILABILITY_LIMITATION = (
    "Historical box scores were retrieved retrospectively. Earlier ET game dates are used, "
    "with explicit stat_available_at_utc/completed_at_utc enforced when present; historical "
    "correction times and exact result availability are not independently archived."
)


class RichFeatureError(ValueError):
    """A selected historical source cannot support the requested feature group."""


def _stamp(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise RichFeatureError("Timestamp must include a timezone")
    return stamp.astimezone(timezone.utc)


def _number(value, label):
    if isinstance(value, bool):
        raise RichFeatureError(f"Boolean numeric value: {label}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise RichFeatureError(f"Missing/non-numeric value: {label}") from None
    if not math.isfinite(result) or result < 0:
        raise RichFeatureError(f"Non-finite/negative value: {label}")
    return result


def _ratio(numerator, denominator, label):
    if denominator <= 0:
        raise RichFeatureError(f"Zero denominator: {label}")
    return numerator / denominator


def _hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _team_stats(team, label):
    if team.get("complete") is False:
        raise RichFeatureError(f"Explicitly incomplete team box score: {label}")
    stats = {name: _number(team.get(name), f"{label}.{name}") for name in TEAM_REQUIRED_FIELDS}
    if any(abs(value - round(value)) > 1e-6 for name, value in stats.items() if name != "minutes"):
        raise RichFeatureError(f"Non-integral box-score counter: {label}")
    for made, attempts in (("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")):
        if stats[made] > stats[attempts]:
            raise RichFeatureError(f"Makes exceed attempts: {label}.{made}")
    if stats["fg3m"] > stats["fgm"] or stats["fg3a"] > stats["fga"]:
        raise RichFeatureError(f"Three-pointers exceed field goals: {label}")
    if abs(stats["pts"] - (2 * stats["fgm"] + stats["fg3m"] + stats["ftm"])) > 0.01:
        raise RichFeatureError(f"Scoring arithmetic mismatch: {label}")
    if "reb" in team and abs(_number(team["reb"], f"{label}.reb") - stats["oreb"] - stats["dreb"]) > .01:
        raise RichFeatureError(f"Rebound arithmetic mismatch: {label}")
    if not 220 <= stats["minutes"] <= 600:
        raise RichFeatureError(f"Implausible team player-minute total: {label}")
    return stats


def _players(team, stats, label):
    if team.get("players_complete") is False:
        raise RichFeatureError(f"Explicitly incomplete player box score: {label}")
    players = team.get("players")
    if not isinstance(players, list) or not players:
        raise RichFeatureError(f"Missing player box score: {label}")
    normalized, ids = [], set()
    for row in players:
        raw_player_id = row.get("player_id")
        player_id = "" if raw_player_id is None else str(raw_player_id)
        if not player_id or player_id in ids:
            raise RichFeatureError(f"Missing/duplicate player ID: {label}")
        ids.add(player_id)
        # DNPs can have zero minutes and absent counters. A positive-minute row
        # must have explicit counters; never fill an unknown performance with 0.
        minutes = _number(row.get("minutes"), f"{label}.{player_id}.minutes")
        if minutes > stats["minutes"] / 5 + 2:
            raise RichFeatureError(f"Player minutes exceed game duration: {label}.{player_id}")
        numbers = {"minutes": minutes}
        for field in PLAYER_REQUIRED_FIELDS[1:]:
            value = row.get(field)
            if value is None and minutes == 0:
                value = 0
            numbers[field] = _number(value, f"{label}.{player_id}.{field}")
            if abs(numbers[field] - round(numbers[field])) > 1e-6:
                raise RichFeatureError(f"Non-integral player counter: {label}.{player_id}.{field}")
        normalized.append({"player_id": player_id, **numbers})
    if abs(sum(p["minutes"] for p in normalized) - stats["minutes"]) > 8:
        raise RichFeatureError(f"Incomplete player minutes: {label}")
    for field in ("fga", "fta"):
        if abs(sum(p[field] for p in normalized) - stats[field]) > .01:
            raise RichFeatureError(f"Player/team {field} mismatch: {label}")
    # Team turnovers may include violations that are not charged to a player.
    player_tov = sum(p["tov"] for p in normalized)
    if player_tov > stats["tov"] + .01:
        raise RichFeatureError(f"Player/team turnover mismatch: {label}")
    return sorted(normalized, key=lambda p: p["player_id"])


def _aggregate_team(games):
    own = {field: sum(g["own"][field] for g in games) for field in TEAM_REQUIRED_FIELDS}
    opp = {field: sum(g["opponent"][field] for g in games) for field in TEAM_REQUIRED_FIELDS}
    possessions = sum(g["estimated_possessions"] for g in games)
    duration = sum(g["game_minutes"] for g in games)
    offensive = 100 * _ratio(own["pts"], possessions, "estimated_offensive_rating")
    defensive = 100 * _ratio(opp["pts"], possessions, "estimated_defensive_rating")
    return {
        "estimated_offensive_rating": offensive,
        "estimated_defensive_rating": defensive,
        "estimated_net_rating": offensive - defensive,
        "estimated_pace": 48 * _ratio(possessions, duration, "estimated_pace"),
        "effective_fg_pct": _ratio(own["fgm"] + .5 * own["fg3m"], own["fga"], "effective_fg_pct"),
        "three_point_attempt_rate": _ratio(own["fg3a"], own["fga"], "three_point_attempt_rate"),
        "three_point_pct": _ratio(own["fg3m"], own["fg3a"], "three_point_pct"),
        "free_throw_attempt_rate": _ratio(own["fta"], own["fga"], "free_throw_attempt_rate"),
        "free_throw_pct": _ratio(own["ftm"], own["fta"], "free_throw_pct"),
        "turnovers_per100_estimated_possessions": 100 * _ratio(own["tov"], possessions, "turnovers_per100_estimated_possessions"),
        "offensive_rebound_pct": _ratio(own["oreb"], own["oreb"] + opp["dreb"], "offensive_rebound_pct"),
        "defensive_rebound_pct": _ratio(own["dreb"], own["dreb"] + opp["oreb"], "defensive_rebound_pct"),
        "opponent_effective_fg_pct": _ratio(opp["fgm"] + .5 * opp["fg3m"], opp["fga"], "opponent_effective_fg_pct"),
        "opponent_three_point_attempt_rate": _ratio(opp["fg3a"], opp["fga"], "opponent_three_point_attempt_rate"),
        "opponent_free_throw_attempt_rate": _ratio(opp["fta"], opp["fga"], "opponent_free_throw_attempt_rate"),
        "opponent_turnovers_per100_estimated_possessions": 100 * _ratio(opp["tov"], possessions, "opponent_turnovers_per100_estimated_possessions"),
    }


def _aggregate_players(games):
    valid = sum(game["players"] is not None for game in games)
    coverage = {"player_boxscore_coverage": valid / len(games),
                "player_boxscore_missing": float(valid < len(games))}
    if valid < len(games):
        return {**{name: float("nan") for name in PLAYER_STAT_METRICS}, **coverage}
    minutes, volume = defaultdict(float), defaultdict(float)
    shares, rotation = [], []
    for game in games:
        players = game["players"]
        total = sum(p["minutes"] for p in players)
        if total <= 0:
            raise RichFeatureError("Zero player-minute total")
        shares.append({p["player_id"]: p["minutes"] / total for p in players})
        rotation.append(sum(p["minutes"] >= 10 for p in players))
        for player in players:
            player_id = player["player_id"]
            minutes[player_id] += player["minutes"]
            volume[player_id] += player["fga"] + .44 * player["fta"] + player["tov"]
    total_minutes, total_volume = sum(minutes.values()), sum(volume.values())
    overlap = [sum(min(before.get(p, 0), after.get(p, 0)) for p in before.keys() | after.keys())
               for before, after in zip(shares, shares[1:])]
    if not overlap:
        raise RichFeatureError("Player continuity needs at least two prior games")
    return {
        "max_player_minutes_per_game": max(minutes.values()) / len(games),
        "top5_minutes_share": sum(sorted(minutes.values(), reverse=True)[:5]) / total_minutes,
        "minutes_concentration": sum((value / total_minutes) ** 2 for value in minutes.values()),
        "top3_offensive_involvement_share": _ratio(sum(sorted(volume.values(), reverse=True)[:3]), total_volume,
                                            "top3_offensive_involvement_share"),
        "rotation_players_10plus_minutes_per_game": sum(rotation) / len(games),
        "consecutive_game_minutes_overlap": sum(overlap) / len(overlap),
        **coverage,
    }


class RichStatsIndex:
    """Build rich features against complete canonical prior-game membership.

    `history` is the existing allowlisted scoreboard history. `box_records` use
    canonical `game_id`, `season`, `game_date_et`, and `teams.home/away` with the
    numeric fields above, a `team_abbr`, and optional normalized `players`.
    Metadata in `provenance` is preserved only in lineage, never as a feature.
    """

    def __init__(self, history, box_records, *, min_previous_games=5, cutoff_minutes=60):
        if min_previous_games < 2 or cutoff_minutes <= 0:
            raise ValueError("At least two prior games and a positive cutoff are required")
        self.min_previous_games, self.cutoff_minutes = min_previous_games, cutoff_minutes
        self.history, self.records = {}, {}
        self.by_team = defaultdict(list)
        for game in history:
            if game["game_id"] in self.history:
                raise ValueError(f"Duplicate canonical game: {game['game_id']}")
            self.history[game["game_id"]] = game
            for side in ("home", "away"):
                self.by_team[(game["season"], game[f"{side}_abbr"])].append(game)
        for games in self.by_team.values():
            games.sort(key=lambda g: (g["game_date_et"], g["scheduled_start_utc"], g["game_id"]))
        for record in box_records:
            game_id = record["game_id"]
            if game_id in self.records:
                raise ValueError(f"Duplicate normalized box score: {game_id}")
            self.records[game_id] = record

    def _prior_record(self, canonical, team, cutoff, include_players):
        game_id = canonical["game_id"]
        if game_id not in self.records:
            raise RichFeatureError(f"Missing historical box score: {game_id}")
        record = self.records[game_id]
        for field in ("season", "game_date_et"):
            if record.get(field) != canonical[field]:
                raise RichFeatureError(f"Box-score identity mismatch: {game_id}.{field}")
        if record.get("complete") is False:
            raise RichFeatureError(f"Explicitly incomplete game box score: {game_id}")
        for field in ("stat_available_at_utc", "completed_at_utc"):
            if record.get(field) is not None and _stamp(record[field]) > cutoff:
                raise RichFeatureError(f"Historical statistics unavailable at cutoff: {game_id}.{field}")
        teams = record.get("teams", {})
        numbers = {}
        for side in ("home", "away"):
            block = teams.get(side, {})
            if block.get("team_abbr") != canonical[f"{side}_abbr"]:
                raise RichFeatureError(f"Box-score team mismatch: {game_id}.{side}")
            numbers[side] = _team_stats(block, f"{game_id}.{side}")
            if numbers[side]["pts"] != canonical[f"{side}_score"]:
                raise RichFeatureError(f"Box-score/canonical score mismatch: {game_id}.{side}")
        if abs(numbers["home"]["minutes"] - numbers["away"]["minutes"]) > 8:
            raise RichFeatureError(f"Opposing team minutes disagree: {game_id}")
        supplied_durations = [value for value in (record.get("game_minutes"),
                              teams["home"].get("game_minutes"), teams["away"].get("game_minutes"))
                              if value is not None]
        supplied_durations = [_number(value, f"{game_id}.game_minutes") for value in supplied_durations]
        if len(set(supplied_durations)) > 1:
            raise RichFeatureError(f"Inconsistent game durations: {game_id}")
        duration = supplied_durations[0] if supplied_durations else None
        if duration is None:
            duration = (numbers["home"]["minutes"] + numbers["away"]["minutes"]) / 10
            duration_basis = "mean_team_player_minutes_divided_by_five"
        else:
            duration = _number(duration, f"{game_id}.game_minutes")
            duration_basis = "normalized_game_minutes"
            if any(abs(value["minutes"] - 5 * duration) > 8 for value in numbers.values()):
                raise RichFeatureError(f"Team minutes disagree with game duration: {game_id}")
        if not 44 <= duration <= 120:
            raise RichFeatureError(f"Implausible game duration: {game_id}")
        possessions = .5 * sum(v["fga"] - v["oreb"] + v["tov"] + .44 * v["fta"]
                               for v in numbers.values())
        if possessions <= 0:
            raise RichFeatureError(f"Nonpositive estimated possessions: {game_id}")
        own_side = "home" if canonical["home_abbr"] == team else "away"
        other_side = "away" if own_side == "home" else "home"
        selected = {"game_id": game_id, "game_date_et": canonical["game_date_et"],
                    "own": numbers[own_side], "opponent": numbers[other_side],
                    "estimated_possessions": possessions, "game_minutes": duration}
        player_issue = None
        if include_players:
            try:
                selected["players"] = _players(teams[own_side], numbers[own_side], f"{game_id}.{own_side}")
            except (RichFeatureError, KeyError, TypeError) as exc:
                # A missing player report must not erase known team statistics.
                # It makes player summaries for each affected actual window
                # unavailable, with explicit indicators for train-only imputation.
                selected["players"] = None
                player_issue = str(exc)
        lineage = {"game_id": game_id, "game_date_et": canonical["game_date_et"],
                   "team_abbr": team, "duration_basis": duration_basis,
                   "used_numeric_sha256": _hash(selected), "provenance": record.get("provenance", {})}
        if player_issue:
            lineage["player_issue"] = player_issue
        for field in ("stat_available_at_utc", "completed_at_utc"):
            if record.get(field) is not None:
                lineage[field] = record[field]
        return selected, lineage

    def for_game(self, game, *, include_players=True):
        """Return features, past-source lineage, coverage and explicit exclusions.

        Last-ten summaries use all available *actual* prior games for a team with
        five to nine prior games, preserving the original five-game warmup.
        Eligibility requires every canonical prior game to have a valid team
        source. Missing player reports leave affected window statistics as NaN
        with explicit indicators; a downstream imputer must be fitted on TRAIN
        only (TRAIN+DEV for the predeclared final refit).
        """
        cutoff = (_stamp(game["cutoff_utc"]) if game.get("cutoff_utc")
                  else _stamp(game["scheduled_start_utc"]) - timedelta(minutes=self.cutoff_minutes))
        result = {"game_id": game["game_id"], "eligible": False, "features": {}, "lineage": {},
                  "coverage": {}, "reasons": [], "player_issues": [], "cutoff_utc": cutoff.isoformat(),
                  "availability_limitation": AVAILABILITY_LIMITATION}
        side_values = {}
        for side in ("home", "away"):
            team = game[f"{side}_abbr"]
            previous = [g for g in self.by_team[(game["season"], team)]
                        if g["game_date_et"] < game["game_date_et"]]
            result["coverage"][side] = {"expected_previous_games": len(previous),
                                         "valid_previous_games": 0,
                                         "window_games": {name: len(previous) if limit is None else min(limit, len(previous))
                                                          for name, limit in WINDOWS}}
            result["lineage"][side] = []
            if len(previous) < self.min_previous_games:
                result["reasons"].append(f"{side}: fewer than {self.min_previous_games} previous games")
                continue
            selected = []
            for prior in previous:
                try:
                    record, lineage = self._prior_record(prior, team, cutoff, include_players)
                    selected.append(record)
                    result["lineage"][side].append(lineage)
                    if lineage.get("player_issue"):
                        result["player_issues"].append({"side": side, "game_id": prior["game_id"],
                                                        "reason": lineage["player_issue"]})
                except (RichFeatureError, KeyError, TypeError) as exc:
                    result["reasons"].append(f"{side}: {prior['game_id']}: {exc}")
            result["coverage"][side]["valid_previous_games"] = len(selected)
            if include_players:
                result["coverage"][side]["valid_player_games"] = sum(g["players"] is not None for g in selected)
            if len(selected) != len(previous):
                continue
            values = {}
            try:
                for name, limit in WINDOWS:
                    window = selected if limit is None else selected[-limit:]
                    metrics = _aggregate_team(window)
                    if include_players:
                        metrics.update(_aggregate_players(window))
                        result["coverage"][side].setdefault("window_player_games", {})[name] = sum(g["players"] is not None for g in window)
                    values.update({f"{name}_{metric}": value for metric, value in metrics.items()})
            except RichFeatureError as exc:
                result["reasons"].append(f"{side}: {exc}")
                continue
            side_values[side] = values
        if result["reasons"] or set(side_values) != {"home", "away"}:
            return result
        names = TEAM_FEATURE_NAMES + (PLAYER_FEATURE_NAMES if include_players else ())
        values = {f"{side}_{name}": value for side, items in side_values.items() for name, value in items.items()}
        values.update({f"difference_{name}": side_values["home"][name] - side_values["away"][name]
                       for name in side_values["home"]})
        result["features"] = {name: values[name] for name in names}
        if any(not math.isfinite(result["features"][name]) for name in TEAM_FEATURE_NAMES):
            raise RichFeatureError("Non-finite assembled features")
        if any(math.isinf(value) for value in result["features"].values()):
            raise RichFeatureError("Infinite assembled player features")
        result["eligible"] = True
        return result


def augment_inputs(inputs, history, box_records, *, include_players=True, min_previous_games=5,
                   cutoff_minutes=60):
    """Preserve original inputs/features and return augmented rows plus exclusions.

    This function reads no label files and does not fit or impute anything.
    Excluded rows retain reasons and coverage, so an evaluator can enforce a
    matched cohort and report the exact coverage loss.
    """
    index = RichStatsIndex(history, box_records, min_previous_games=min_previous_games,
                           cutoff_minutes=cutoff_minutes)
    augmented, excluded = [], []
    for row in inputs:
        result = index.for_game(row, include_players=include_players)
        if not result["eligible"]:
            excluded.append({key: result[key] for key in ("game_id", "coverage", "reasons")})
            continue
        augmented.append({**row, "features": {**row.get("features", {}), **result["features"]},
                          "rich_stats_lineage": result["lineage"], "rich_stats_coverage": result["coverage"],
                          "rich_stats_player_issues": result["player_issues"],
                          "rich_stats_availability_limitation": result["availability_limitation"]})
    return augmented, excluded
