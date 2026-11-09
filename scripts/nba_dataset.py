"""Shared, typed NBA tables and a lossless migration from the frozen local sources.

The staging command never edits its source dataset. Publishing and removing old
copies are separate operations, performed only after the round-trip checks pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "test")
SCHEMA_VERSION = "nba-shared-parquet-1"
SOURCE_REVISION = "7681631"
RAW_DIRECTORIES = {
    "scoreboards": "espn/scoreboards", "summaries": "espn/summaries",
    "rich_stats_nba": "nba/box_scores", "rich_stats_repairs": "nba/repairs",
    "rich_stats_probe": "nba/probes", "archive_cdx": "notes/archive_cdx",
    "archive_versions": "notes/archive_versions", "html_notes_api": "notes/html_api",
    "team_notes": "notes/team_notes", "polymarket_markets": "polymarket/markets",
    "polymarket_prices": "polymarket/prices", "polymarket_trades": "polymarket/trades",
}
UTC = pa.timestamp("us", tz="UTC")
RAW_FILES = {"data/research/archive_coverage/data/cdx_gamenotes_all.json":
             "data/raw/notes/archive_cdx/cdx_gamenotes_all.json"}
PREDICTION_LOCK = ".predictions.lock"
PREDICTION_JOURNAL = ".prediction_transaction.json"
PREDICTION_TEMPORARIES = (".predictions.next.parquet", ".predictions.next.parquet.tmp",
                         ".predictions.previous.parquet", ".manifest.next.json",
                         ".manifest.next.json.tmp", ".manifest.previous.json",
                         PREDICTION_JOURNAL + ".tmp")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def logical_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _prediction_lock(directory):
    """Serialize the only mutable shared table and its manifest for readers/writers."""
    with (Path(directory) / PREDICTION_LOCK).open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _clean_prediction_transaction(directory):
    for name in PREDICTION_TEMPORARIES:
        (directory / name).unlink(missing_ok=True)
    (directory / PREDICTION_JOURNAL).unlink(missing_ok=True)


def _recover_predictions(directory):
    """Finish or roll back an interrupted append while holding the writer lock."""
    directory = Path(directory)
    journal_path = directory / PREDICTION_JOURNAL
    if not journal_path.exists():
        # An interruption before the journal was published never touched live files.
        _clean_prediction_transaction(directory)
        return
    journal = json.loads(journal_path.read_text())
    live_table, live_manifest = directory / "predictions.parquet", directory / "manifest.json"
    candidate_table, candidate_manifest = directory / ".predictions.next.parquet", directory / ".manifest.next.json"
    previous_table, previous_manifest = directory / ".predictions.previous.parquet", directory / ".manifest.previous.json"
    if journal.get("schema_version") != 1:
        raise ValueError("Unknown prediction transaction journal")
    matches = lambda path, expected: path.is_file() and sha256(path) == expected
    for path, old, new in ((live_table, journal["old_table_sha256"], journal["new_table_sha256"]),
                           (live_manifest, journal["old_manifest_sha256"], journal["new_manifest_sha256"])):
        if path.exists() and not (matches(path, old) or matches(path, new)):
            raise ValueError("Prediction transaction encountered an unrecorded live-file change")
    new_table_available = matches(live_table, journal["new_table_sha256"]) or matches(candidate_table, journal["new_table_sha256"])
    new_manifest_available = matches(live_manifest, journal["new_manifest_sha256"]) or matches(candidate_manifest, journal["new_manifest_sha256"])
    if new_table_available and new_manifest_available:
        if not matches(live_table, journal["new_table_sha256"]):
            candidate_table.replace(live_table)
        if not matches(live_manifest, journal["new_manifest_sha256"]):
            candidate_manifest.replace(live_manifest)
    elif matches(previous_table, journal["old_table_sha256"]) and matches(previous_manifest, journal["old_manifest_sha256"]):
        previous_table.replace(live_table)
        previous_manifest.replace(live_manifest)
    else:
        raise ValueError("Prediction transaction cannot recover verified old or new files")
    _clean_prediction_transaction(directory)


def _commit_predictions(directory, manifest, rows):
    """Journal before replacing either live file; readers recover after SIGALRM/exit."""
    directory = Path(directory)
    live_table, live_manifest = directory / "predictions.parquet", directory / "manifest.json"
    candidate_table, candidate_manifest = directory / ".predictions.next.parquet", directory / ".manifest.next.json"
    previous_table, previous_manifest = directory / ".predictions.previous.parquet", directory / ".manifest.previous.json"
    shutil.copyfile(live_table, previous_table)
    shutil.copyfile(live_manifest, previous_manifest)
    updated = json.loads(json.dumps(manifest))
    table = write_table(candidate_table, rows)
    table["file"] = "predictions.parquet"
    updated["tables"]["predictions"] = table
    _atomic_json(candidate_manifest, updated)
    journal = {"schema_version": 1, "old_table_sha256": sha256(previous_table),
               "new_table_sha256": table["sha256"], "old_manifest_sha256": sha256(previous_manifest),
               "new_manifest_sha256": sha256(candidate_manifest)}
    _atomic_json(directory / PREDICTION_JOURNAL, journal)
    candidate_table.replace(live_table)
    candidate_manifest.replace(live_manifest)
    _clean_prediction_transaction(directory)
    return updated


def remap_raw_path(value):
    """Translate recorded old paths without modifying the original cache sidecar."""
    if isinstance(value, str) and value in RAW_FILES:
        return RAW_FILES[value]
    if not isinstance(value, str) or not value.startswith("data/raw/"):
        return value
    parts = value.split("/")
    if ".." in parts:
        raise ValueError("Raw source path contains traversal")
    if len(parts) >= 3 and parts[2] in RAW_DIRECTORIES:
        return "/".join(("data/raw", RAW_DIRECTORIES[parts[2]], *parts[3:]))
    return value


def remap_paths(value):
    if isinstance(value, dict):
        return {key: remap_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_paths(item) for item in value]
    return remap_raw_path(value)


def resolve_source_path(root, value):
    """Read old or regrouped raw paths safely during the migration window."""
    root, value = Path(root), str(value)
    if (not value.startswith("data/raw/") and value not in RAW_FILES) or ".." in Path(value).parts:
        raise ValueError("Expected a repository-relative raw source path")
    candidates = [root / remap_raw_path(value), root / value]
    for path in candidates:
        if path.exists():
            allowed = root / ("data" if value in RAW_FILES and path == root / value else "data/raw")
            if not path.resolve().is_relative_to(allowed.resolve()):
                raise ValueError("Raw source path escapes data/raw")
            return path
    # New-format records are usable before the raw directories have moved.
    for old, new in RAW_DIRECTORIES.items():
        prefix = "data/raw/" + new + "/"
        if value.startswith(prefix):
            path = root / ("data/raw/" + old + "/" + value[len(prefix):])
            if path.exists():
                if not path.resolve().is_relative_to((root / "data/raw").resolve()):
                    raise ValueError("Raw source path escapes data/raw")
                return path
    for old, new in RAW_FILES.items():
        if value == new and (root / old).is_file():
            path = root / old
            if not path.resolve().is_relative_to((root / "data").resolve()):
                raise ValueError("Raw source path escapes data")
            return path
    raise FileNotFoundError(value)


def _strings(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strings(item) for item in value]
    return value


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return _strings(pq.read_table(path).to_pylist())
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_table(path, rows, schema=None):
    """Write a typed table atomically; all top-level fields participate in inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if schema is None:
        names = list(dict.fromkeys(key for row in rows for key in row))
        schema = pa.schema([(key, pa.array([row.get(key) for row in rows]).type) for key in names])
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", version="2.6")
    temporary.replace(path)
    return {"file": path.name, "sha256": sha256(path), "rows": table.num_rows,
            "schema": str(table.schema)}


def _pack(value):
    """Preserve absent versus present-null fields in heterogeneous audit structs."""
    if isinstance(value, dict):
        return {"__fields__": list(value), **{k: _pack(v) for k, v in value.items()}}
    if isinstance(value, list):
        return [_pack(v) for v in value]
    return value


def _unpack(value):
    if isinstance(value, dict):
        keys = value.get("__fields__", value.keys())
        return {key: _unpack(value[key]) for key in keys}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


def _stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def _unique(rows, key, name):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {name} key: {value}")
        result[value] = row
    return result


def _verify_frozen_sources(root, pins):
    """Data bytes must still match; edited source code is checked at its frozen revision."""
    for relative, expected in pins.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Frozen source paths must be repository-relative")
        if relative.startswith("data/"):
            actual = sha256(_original_path(root, relative))
        else:
            result = subprocess.run(["git", "show", SOURCE_REVISION + ":" + relative], cwd=root,
                                    capture_output=True, check=False)
            if result.returncode:
                raise ValueError(f"Cannot recover frozen source at {SOURCE_REVISION}: {relative}")
            actual = hashlib.sha256(result.stdout).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source checksum mismatch: {relative}")


def write_box_tables(directory, records):
    """Normalize completed-game records without placing postgame metadata in games."""
    directory = Path(directory)
    games, teams, players, sources = [], [], [], {}
    for box_index, box in enumerate(records):
        source = remap_paths(box["provenance"])
        source_id = logical_hash(source)
        sources[source_id] = {"source_record_id": source_id, **_pack(source)}
        games.append({"box_index": box_index, "source_record_id": source_id,
                      **_pack(remap_paths({k: v for k, v in box.items() if k not in ("teams", "provenance")}))})
        for side, team in box["teams"].items():
            teams.append({"game_id": box["game_id"], "side": side, "source_record_id": source_id,
                          **_pack({k: v for k, v in team.items() if k != "players"})})
            for index, player in enumerate(team["players"]):
                players.append({"game_id": box["game_id"], "side": side, "player_index": index, **_pack(player)})
    return {name: write_table(directory / (name + ".parquet"), rows)
            for name, rows in (("box_score_games", games), ("team_box_scores", teams),
                               ("player_box_scores", players), ("sources", sources.values()))}


class Dataset:
    def __init__(self, root=ROOT, processed_dir=None):
        self.root = Path(root)
        self.directory = Path(processed_dir) if processed_dir is not None else self.root / "data/processed"
        self.processed_dir = self.directory
        with _prediction_lock(self.directory):
            _recover_predictions(self.directory)
            self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unknown shared dataset schema")
        self._cache = {}

    def table(self, name):
        name = name.removesuffix(".parquet")
        if name == "predictions":
            with _prediction_lock(self.directory):
                _recover_predictions(self.directory)
                self._refresh_manifest()
                entry = self.manifest["tables"][name]
                path = self.directory / entry["file"]
                if sha256(path) != entry["sha256"]:
                    raise ValueError("Shared table checksum mismatch: predictions")
                return read_rows(path)
        if name not in self._cache:
            entry = self.manifest["tables"][name]
            path = self.directory / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Shared table checksum mismatch: {name}")
            self._cache[name] = read_rows(path)
        return self._cache[name]

    def _refresh_manifest(self):
        current = json.loads((self.directory / "manifest.json").read_text())
        if current.get("dataset_id") != self.manifest.get("dataset_id"):
            raise ValueError("Dataset identity changed while it was open")
        self.manifest = current

    def verify(self, label_splits=SPLITS):
        with _prediction_lock(self.directory):
            _recover_predictions(self.directory)
            self._refresh_manifest()
            return self._verify(label_splits)

    def _verify(self, label_splits):
        allowed_labels = set(label_splits)
        if not allowed_labels <= set(SPLITS):
            raise ValueError("Unknown label split")
        for name, entry in self.manifest["tables"].items():
            path = self.directory / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Shared table checksum mismatch: {name}")
            # Hash all files, but do not deserialize source boxes or forbidden labels.
            if pq.read_metadata(path).num_rows != entry["rows"]:
                raise ValueError(f"Row count changed: {name}")
        for relative, expected in self.manifest.get("artifacts", {}).items():
            if sha256(self.directory / relative) != expected:
                raise ValueError(f"Shared metadata checksum mismatch: {relative}")
        for relative, expected in self.manifest.get("external_artifacts", {}).items():
            if sha256(_destination(self.root, relative)) != expected:
                raise ValueError(f"Installed model/run artifact checksum mismatch: {relative}")
        games = _unique(self.table("games"), "game_id", "game")
        features = _unique(self.table("features"), "game_id", "feature")
        names = self.manifest["feature_names"]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names")
        actual = pq.read_schema(self.directory / "features.parquet").names
        if actual != ["game_id", *names]:
            raise ValueError("Feature column order differs from the manifest")
        if set(features) != {key for key, game in games.items() if game["feature_eligible"]}:
            raise ValueError("Feature cohort differs from game catalog")
        for row in self.table("games"):
            if row["split"] not in SPLITS:
                raise ValueError("Unassigned game split")
            if _stamp(row["scheduled_start_utc"]) - _stamp(row["cutoff_utc"]) != timedelta(minutes=self.manifest["cutoff_minutes"]):
                raise ValueError("Game cutoff differs from policy")
        for row in self.table("feature_lineage"):
            game, prior = games[row["game_id"]], games[row["prior_game_id"]]
            if game["season"] != prior["season"] or prior["game_date_et"] >= game["game_date_et"]:
                raise ValueError("Current/future game in feature lineage")
        label_ids = []
        for split in SPLITS:
            if split not in allowed_labels:
                continue
            rows = self.table("labels_" + split)
            _unique(rows, "game_id", "label")
            for row in rows:
                if row["home_win"] not in (0, 1) or games[row["game_id"]]["split"] != split:
                    raise ValueError("Invalid outcome or label partition")
            label_ids.extend(row["game_id"] for row in rows)
        expected_labels = {gid for gid, game in games.items() if game["split"] in allowed_labels}
        if set(label_ids) != expected_labels or len(label_ids) != len(expected_labels):
            raise ValueError("Labels must cover the catalog exactly once")
        expected_identity = logical_hash({name: entry["sha256"] for name, entry in self.manifest["tables"].items()
                                          if name != "predictions"})
        if "dataset_id" in self.manifest and self.manifest["dataset_id"] != expected_identity:
            raise ValueError("Dataset identity differs from its input table hashes")
        return {"games": len(games), "features": len(features), "columns": len(names),
                "splits": dict(Counter(row["split"] for row in games.values() if row["feature_eligible"]))}

    def inputs(self, splits=None):
        allowed = set(SPLITS if splits is None else splits)
        if not allowed <= set(SPLITS):
            raise ValueError("Unknown split")
        games = sorted((g for g in self.table("games") if g["feature_eligible"] and g["split"] in allowed),
                       key=lambda row: row["feature_index"])
        features = _unique(self.table("features"), "game_id", "feature")
        lineage = defaultdict(lambda: {"home": [], "away": []})
        missing = defaultdict(lambda: {"home": [], "away": []})
        for row in sorted(self.table("feature_lineage"), key=lambda r: (r["game_id"], r["side"], r["prior_index"])):
            lineage[row["game_id"]][row["side"]].append(row["prior_game_id"])
            if row["missing_box_score"]:
                missing[row["game_id"]][row["side"]].append(row["prior_game_id"])
        headers = ("game_id", "season", "scheduled_start_utc", "game_date_et", "home_abbr", "away_abbr", "split", "cutoff_utc")
        return [{**{key: game[key] for key in headers},
                 "features": {name: features[game["game_id"]][name] for name in self.manifest["feature_names"]},
                 "prior_game_ids": lineage[game["game_id"]],
                 "missing_box_score_game_ids": missing[game["game_id"]],
                 "box_score_coverage": _unpack(game["box_score_coverage"])} for game in games]

    def labels(self, splits, eligible_only=True):
        """Only open the explicitly requested outcome partitions."""
        requested = [split for split in SPLITS if split in set(splits)]
        if set(requested) != set(splits):
            raise ValueError("Unknown label split")
        games = {row["game_id"]: row for row in self.table("games")}
        rows = [row for split in requested for row in self.table("labels_" + split)
                if not eligible_only or games[row["game_id"]]["feature_eligible"]]
        order = "feature_index" if eligible_only else "schedule_index"
        rows.sort(key=lambda row: games[row["game_id"]][order])
        return [{"game_id": row["game_id"], "home_win": row["home_win"]} for row in rows]

    def history(self, seasons=None, include_non_franchise=False):
        games = [row for row in self.table("games") if (include_non_franchise or row["franchise_game"])
                 and (seasons is None or row["season"] in set(seasons))]
        splits = {game["split"] for game in games}
        labels = {row["game_id"]: row for split in splits for row in self.table("labels_" + split)}
        names = ("game_id", "espn_id", "season", "scheduled_start_utc", "game_date_et", "schedule_source",
                 "home_team", "home_abbr", "away_team", "away_abbr")
        return [{**{key: game[key] for key in names}, "home_score": labels[game["game_id"]]["home_score"],
                 "away_score": labels[game["game_id"]]["away_score"]}
                for game in sorted(games, key=lambda row: row["schedule_index"])]

    def excluded(self):
        return [_unpack(row["exclusion"]) for row in sorted(self.table("games"), key=lambda r: r["schedule_index"])
                if row["exclusion"] is not None]

    def box_scores(self):
        teams, players = defaultdict(dict), defaultdict(list)
        for row in sorted(self.table("player_box_scores"), key=lambda r: r["player_index"]):
            players[(row["game_id"], row["side"])].append(_unpack(row))
        for row in self.table("team_box_scores"):
            team = _unpack(row)
            team["players"] = players[(row["game_id"], row["side"])]
            teams[row["game_id"]][row["side"]] = team
        sources = {row["source_record_id"]: _unpack(row) for row in self.table("sources")}
        rows = []
        for game in sorted(self.table("box_score_games"), key=lambda g: g["box_index"]):
            metadata = _unpack(game)
            rows.append({**metadata, "teams": teams[game["game_id"]],
                         "provenance": sources[game["source_record_id"]]})
        return rows

    def predictions(self, model_id="gradient_boosting", stage=None):
        rows = [row for row in self.table("predictions") if row["model_id"] == model_id and (stage is None or row["stage"] == stage)]
        rows.sort(key=lambda row: row["prediction_index"])
        if model_id == "gradient_boosting":
            return [{**{key: row[key] for key in ("game_id", "season", "game_date_et", "split", "produced_by")},
                     "p_boost": row["p_home_win"]} for row in rows]
        return rows


def store_predictions(root, rows, model_id, stage="baseline", run_id=None):
    """Publish one model/stage cohort, permitting only exact retries of existing rows."""
    dataset = Dataset(root)
    with _prediction_lock(dataset.directory):
        _recover_predictions(dataset.directory)
        dataset._refresh_manifest()
        return _store_predictions_locked(dataset, rows, model_id, stage, run_id)


def _store_predictions_locked(dataset, rows, model_id, stage, run_id):
    entry = dataset.manifest["tables"]["predictions"]
    path = dataset.directory / entry["file"]
    if sha256(path) != entry["sha256"]:
        raise ValueError("Shared table checksum mismatch: predictions")
    existing = read_rows(path)
    games = {row["game_id"]: row for row in dataset.table("games")}
    by_key = {(row["model_id"], row["stage"], row["game_id"]): row for row in existing}
    incoming = []
    seen = set()
    for index, given in enumerate(rows):
        gid = given["game_id"]
        if gid not in games or gid in seen:
            raise ValueError("Prediction game IDs must be unique and present in the dataset")
        seen.add(gid)
        p = given["p_home_win"]
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Prediction must be a finite probability in [0, 1]")
        game = games[gid]
        for key in ("season", "game_date_et", "split"):
            if key in given and given[key] != game[key]:
                raise ValueError(f"Prediction {key} differs from its game")
        row = {"game_id": gid, "season": game["season"], "game_date_et": game["game_date_et"],
               "split": game["split"], "model_id": model_id, "stage": stage,
               "run_id": run_id or model_id, "prediction_index": index,
               "produced_by": given.get("produced_by"), "p_home_win": float(p)}
        key = (model_id, stage, gid)
        if key in by_key:
            if by_key[key] != row:
                raise ValueError(f"Refusing to overwrite saved prediction: {key}")
        else:
            incoming.append(row)
    if incoming:
        dataset.manifest = _commit_predictions(dataset.directory, dataset.manifest, existing + incoming)
    return {"added": len(incoming), "dataset_id": dataset.manifest["dataset_id"]}

def _feature_dictionary(names):
    from box_score_features import FEATURE_DEFINITIONS
    basics = {
        "previous_games": "Number of earlier same-season games on earlier ET dates",
        "season_win_rate": "Wins divided by earlier same-season games",
        "last_5_win_rate": "Wins divided by the last five earlier same-season games",
        "last_5_point_differential": "Mean team-minus-opponent points in the last five earlier games",
        "days_since_previous_game": "ET calendar days since the previous game",
        "back_to_back": "1 when the previous game was on the preceding ET date",
    }
    rows = []
    for index, name in enumerate(names):
        side, remainder = name.split("_", 1)
        window = "season"
        metric = remainder
        if remainder not in basics:
            window, metric = remainder.split("_", 1)
        elif remainder.startswith("last_5_"):
            window = "last5"
        elif remainder in ("days_since_previous_game", "back_to_back"):
            window = "previous_game"
        definition = basics.get(remainder, FEATURE_DEFINITIONS.get(metric, metric))
        if side == "difference":
            definition = "Home minus away: " + definition
        unit = "ratio"
        if metric in ("previous_games", "rotation_players_10plus_minutes_per_game"):
            unit = "games" if metric == "previous_games" else "players per game"
        elif metric == "days_since_previous_game":
            unit = "ET calendar days"
        elif metric == "back_to_back" or metric.endswith("_missing"):
            unit = "binary flag"
        elif metric == "last_5_point_differential":
            unit = "points per game"
        elif metric.endswith("_rating"):
            unit = "points per 100 estimated possessions"
        elif metric == "estimated_pace":
            unit = "estimated possessions per 48 minutes"
        elif "turnovers_per100" in metric:
            unit = "turnovers per 100 estimated possessions"
        elif metric == "max_player_minutes_per_game":
            unit = "minutes per game"
        if remainder in basics:
            fields = ["game_date_et"] if window == "previous_game" else ["game_id", "home_score", "away_score"]
            missingness = "Defined after the fixed minimum-history eligibility rule"
        elif metric.startswith("player_boxscore_") or metric.startswith("team_boxscore_"):
            fields = ["complete", "players_complete", "prior_game_ids"]
            missingness = "Explicit coverage indicator; a missing source is counted in the denominator"
        elif metric in ("max_player_minutes_per_game", "top5_minutes_share", "minutes_concentration",
                        "top3_offensive_involvement_share", "rotation_players_10plus_minutes_per_game",
                        "consecutive_game_minutes_overlap"):
            fields = ["players.player_id", "players.minutes"]
            if metric == "top3_offensive_involvement_share":
                fields += ["players.fga", "players.fta", "players.tov"]
            missingness = "Null when any game in the actual window lacks a complete player box score"
        else:
            fields = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "game_minutes"]
            missingness = "Pools valid team boxes in the actual window; coverage columns report missing sources"
        rows.append({"position": index, "name": name, "side": side, "window": window,
                     "definition": definition, "unit": unit, "source_fields": fields,
                     "dtype": "float64", "nullable": True, "missingness": missingness})
    return rows


def stage_dataset(root, stage_dir, hooks=()):
    """Convert and verify frozen inputs. Only the fresh staging directory is written."""
    root, stage = Path(root), Path(stage_dir)
    import inspect
    implementation_paths = {Path(__file__).resolve(), *(Path(inspect.getsourcefile(hook)).resolve() for hook in hooks)}
    implementation_sha256 = {str(path.relative_to(ROOT)): sha256(path) for path in implementation_paths}
    if stage.exists() and any(stage.iterdir()):
        raise ValueError("Staging directory must be empty; existing data is never overwritten")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / ".incomplete").write_text("Conversion in progress; do not publish.\n")
    start = time.perf_counter()
    metrics_path = root / "reports/gradient_boosting/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    _verify_frozen_sources(root, metrics.get("source_sha256", {}))
    config, names = metrics["config"], metrics["feature_names"]
    sources = {}
    cleanup_coverage = {}
    extra_equality = {}
    imported_prediction_sha256 = {}

    def record(path):
        path = Path(path)
        relative = str(path.relative_to(root))
        sources[relative] = sha256(path)
        return path

    def cover(path, replacement):
        path = record(path)
        cleanup_coverage[str(path.relative_to(root))] = {"sha256": sha256(path), "replacement": replacement}

    record(metrics_path)
    original = root / "data/processed/gradient_boosting"
    inputs = read_rows(record(original / "inputs.jsonl"))
    old_labels = read_rows(record(original / "labels.jsonl"))
    excluded = read_rows(record(original / "excluded.jsonl"))
    forecasts = read_rows(record(original / "predictions.jsonl"))
    for path, expected in metrics["artifact_sha256"].items():
        actual = sha256(record(root / path))
        if actual != expected:
            raise ValueError(f"Frozen artifact differs from its original manifest: {path}")
    history = [row for source in config["histories"] for row in read_rows(record(root / source["path"]))]
    _unique(history, "game_id", "schedule")
    inputs_by_id = _unique(inputs, "game_id", "input")
    excluded_by_id = _unique(excluded, "game_id", "exclusion")
    box_dir = root / config["box_scores"]
    box_manifest = json.loads(record(box_dir / "manifest.json").read_text())
    boxes = read_rows(record(box_dir / "games.jsonl"))
    if sources[str((box_dir / "games.jsonl").relative_to(root))] != box_manifest["artifact_sha256"]["games.jsonl"]:
        raise ValueError("Frozen normalized box scores differ from their manifest")
    boxes_by_id = _unique(boxes, "game_id", "box score")
    feature_order = {row["game_id"]: i for i, row in enumerate(inputs)}
    split_for_season = {season: split for split, (first, last) in config["split_seasons"].items() for season in range(first, last + 1)}
    from collect_rich_stats import NBA_FRANCHISES
    from collect_pilot import ABBR
    games, labels = [], {split: [] for split in SPLITS}
    for index, old in enumerate(history):
        game = remap_paths(old)
        gid = game["game_id"]
        split = split_for_season[game["season"]]
        labels[split].append({"game_id": gid, "home_win": int(game["home_score"] > game["away_score"]),
                              "home_score": game["home_score"], "away_score": game["away_score"]})
        row = {key: value for key, value in game.items() if key not in ("home_score", "away_score")}
        row.update(schedule_index=index, feature_index=feature_order.get(gid), split=split,
                   franchise_game=all(ABBR.get(game[f"{side}_abbr"], game[f"{side}_abbr"]) in NBA_FRANCHISES for side in ("home", "away")),
                   feature_eligible=gid in inputs_by_id,
                   cutoff_utc=_stamp(game["scheduled_start_utc"]) - timedelta(minutes=config["cutoff_minutes"]),
                   box_score_coverage=_pack(inputs_by_id[gid]["box_score_coverage"]) if gid in inputs_by_id else None,
                   exclusion=_pack(excluded_by_id[gid]) if gid in excluded_by_id else None)
        row["scheduled_start_utc"] = _stamp(row["scheduled_start_utc"])
        row["game_date_et"] = date.fromisoformat(row["game_date_et"])
        games.append(row)
    tables = {}

    def put(name, rows, schema=None):
        tables[name] = write_table(stage / (name + ".parquet"), rows, schema)

    put("games", games)
    feature_schema = pa.schema([("game_id", pa.string()), *[(name, pa.float64()) for name in names]])
    for row in inputs:
        if set(row["features"]) != set(names):
            raise ValueError("Input feature schema differs from the frozen model")
    put("features", ({"game_id": row["game_id"], **row["features"]} for row in inputs), feature_schema)
    for split in SPLITS:
        put("labels_" + split, labels[split], pa.schema([("game_id", pa.string()), ("home_win", pa.int8()),
                                                       ("home_score", pa.int16()), ("away_score", pa.int16())]))
    lineage = []
    for row in inputs:
        for side in ("home", "away"):
            missing = set(row["missing_box_score_game_ids"][side])
            for index, prior in enumerate(row["prior_game_ids"][side]):
                lineage.append({"game_id": row["game_id"], "side": side, "prior_index": index,
                                "prior_game_id": prior, "missing_box_score": prior in missing})
    put("feature_lineage", lineage)
    tables.update(write_box_tables(stage, boxes))
    predictions = []
    by_game = {game["game_id"]: game for game in games}
    for index, row in enumerate(forecasts):
        predictions.append({**{k: row[k] for k in ("game_id", "season", "game_date_et", "split", "produced_by")},
                            "model_id": "gradient_boosting", "run_id": "frozen_gradient_boosting", "stage": "baseline",
                            "prediction_index": index, "p_home_win": row["p_boost"]})
    install_files = {}
    legacy = {"reports/gradient_boosting/metrics.json": metrics,
              str((box_dir / "manifest.json").relative_to(root)): box_manifest}
    for stage_name in ("train-smoke", "dev"):
        folder = root / "data/processed/tabpfn" / stage_name
        if not folder.exists():
            continue
        for name in ("manifest.json", "run.json"):
            path = record(folder / name)
            legacy[str(path.relative_to(root))] = json.loads(path.read_text())
            target = "reports/tabpfn/" + stage_name + "/" + name
            export = stage / "exports" / target
            export.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, export)
            install_files[str(export.relative_to(stage))] = target
        prepared_manifest = legacy[str((folder / "manifest.json").relative_to(root))]
        for filename, expected in prepared_manifest["artifact_sha256"].items():
            if Path(filename).name != filename or sha256(folder / filename) != expected:
                raise ValueError(f"Frozen TabPFN prepared artifact checksum mismatch: {stage_name}/{filename}")
        with record(folder / "predictions.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
            imported_prediction_sha256[stage_name] = logical_hash([
                {"game_id": row["game_id"], "game_date_et": row["game_date_et"],
                 "tabpfn": float(row["tabpfn"]), "rich_gradient_boosting": float(row["rich_gradient_boosting"])}
                for row in csv_rows])
            for index, row in enumerate(csv_rows):
                game = by_game[row["game_id"]]
                for field, model_id in (("tabpfn", "tabpfn"), ("rich_gradient_boosting", "tabpfn_comparator")):
                    predictions.append({"game_id": row["game_id"], "game_date_et": row["game_date_et"],
                                        "season": game["season"], "split": game["split"], "model_id": model_id,
                                        "run_id": "frozen_tabpfn", "stage": stage_name, "prediction_index": index,
                                        "p_home_win": float(row[field]), "produced_by": "context" if model_id == "tabpfn" else "train_fit"})
        run = legacy[str((folder / "run.json").relative_to(root))]
        if run["status"] != "complete":
            raise ValueError("Only a completed TabPFN run can retire its resume state")
        for relative, expected in run["artifact_sha256"].items():
            if Path(relative).name != relative or sha256(folder / relative) != expected:
                raise ValueError("TabPFN run artifact differs from its frozen run record")
        if (folder / "prediction_progress.json").exists():
            progress = json.loads(record(folder / "prediction_progress.json").read_text())
            expected_rows = [{"game_id": row["game_id"], "game_date_et": row["game_date_et"], "tabpfn": float(row["tabpfn"])}
                             for row in csv_rows]
            checksum = hashlib.sha256(json.dumps(progress["predictions"], sort_keys=True, allow_nan=False).encode()).hexdigest()
            if progress["predictions"] != expected_rows or progress["predictions_sha256"] != checksum:
                raise ValueError("Completed TabPFN progress differs from saved predictions")
            if any(progress[key] != run[key] for key in ("manifest_sha256", "versions", "attempts", "context_audit")):
                raise ValueError("Unique TabPFN progress metadata is not preserved in run.json")
            if not math.isfinite(progress["runtime_seconds"]) or not 0 <= progress["runtime_seconds"] <= run["runtime_seconds"]:
                raise ValueError("TabPFN progress runtime exceeds the completed run")
            extra_equality["tabpfn_" + stage_name + "_progress"] = {
                "exact": True, "predictions": len(expected_rows), "runtime_seconds": progress["runtime_seconds"]}
            cover(folder / "prediction_progress.json", ["predictions.parquet", "reports/tabpfn/" + stage_name + "/run.json"])
        if (folder / "baseline_predictions.csv").exists():
            with record(folder / "baseline_predictions.csv").open() as stream:
                baseline_rows = list(csv.DictReader(stream))
            by_id = {row["game_id"]: row for row in forecasts}
            if any(float(row["rich_gradient_boosting"]) != by_id[row["game_id"]]["p_boost"] for row in baseline_rows):
                raise ValueError("TabPFN's saved baseline differs from the shared boosting forecasts")
            if [row["game_id"] for row in baseline_rows] != [row["game_id"] for row in csv_rows]:
                raise ValueError("TabPFN's saved baseline cohort differs from its prediction cohort")
            cover(folder / "baseline_predictions.csv", "predictions.parquet: gradient_boosting")
        for filename, replacement in (("inputs.jsonl", ["games.parquet", "features.parquet", "feature_lineage.parquet"]),
                                      ("labels.jsonl", ["labels_train.parquet", "labels_dev.parquet"]),
                                      ("predictions.csv", "predictions.parquet")):
            cover(folder / filename, replacement)
        for filename in ("manifest.json", "run.json"):
            cover(folder / filename, "reports/tabpfn/" + stage_name + "/" + filename)
        extra_equality["tabpfn_" + stage_name] = {"predictions": len(csv_rows), "exact": True,
                                                 "resume_state": "completed, verified redundant with predictions and run metadata"}
        # Confirm that duplicate feature/label cohorts contain no unique data.
        prepared = read_rows(record(folder / "inputs.jsonl"))
        prepared_labels = read_rows(record(folder / "labels.jsonl"))
        if prepared != [inputs_by_id[r["game_id"]] for r in prepared]:
            raise ValueError("TabPFN feature copy differs from shared source")
        labels_by_id = {r["game_id"]: r for r in old_labels}
        if prepared_labels != [labels_by_id[r["game_id"]] for r in prepared_labels]:
            raise ValueError("TabPFN label copy differs from shared source")
    put("predictions", predictions)
    model = record(original / "model.joblib")
    model_target = "data/models/gradient_boosting/model.joblib"
    export = stage / "exports" / model_target
    export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, export)
    install_files[str(export.relative_to(stage))] = model_target
    # Heterogeneous failures/repairs are audit evidence, not opaque model features.
    box_audit = {}
    for path in sorted(box_dir.glob("*.jsonl")):
        if path.name != "games.jsonl":
            box_audit[path.stem] = remap_paths(read_rows(record(path)))
            cover(path, "box_score_audit.json:" + path.stem)
    if (box_dir / "progress.json").exists():
        box_audit["collection_progress"] = json.loads(record(box_dir / "progress.json").read_text())
        cover(box_dir / "progress.json", "box_score_audit.json:collection_progress")
    cover(box_dir / "games.jsonl", ["box_score_games.parquet", "team_box_scores.parquet", "player_box_scores.parquet", "sources.parquet"])
    cover(box_dir / "manifest.json", "manifest.json:legacy_records")
    for source in config["histories"]:
        cover(root / source["path"], ["games.parquet", "labels_train.parquet", "labels_dev.parquet", "labels_test.parquet"])
    for filename, replacement in (("inputs.jsonl", ["games.parquet", "features.parquet", "feature_lineage.parquet"]),
                                  ("labels.jsonl", ["labels_train.parquet", "labels_dev.parquet", "labels_test.parquet"]),
                                  ("excluded.jsonl", "games.parquet:exclusion"), ("predictions.jsonl", "predictions.parquet"),
                                  ("model.joblib", model_target)):
        cover(original / filename, replacement)
    write_json(stage / "box_score_audit.json", box_audit)
    write_json(stage / "feature_dictionary.json", _feature_dictionary(names))
    remove_candidates = [str(original.relative_to(root)), str(box_dir.relative_to(root))]
    for stage_name in ("train-smoke", "dev"):
        candidate = "data/processed/tabpfn/" + stage_name
        if (root / candidate).exists():
            remove_candidates.append(candidate)
    for source in config["histories"]:
        path = root / source["path"]
        files = [p for p in path.parent.rglob("*") if p.is_file()]
        remove_candidates.append(str((path.parent if files == [path] else path).relative_to(root)))
    manifest = {"schema_version": SCHEMA_VERSION, "feature_names": names, "cutoff_minutes": config["cutoff_minutes"],
                "source_revision": SOURCE_REVISION,
                "split_seasons": config["split_seasons"], "tables": tables, "source_sha256": sources,
                "legacy_records": legacy, "install_files": install_files, "artifacts": {},
                "cleanup_coverage": cleanup_coverage,
                "imported_prediction_sha256": imported_prediction_sha256,
                "remove_candidates": remove_candidates,
                "availability_limitation": box_manifest.get("limitation"), "equality": extra_equality}
    write_json(stage / "manifest.json", manifest)
    dataset = Dataset(root, stage)
    comparisons = {"inputs": (dataset.inputs(), inputs), "labels": (dataset.labels(SPLITS), old_labels),
                   "history": (dataset.history(include_non_franchise=True), remap_paths(history)),
                   "box_scores": (dataset.box_scores(), remap_paths(boxes)),
                   "excluded": (dataset.excluded(), excluded), "boosting_predictions": (dataset.predictions(), forecasts)}
    for name, (actual, expected) in comparisons.items():
        if actual != expected:
            for index, (left, right) in enumerate(zip(actual, expected)):
                if left != right:
                    raise ValueError(f"Migration round trip differs for {name} row {index}: {left.get('game_id')}")
            raise ValueError(f"Migration row count differs for {name}")
        manifest["equality"][name] = {"rows": len(actual), "logical_sha256": logical_hash(actual), "exact": True}
    del comparisons, dataset, lineage, boxes, inputs
    for hook in hooks:
        result = hook(root, stage) or {}
        manifest["source_sha256"].update(result.get("source_sha256", {}))
        manifest["equality"].update(result.get("equality", {}))
        manifest["install_files"].update(result.get("install_files", result.get("move_files", {})))
        manifest["remove_candidates"].extend(result.get("remove_candidates", []))
        manifest["cleanup_coverage"].update(result.get("cleanup_coverage", {}))
    manifest["tables"] = {path.stem: {"file": path.name, "sha256": sha256(path),
                                      "rows": pq.read_metadata(path).num_rows, "schema": str(pq.read_schema(path))}
                           for path in sorted(stage.glob("*.parquet"))}
    manifest["artifacts"] = {str(path.relative_to(stage)): sha256(path) for path in sorted(stage.rglob("*"))
                             if path.is_file() and path not in (stage / "manifest.json", stage / ".incomplete", stage / PREDICTION_LOCK)
                             and path.suffix != ".parquet"}
    manifest["dataset_id"] = logical_hash({name: entry["sha256"] for name, entry in manifest["tables"].items() if name != "predictions"})
    manifest["runtime_seconds"] = time.perf_counter() - start
    for relative, expected in implementation_sha256.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Converter changed during staging: {relative}")
    manifest["writer"] = {"pyarrow": pa.__version__, "implementation_sha256": implementation_sha256}
    for relative, expected in manifest["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source changed during conversion: {relative}")
    write_json(stage / "manifest.json", manifest)
    Dataset(root, stage).verify()
    (stage / ".incomplete").unlink()
    return manifest


def _destination(root, relative):
    """Only install declared data/model or report files, never arbitrary paths."""
    relative = str(relative)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Migration destinations must be repository-relative")
    roots = ("data/processed/", "data/models/", "reports/")
    matched = next((prefix for prefix in roots if relative.startswith(prefix)), None)
    if matched is None:
        raise ValueError("Migration destination is outside processed data, models and reports")
    result = Path(root) / relative
    if not result.resolve().is_relative_to((Path(root) / matched).resolve()):
        raise ValueError("Migration destination escapes its declared directory")
    return result


def _original_path(root, relative):
    path = Path(root) / relative
    if path.exists():
        return path
    if relative.startswith("data/raw/") or relative in RAW_FILES:
        return resolve_source_path(root, relative)
    raise FileNotFoundError(relative)


def _check_sources(root, manifest, allow_removed=False):
    for relative, expected in manifest["source_sha256"].items():
        try:
            path = _original_path(root, relative)
        except FileNotFoundError:
            if allow_removed and relative in manifest.get("cleanup_coverage", {}):
                continue
            raise
        if sha256(path) != expected:
            raise ValueError(f"Original changed after staging: {relative}")


def _raw_move_plan(root):
    root = Path(root)
    moves, inventory = [], []
    mapping = [("data/raw/" + old, "data/raw/" + new) for old, new in RAW_DIRECTORIES.items()] + list(RAW_FILES.items())
    for old, new in mapping:
        source, destination = root / old, root / new
        if not source.exists():
            continue
        if source.is_symlink() or destination.exists():
            raise ValueError(f"Ambiguous raw relocation: {old} -> {new}")
        paths = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in paths:
            if path.is_symlink():
                raise ValueError("Raw migration refuses untracked symlinks")
            if path.is_file():
                suffix = str(path.relative_to(source)) if source.is_dir() else ""
                target = destination / suffix if suffix else destination
                inventory.append({"old_path": str(path.relative_to(root)), "path": str(target.relative_to(root)),
                                  "sha256": sha256(path), "bytes": path.stat().st_size})
        moves.append((source, destination))
    return moves, inventory


def publish_stage(root, stage_dir, *, relocate_raw=True):
    """Publish a checked candidate and regroup raw files, rolling back any failure.

    Old processed inputs are deliberately retained. The separately invoked cleanup
    command is the boundary after which rollback to old paths is no longer offered.
    """
    root, stage = Path(root), Path(stage_dir)
    if (stage / ".incomplete").exists():
        raise ValueError("Cannot publish an incomplete conversion")
    dataset = Dataset(root, stage)
    manifest = json.loads(json.dumps(dataset.manifest))
    final = root / "data/processed"
    if (final / "manifest.json").exists():
        installed = Dataset(root)
        if installed.manifest.get("dataset_id") == manifest["dataset_id"] and installed.manifest.get("publication", {}).get("status") == "complete":
            installed.verify()
            return installed.manifest
        raise ValueError("A different shared dataset already exists; refusing to overwrite it")
    dataset.verify()
    _check_sources(root, manifest)
    planned = []
    for name, entry in manifest["tables"].items():
        planned.append((stage / entry["file"], _destination(root, "data/processed/" + entry["file"]), entry["sha256"]))
    local_artifacts = {name: expected for name, expected in manifest["artifacts"].items() if not name.startswith("exports/")}
    for relative, expected in local_artifacts.items():
        planned.append((stage / relative, _destination(root, "data/processed/" + relative), expected))
    external = {}
    for source, destination in manifest.get("install_files", {}).items():
        if not source.startswith("exports/") or ".." in Path(source).parts:
            raise ValueError("Only staged exports may install outside the shared tables")
        export = stage / source
        expected = manifest["artifacts"].get(source)
        if expected is None or sha256(export) != expected:
            raise ValueError(f"Unverified staged export: {source}")
        planned.append((export, _destination(root, destination), expected))
        external[destination] = expected
    exports = stage / "exports"
    if exports.exists():
        actual = {str(path.relative_to(stage)) for path in exports.rglob("*") if path.is_file()}
        if actual != set(manifest.get("install_files", {})):
            raise ValueError("Staged exports contain files without an install mapping")
    destinations = [target.resolve() for _, target, _ in planned]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate migration destination")
    for _, target, _ in planned:
        if target.exists():
            raise FileExistsError(f"Migration target already exists: {target}")
    # Inventory before moving: raw downloads and sidecar bytes are never rewritten.
    raw_moves, raw_inventory = _raw_move_plan(root)
    write_json(stage / "raw_migration.json", {"files": raw_inventory, "bytes": sum(row["bytes"] for row in raw_inventory),
                                               "status": "complete" if relocate_raw else "pending",
                                               "method": "filesystem rename; exact body and sidecar bytes retained"})
    raw_hash = sha256(stage / "raw_migration.json")
    if (final / "raw_migration.json").exists():
        raise FileExistsError(final / "raw_migration.json")
    planned.append((stage / "raw_migration.json", final / "raw_migration.json", raw_hash))
    local_artifacts["raw_migration.json"] = raw_hash
    manifest["artifacts"] = local_artifacts
    manifest["external_artifacts"] = external
    journal_path = stage / "publish_journal.json"
    journal = {"status": "publishing", "installed": [], "raw_moves": [], "dataset_id": manifest["dataset_id"]}
    write_json(journal_path, journal)
    created_directories = []

    def parents(path):
        missing = []
        current = path.parent
        while not current.exists():
            missing.append(current)
            current = current.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        created_directories.extend(reversed(missing))

    try:
        for source, target, expected in planned:
            parents(target)
            temporary = target.with_name(target.name + ".publishing")
            if temporary.exists():
                raise FileExistsError(temporary)
            # Register intent before touching the filesystem so rollback sees partial copies.
            journal["installed"].append({"path": str(target.relative_to(root)), "sha256": expected,
                                         "temporary": str(temporary.relative_to(root)), "published": False})
            write_json(journal_path, journal)
            shutil.copyfile(source, temporary)
            if sha256(temporary) != expected:
                raise ValueError(f"Published copy differs from stage: {target}")
            if target.exists():
                raise FileExistsError(f"Migration target appeared during publication: {target}")
            temporary.replace(target)
            journal["installed"][-1]["published"] = True
        for source, target in raw_moves if relocate_raw else []:
            parents(target)
            entry = {"old": str(source.relative_to(root)), "new": str(target.relative_to(root))}
            journal["raw_moves"].append(entry)
            write_json(journal_path, journal)
            source.rename(target)
        for entry in raw_inventory:
            if sha256(root / entry["path" if relocate_raw else "old_path"]) != entry["sha256"]:
                raise ValueError(f"Raw bytes changed during relocation: {entry['path']}")
        manifest["publication"] = {"status": "complete", "published_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "raw_layout": "grouped" if relocate_raw else "original",
                                   "raw_files": len(raw_inventory), "raw_bytes": sum(row["bytes"] for row in raw_inventory)}
        write_json(stage / "published_manifest.json", manifest)
        parents(final / "manifest.json")
        shutil.copyfile(stage / "published_manifest.json", final / "manifest.json.publishing")
        (final / "manifest.json.publishing").replace(final / "manifest.json")
        Dataset(root).verify()
        journal["status"] = "complete"
        write_json(journal_path, journal)
    except BaseException as error:
        failures = []
        (final / "manifest.json").unlink(missing_ok=True)
        (final / "manifest.json.publishing").unlink(missing_ok=True)
        for entry in reversed(journal["raw_moves"]):
            source, target = root / entry["old"], root / entry["new"]
            if target.exists() and not source.exists():
                try:
                    target.rename(source)
                except OSError as rollback_error:
                    failures.append(str(rollback_error))
        for entry in reversed(journal["installed"]):
            for key in (("path", "temporary") if entry.get("published") else ("temporary",)):
                try:
                    (root / entry[key]).unlink(missing_ok=True)
                except OSError as rollback_error:
                    failures.append(str(rollback_error))
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except OSError:
                pass
        journal.update(status="rollback_failed" if failures else "rolled_back", error=str(error), rollback_errors=failures)
        write_json(journal_path, journal)
        if failures:
            raise RuntimeError(f"Publish failed and rollback needs attention: {journal_path}") from error
        raise
    # Only redundant staged exports are removed; the candidate tables remain until final checks.
    if exports.exists():
        shutil.rmtree(exports)
    return manifest


def relocate_raw_sources(root=ROOT):
    """Finish raw-path regrouping after readers using the old layout have stopped."""
    root = Path(root)
    dataset = Dataset(root)
    dataset.verify()
    if dataset.manifest.get("publication", {}).get("raw_layout") == "grouped":
        return dataset.manifest["publication"]
    inventory_path = dataset.directory / "raw_migration.json"
    previous_inventory = inventory_path.read_bytes()
    previous_manifest = (dataset.directory / "manifest.json").read_bytes()
    recorded = json.loads(previous_inventory)
    for entry in recorded["files"]:
        if sha256(root / entry["old_path"]) != entry["sha256"]:
            raise ValueError(f"Raw source changed while relocation was pending: {entry['old_path']}")
    moves, inventory = _raw_move_plan(root)
    journal_path = dataset.directory / ".raw_relocation.json"
    journal = {"status": "relocating", "moves": []}
    write_json(journal_path, journal)
    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            journal["moves"].append({"old": str(source.relative_to(root)), "new": str(target.relative_to(root))})
            write_json(journal_path, journal)
            source.rename(target)
        for entry in inventory:
            if sha256(root / entry["path"]) != entry["sha256"]:
                raise ValueError("Raw source changed during relocation")
        write_json(inventory_path, {"files": inventory, "bytes": sum(row["bytes"] for row in inventory),
                                    "status": "complete", "method": recorded["method"]})
        dataset.manifest["artifacts"]["raw_migration.json"] = sha256(inventory_path)
        dataset.manifest["publication"].update(raw_layout="grouped", raw_files=len(inventory),
                                                raw_bytes=sum(row["bytes"] for row in inventory))
        write_json(dataset.directory / "manifest.json", dataset.manifest)
        Dataset(root).verify()
        journal_path.unlink()
    except BaseException as error:
        failures = []
        for entry in reversed(journal["moves"]):
            old, new = root / entry["old"], root / entry["new"]
            if new.exists() and not old.exists():
                try:
                    new.rename(old)
                except OSError as rollback_error:
                    failures.append(str(rollback_error))
        inventory_path.write_bytes(previous_inventory)
        (dataset.directory / "manifest.json").write_bytes(previous_manifest)
        journal.update(status="rollback_failed" if failures else "rolled_back", error=str(error), rollback_errors=failures)
        write_json(journal_path, journal)
        if failures:
            raise RuntimeError(f"Raw relocation rollback needs attention: {journal_path}") from error
        raise
    return dataset.manifest["publication"]


def verify_raw_inventory(root, manifest=None):
    """Verify surviving source bytes, including sidecars, before removing duplicates."""
    root = Path(root)
    manifest = manifest if manifest is not None else Dataset(root).manifest
    inventory_path = root / "data/processed/raw_migration.json"
    if sha256(inventory_path) != manifest["artifacts"]["raw_migration.json"]:
        raise ValueError("Raw migration inventory checksum mismatch")
    inventory = json.loads(inventory_path.read_text())
    grouped = manifest["publication"].get("raw_layout") == "grouped"
    total_bytes = 0
    for entry in inventory["files"]:
        relative = entry["path" if grouped else "old_path"]
        if ".." in Path(relative).parts or not (relative.startswith("data/raw/") or (not grouped and relative in RAW_FILES)):
            raise ValueError("Raw inventory contains an invalid source path")
        path = root / relative
        allowed = root / ("data/raw" if relative.startswith("data/raw/") else "data")
        if path.is_symlink() or not path.resolve().is_relative_to(allowed.resolve()):
            raise ValueError("Surviving raw source escapes its recorded directory")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Surviving raw source changed or disappeared: {relative}")
        total_bytes += entry["bytes"]
    return {"files": len(inventory["files"]), "bytes": total_bytes}


def cleanup_dataset(root=ROOT, *, validated=False):
    """Remove only explicitly covered old files after the caller's consumer validation."""
    if not validated:
        raise ValueError("Run the migrated consumers and validators before requesting cleanup")
    root = Path(root)
    dataset = Dataset(root)
    dataset.verify()
    manifest = dataset.manifest
    if manifest.get("publication", {}).get("status") != "complete":
        raise ValueError("Only a published migration can retire original inputs")
    if manifest["publication"].get("raw_layout") != "grouped":
        raise ValueError("Raw relocation is pending; retain the original inputs for active readers")
    if manifest.get("cleanup", {}).get("status") == "complete":
        return manifest["cleanup"]
    retry = manifest.get("cleanup", {}).get("status") in ("in_progress", "complete")
    _check_sources(root, manifest, allow_removed=retry)
    raw_validation = verify_raw_inventory(root, manifest)
    coverage = manifest.get("cleanup_coverage", {})
    paths, directories = {}, []
    for relative in manifest["remove_candidates"]:
        target = _destination(root, relative)
        if not relative.startswith("data/processed/") or target.resolve() == dataset.directory.resolve():
            raise ValueError("Cleanup is restricted to superseded processed data")
        if not target.exists():
            if retry:
                continue
            raise FileNotFoundError(f"Cleanup candidate disappeared before validation: {relative}")
        if target.is_symlink():
            raise ValueError("Cleanup refuses symlink candidates")
        if target.is_dir():
            directories.append(target)
            children = target.rglob("*")
        else:
            children = [target]
        for path in children:
            if path.is_symlink():
                raise ValueError("Cleanup refuses symlink contents")
            if path.is_file():
                name = str(path.relative_to(root))
                record = coverage.get(name)
                if not isinstance(record, dict) or not record.get("replacement"):
                    raise ValueError(f"File lacks verified replacement coverage: {name}")
                if sha256(path) != record["sha256"]:
                    raise ValueError(f"Cleanup source changed since migration: {name}")
                paths[name] = path
    total_bytes = sum(path.stat().st_size for path in paths.values())
    manifest["cleanup"] = {"status": "in_progress", "consumer_validation_asserted": True,
                           "files": len(paths), "bytes": total_bytes, "surviving_raw_validation": raw_validation}
    write_json(dataset.directory / "manifest.json", manifest)
    for path in paths.values():
        path.unlink()
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        for child in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if child.is_dir():
                child.rmdir()
        directory.rmdir()
        # Retire an empty snapshot parent too, without touching other snapshots.
        parent = directory.parent
        if parent.name in ("bam_pipeline", "tabpfn") and not any(parent.iterdir()):
            parent.rmdir()
    manifest["cleanup"].update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(dataset.directory / "manifest.json", manifest)
    return manifest["cleanup"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("directory", type=Path)
    stage.add_argument("--core-only", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("directory", type=Path, nargs="?", default=ROOT / "data/processed")
    publish = sub.add_parser("publish")
    publish.add_argument("directory", type=Path)
    publish.add_argument("--keep-raw-layout", action="store_true", help="Publish shared tables while old readers continue using their raw paths")
    sub.add_parser("relocate-raw", help="Regroup raw files after old-layout readers have stopped")
    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--validated", action="store_true", help="Assert that migrated consumers and matching validators have passed")
    args = parser.parse_args()
    if args.command == "stage":
        hooks = []
        if not args.core_only:
            from bam_data import migrate_notes
            from collect_polymarket import migrate_market_data
            hooks = [migrate_notes, migrate_market_data]
        manifest = stage_dataset(ROOT, args.directory, hooks)
        print(json.dumps({"dataset_id": manifest["dataset_id"], "tables": {k: v["rows"] for k, v in manifest["tables"].items()},
                          "equality": manifest["equality"]}, indent=2))
    elif args.command == "publish":
        manifest = publish_stage(ROOT, args.directory, relocate_raw=not args.keep_raw_layout)
        print(json.dumps(manifest["publication"], indent=2))
    elif args.command == "relocate-raw":
        print(json.dumps(relocate_raw_sources(ROOT), indent=2))
    elif args.command == "cleanup":
        print(json.dumps(cleanup_dataset(ROOT, validated=args.validated), indent=2))
    else:
        print(json.dumps(Dataset(ROOT, args.directory).verify(), indent=2))


if __name__ == "__main__":
    main()
