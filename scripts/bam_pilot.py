"""TRAIN concept-pilot orchestrator: from the shared dataset to a schema-2 training bundle.

Stages share one run directory and are safe to repeat; every model answer is stored
once by request ID and reused.

  check       report undecided config fields (models, prices) that block a paid run
  select      choose candidate TRAIN games (season, PDF-only, spread across publishers
              and boosting probabilities) from the shared BAM cohort
  research    write the next round of researcher/critic requests for bam_inference
  infer       (GPU) run one research round with vLLM on a pinned local checkpoint
  ingest      validate a finished round's answers into research artifacts
  rows        build pregame rows (features, booster and research provenance) for games
              whose three modules passed independent review
  postgame    normalize TRAIN postgame records (needs the raw data bundle)
  teacher     postmortem -> gold -> blind review -> concept labels through Bedrock
              (--dry-run writes requests and a worst-case cost only)
  assemble    write and validate the schema-2 bundle; prints its manifest SHA256
  status      per-game progress

Research rounds: run `research`, then `infer` on the GPU instance, then `ingest`;
repeat until `research` reports no pending requests. Failed or rejected answers drop
that game (the pilot over-selects candidates); nothing is retried silently.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import bam_concepts as concepts
import bam_research as research
import bam_train_contract as contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/bam_pilot.json"
PROCESSED = ROOT / "data/processed"
BOOSTING_MODEL = ROOT / "data/models/gradient_boosting/model.joblib"
SOURCE_KEYS = ("source_id", "publisher", "url", "sha256", "archive_time", "source_kind", "pagination")
PAGE_KEYS = ("page", "page_kind", "section", "label", "flags", "method", "coverage_audit")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_once(path, value):
    """Immutable artifact: a retry may only reproduce identical content."""
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        _require(path.read_text(encoding="utf-8") == content, f"Immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def model_name(spec):
    return f"{spec['model_id']}@{spec['model_revision']}"


# ------------------------------------------------------------------ config

def load_config(path=CONFIG_PATH):
    config = read_json(path)
    _require(config.get("schema_version") == 1, "Unknown pilot config schema")
    _require(16 <= config["pilot_games"] <= 32 and config["candidate_games"] >= config["pilot_games"],
             "Pilot needs 16-32 games and at least as many candidates")
    researcher = config["researcher"]
    _require((researcher["model_id"], researcher["model_revision"]) == (contract.RESEARCH_MODEL_ID, contract.RESEARCH_MODEL_REVISION),
             "The researcher must be the pinned Qwen2.5-14B checkpoint the contract accepts")
    return config


def readiness(config):
    """Decisions that must be made before paid work; empty means ready."""
    blockers = []
    critic = config["research_critic"]
    if not critic.get("model_id") or not critic.get("model_revision"):
        blockers.append("research_critic: choose a pinned local review model (model_id and 40-char revision)")
    elif model_name(critic) == model_name(config["researcher"]):
        blockers.append("research_critic must differ from the researcher")
    import bam_bedrock as bedrock
    for role in ("teacher", "reviewer"):
        try:
            bedrock.role_spec(config[role])
        except ValueError as error:
            blockers.append(f"{role}: {error}")
    if not blockers and bedrock.identity(config["teacher"]) == bedrock.identity(config["reviewer"]):
        blockers.append("teacher and reviewer must be different models")
    return blockers


# ------------------------------------------------------------------ games

def clean_sources(sources):
    """Contract-allowed source/page fields only; line text is preserved exactly."""
    result = []
    for source in sources:
        item = {key: source[key] for key in SOURCE_KEYS if source.get(key) is not None}
        item["pages"] = [{**{key: page[key] for key in PAGE_KEYS if page.get(key) is not None},
                          "lines": [{"line": line["line"], "text": line["text"]} for line in page["lines"]]}
                         for page in source["pages"]]
        result.append(item)
    return result


def research_game(snapshot_game):
    """The exact game view bam_train_contract._game rebuilds from a pregame row."""
    return {"game_id": snapshot_game["game_id"], "game_date_et": snapshot_game["game_date_et"],
            "cutoff_utc": snapshot_game["cutoff_utc"], "home_abbr": snapshot_game["home_abbr"],
            "away_abbr": snapshot_game["away_abbr"], "tipoff_utc": snapshot_game["scheduled_start_utc"],
            "split": "train", "p_boost": float(snapshot_game["p_boost"]),
            "sources": clean_sources(snapshot_game["sources"]), "forecast_eligible": True}


def choose_candidates(rows, config):
    """Deterministic spread across publishers and boosting probabilities."""
    eligible = [r for r in rows if r["split"] == "train" and r["season"] == config["season"] and r["p_boost"] is not None
                and r["forecast_eligible"] and r["sources"]
                and (not config["pdf_only"] or all(s["source_kind"] == "pdf" for s in r["sources"]))]
    _require(len(eligible) >= config["candidate_games"], "Too few eligible TRAIN games for the requested candidates")
    groups = defaultdict(list)
    for row in sorted(eligible, key=lambda r: (r["p_boost"], r["game_id"])):
        groups[row["sources"][0]["publisher"]].append(row)
    rng = random.Random(config["seed"])
    for publisher in groups:
        rng.shuffle(groups[publisher])
        groups[publisher].sort(key=lambda r: r["p_boost"])  # stable spread after a seeded tie-break
    picked, order = [], sorted(groups, key=lambda p: (-len(groups[p]), p))
    while len(picked) < config["candidate_games"]:
        for publisher in order:
            group = groups[publisher]
            if group and len(picked) < config["candidate_games"]:
                # Take alternately from the low and high ends of each publisher's probability range.
                picked.append(group.pop(0 if len(picked) % 2 == 0 else -1))
    return sorted(picked, key=lambda r: (r["game_date_et"], r["game_id"]))


def stage_select(run, config):
    from bam_data import read_snapshot_inputs
    rows = read_snapshot_inputs(PROCESSED, ROOT, splits=("train",))
    picked = choose_candidates(rows, config)
    record = {"game_ids": [r["game_id"] for r in picked], "season": config["season"], "pdf_only": config["pdf_only"],
              "publishers": dict(Counter(r["sources"][0]["publisher"] for r in picked)),
              "snapshot_manifest_sha256": hashlib.sha256((PROCESSED / "bam_manifest.json").read_bytes()).hexdigest(),
              "selected_at_utc": now_utc()}
    write_once(Path(run) / "games.json", record)
    return record


def load_games(run):
    """Research views of the selected games, cached once per run.

    Verifies the shared files once, then each source packet against its pinned page hash
    (what bam_data.load_snapshot_game does per call, without re-hashing every table per game).
    """
    from bam_data import _verify_shared_files, canonical, digest, read_note_packet, read_snapshot_inputs
    cache = Path(run) / "games"
    ids = read_json(Path(run) / "games.json")["game_ids"]
    missing = [g for g in ids if not (cache / f"{g.replace(':', '_')}.json").exists()]
    if missing:
        _verify_shared_files(PROCESSED, root=ROOT)
        snapshot = {r["game_id"]: r for r in read_snapshot_inputs(PROCESSED, ROOT, splits=("train",))}
        for game_id in missing:
            game = snapshot[game_id]
            sources = []
            for source in game["sources"]:
                if not source["active_for_research"]:
                    continue
                packet = read_note_packet(PROCESSED, source["sha256"])
                _require(digest(canonical(packet)) == source["pages_sha256"], "Source page checksum changed")
                sources.append({**source, "pages": packet["pages"]})
            write_once(cache / f"{game_id.replace(':', '_')}.json", research_game({**game, "sources": sources}))
    return {game_id: read_json(cache / f"{game_id.replace(':', '_')}.json") for game_id in ids}


# ---------------------------------------------------------------- research

class Store:
    """request_id -> request and ingested artifact, one file each."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def put_request(self, request):
        write_once(self.directory / "requests" / f"{request['request_id']}.json", request)

    def request(self, request_id):
        return read_json(self.directory / "requests" / f"{request_id}.json")

    def artifact(self, request_id):
        path = self.directory / "artifacts" / f"{request_id}.json"
        return read_json(path) if path.exists() else None

    def put_artifact(self, request_id, artifact):
        write_once(self.directory / "artifacts" / f"{request_id}.json", artifact)


def research_state(game, store, config):
    """(pending [(role, request)], result) where result is None, 'failed: ...' or the approved modules."""
    chunk = config["max_chunk_chars"]
    selection_requests = research.build_selection_requests(game, max_chunk_chars=chunk)
    selections, pending = [], []
    for request in selection_requests:
        artifact = store.artifact(request["request_id"])
        if artifact is None:
            pending.append(("researcher", request))
        elif artifact["status"] != "mechanically_valid":
            return [], "failed: selection " + "; ".join(artifact["errors"])[:200]
        else:
            selections.append(artifact)
    if pending:
        return pending, None
    requests, finals = research.build_module_requests(game, selections, max_chunk_chars=chunk), {}
    while requests:
        missing = [r for r in requests if store.artifact(r["request_id"]) is None]
        if missing:
            return [("researcher", r) for r in missing], None
        artifacts = [store.artifact(r["request_id"]) for r in requests]
        bad = [a for a in artifacts if a["status"] != "mechanically_valid"]
        if bad:
            return [], "failed: module " + "; ".join(bad[0]["errors"])[:200]
        for request, artifact in zip(requests, artifacts):
            if artifact["stage"] == "module":
                finals[artifact["module"]] = (request, artifact)
        partials = [a for a in artifacts if a["stage"] == "module_partial"]
        requests = research.build_module_synthesis_requests(game, partials) if partials else []
    _require(set(finals) == set(research.MODULES), "Module chain ended without all three modules")
    approved, reviews, pending = [], [], []
    critic = model_name(config["research_critic"])
    for module in research.MODULES:
        request, candidate = finals[module]
        review_request = research.build_review_request(game, candidate, critic, original_request=request)
        review = store.artifact(review_request["request_id"])
        if review is None:
            pending.append(("critic", review_request))
            continue
        result = research.apply_semantic_review(candidate, review)
        if result["status"] != "approved":
            return [], f"failed: {module} review rejected"
        approved.append(result)
        reviews.append(review)
    if pending:
        return pending, None
    return [], {"modules": approved, "reviews": reviews, "selections": selections}


def stage_research(run, config, games=None):
    """Write the next inference round (one file per model role); returns progress counts."""
    run = Path(run)
    games = games if games is not None else load_games(run)
    store = Store(run / "research")
    pending, status = defaultdict(list), Counter()
    for game in games.values():
        requests, result = research_state(game, store, config)
        status["pending" if requests else ("approved" if isinstance(result, dict) else "failed")] += 1
        for role, request in requests:
            store.put_request(request)
            pending[role].append(request)
    rounds = run / "research" / "rounds"
    written = {}
    for role, requests in pending.items():
        number = len(list(rounds.glob(f"*-{role}"))) if rounds.exists() else 0
        directory = rounds / f"{number:03d}-{role}"
        write_once(directory / "request.json", {"role": role, "model": config["researcher" if role == "researcher" else "research_critic"],
                                                "requests": requests, "config": {"seed": config["seed"], "generation": config["inference"]}})
        written[role] = str(directory)
    return {"games": dict(status), "rounds_written": written,
            "pending_requests": {role: len(requests) for role, requests in pending.items()}}


def model_manifest(snapshot_path, spec):
    """Hash every file of a downloaded checkpoint (run on the GPU instance)."""
    root = Path(snapshot_path)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.is_file() and ".cache" not in p.parts}
    _require(files, "Empty model snapshot")
    return {"model_spec": {"model_id": spec["model_id"], "model_revision": spec["model_revision"]},
            "snapshot_path": str(root), "files": files}


def stage_infer(round_dir, snapshot_path):
    from bam_inference import run_requests
    round_dir = Path(round_dir)
    spec = read_json(round_dir / "request.json")
    manifest = model_manifest(snapshot_path, spec["model"])
    return run_requests(round_dir / "output", spec["requests"], manifest, spec["config"])["summary"]


def stage_ingest(run, round_dir, games=None):
    run, round_dir = Path(run), Path(round_dir)
    games = games if games is not None else load_games(run)
    store = Store(run / "research")
    spec = read_json(round_dir / "request.json")
    model = model_name(spec["model"])
    by_id = {r["request_id"]: r for r in spec["requests"]}
    counts = Counter()
    for response in (json.loads(line) for line in (round_dir / "output/responses.jsonl").read_text().splitlines() if line.strip()):
        request = by_id[response["request_id"]]
        complete = response["status"] == "complete" and response["finish_reason"] == "stop"
        artifact = research.ingest_response(request, response["text"] if complete else response.get("raw_text", ""),
                                            games[request["game_id"]], model,
                                            finish_reason="stop" if complete else (response["finish_reason"] or "error"))
        store.put_artifact(request["request_id"], artifact)
        counts[artifact["status"]] += 1
    return dict(counts)


# -------------------------------------------------------------- pregame rows

class Numeric:
    """Prior-game features and booster provenance from the shared dataset and saved boosting models."""

    def __init__(self):
        import joblib
        from nba_dataset import Dataset
        self.dataset = Dataset(ROOT)
        self.inputs = {r["game_id"]: r for r in self.dataset.inputs(splits={"train"})}
        self.dates = {g["game_id"]: g["game_date_et"] for g in self.dataset.table("games")}
        self.games = {g["game_id"]: g for g in self.dataset.table("games")}
        self.bundle = joblib.load(BOOSTING_MODEL)
        self.model_sha256 = hashlib.sha256(BOOSTING_MODEL.read_bytes()).hexdigest()
        self._booster = {}

    def features(self, game_id):
        record = self.inputs[game_id]
        history = sorted(set(record["prior_game_ids"]["home"]) | set(record["prior_game_ids"]["away"]))
        features = dict(record["features"])
        return features, {"history_max_date": max(self.dates[g] for g in history), "history_game_ids": history,
                          "normalized_history_sha256": research.value_hash([self.games[g] for g in history]),
                          "features_sha256": research.value_hash(features)}

    def booster(self, season):
        if season not in self._booster:
            from run_gradient_boosting import FEATURE_NAMES
            fit = self.bundle["prior_seasons"][season]["fit_game_ids"]
            latest = max(self.dates[g] for g in fit)
            self._booster[season] = {"training_max_date": latest, "imputer_fit_max_date": latest, "training_game_ids": list(fit),
                                     "training_inputs_sha256": research.value_hash([[g, self.inputs[g]["features"]] for g in fit]),
                                     "feature_schema_sha256": research.value_hash(list(FEATURE_NAMES)), "model_sha256": self.model_sha256}
        return self._booster[season]


def pregame_row(game, approved, season, numeric, completed_at, max_chunk_chars):
    modules = approved["modules"]
    features, feature_provenance = numeric.features(game["game_id"])
    row = {key: game[key] for key in research.GAME_KEYS + ("tipoff_utc", "split", "p_boost")}
    row.update(statistical_features=features, feature_provenance=feature_provenance,
               booster_provenance=numeric.booster(season), pregame_sources=game["sources"],
               source_hashes=[s["sha256"] for s in game["sources"]],
               ordinary_brief=[research._compact_report(a["output"]) for a in modules],
               packet_coverage={"max_chunk_chars": max_chunk_chars, "selection_artifacts": approved["selections"]},
               research_provenance={"mode": "cold_prompted", "training_game_ids": [], "training_max_date": None,
                                    "adapter_sha256": None, "model_id": contract.RESEARCH_MODEL_ID,
                                    "model_revision": contract.RESEARCH_MODEL_REVISION, "completed_at_utc": completed_at,
                                    "module_artifact_hashes": [a["artifact_hash"] for a in modules],
                                    "module_artifacts": modules, "module_review_artifacts": approved["reviews"]})
    row["messages"] = contract.forecast_messages(row)
    return row


def stage_rows(run, config, games=None, numeric=None, seasons=None):
    run = Path(run)
    games = games if games is not None else load_games(run)
    store = Store(run / "research")
    built = 0
    for game_id, game in games.items():
        path = run / "rows" / f"{game_id.replace(':', '_')}.json"
        if path.exists():
            built += 1
            continue
        pending, result = research_state(game, store, config)
        if pending or not isinstance(result, dict):
            continue
        numeric = numeric or Numeric()
        season = (seasons or {}).get(game_id, config["season"])
        row = pregame_row(game, result, season, numeric, now_utc(), config["max_chunk_chars"])
        contract.validate_pregame(row)
        write_once(path, row)
        built += 1
    return {"rows": built}


def load_rows(run):
    return {read_json(p)["game_id"]: read_json(p) for p in sorted((Path(run) / "rows").glob("*.json"))}


# ------------------------------------------------------------------ teacher

def stage_postgame(run):
    import bam_postgame
    output = Path(run) / "postgame"
    if not output.exists():
        bam_postgame.prepare(PROCESSED, output)
    return read_json(output / "manifest.json")


def load_postgame(run):
    path = Path(run) / "postgame" / "records.jsonl"
    return {r["game_id"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def teacher_state(row, postgame, store, config, taxonomy):
    """(pending (role, request, targets) or None, gold record or None, concept record or None, failure or None)."""
    import bam_bedrock as bedrock
    game = contract._game(row)
    modules = row["research_provenance"]["module_artifacts"]
    post_request = research.build_postmortem_request(game, postgame)
    postmortem = store.artifact(post_request["request_id"])
    if postmortem is None:
        return ("teacher", post_request, None), None, None, None
    if postmortem["status"] != "mechanically_valid":
        return None, None, None, "postmortem rejected"
    gold_request = contract.build_teacher_request(row, modules, postmortem)
    candidate = store.artifact(gold_request["request_id"])
    if candidate is None:
        return ("teacher", gold_request, None), None, None, None
    if candidate["status"] != "mechanically_valid":
        return None, None, None, "gold candidate rejected"
    review_request = contract.build_review_request(row, candidate, gold_request, bedrock.identity(config["reviewer"]))
    review = store.artifact(review_request["request_id"])
    if review is None:
        return ("reviewer", review_request, {candidate["artifact_hash"]: candidate}), None, None, None
    approved = research.apply_semantic_review(candidate, review)
    if approved["status"] != "approved":
        return None, None, None, "gold review rejected"
    gold = {"game_id": row["game_id"], "split": "train",
            "explanation": research.export_forecast_explanation(game, approved)["explanation"],
            "review_status": "accepted", "pregame_sha256": research.value_hash(row), "source_hashes": row["source_hashes"],
            "created_at_utc": store.created(review_request["request_id"]), "teacher_request": gold_request,
            "candidate": candidate, "review_request": review_request, "review": review}
    concept_request = concepts.build_concept_request(gold_request, taxonomy)
    labels = store.artifact(concept_request["request_id"])
    if labels is None:
        return ("teacher", concept_request, None), gold, None, None
    if labels["status"] != "mechanically_valid":
        return None, gold, None, "concept labels rejected"
    record = {"game_id": row["game_id"], "split": "train", "concept_pool_sha256": concepts.pool_sha256(taxonomy),
              "created_at_utc": store.created(concept_request["request_id"]), "request": concept_request, "candidate": labels}
    return None, gold, record, None


class TeacherStore(Store):
    def put_artifact(self, request_id, artifact, created):
        super().put_artifact(request_id, artifact)
        write_once(self.directory / "created" / f"{request_id}.json", {"created_at_utc": created})

    def created(self, request_id):
        return read_json(self.directory / "created" / f"{request_id}.json")["created_at_utc"]


def registry_for(rows, requests, snapshot_sha256):
    return {"games": {r["game_id"]: {"game_id": r["game_id"], "split": "train", "game_date_et": r["game_date_et"],
                                     "cutoff_utc": r["cutoff_utc"]} for r in rows.values()},
            "trusted_source_hashes": {r["game_id"]: research.value_hash(contract._game(r)["sources"]) for r in rows.values()},
            "authorized_training_requests": {q["request_id"]: {k: q[k] for k in ("request_id", "game_id", "stage", "source_hash")}
                                             | {"request_sha256": research.value_hash(q)} for q in requests},
            "snapshot_manifest_sha256": snapshot_sha256}


def stage_teacher(run, config, *, client=None, dry_run=False, max_rounds=8, taxonomy=None, rows=None, postgame=None):
    """Advance every game through postmortem, gold, review and concept labels."""
    import bam_bedrock as bedrock
    run = Path(run)
    blockers = readiness(config)
    _require(not blockers, "Pilot config not ready: " + " | ".join(blockers))
    taxonomy = taxonomy or concepts.spec_taxonomy(concepts.head_spec(read_json(ROOT / config["forecaster_config"])))
    rows = rows if rows is not None else load_rows(run)
    postgame = postgame if postgame is not None else load_postgame(run)
    store = TeacherStore(run / "teacher")
    snapshot_sha = read_json(run / "games.json")["snapshot_manifest_sha256"] if (run / "games.json").exists() else "0" * 64
    cap = round(config["teacher_spend_cap_usd"] * 100)
    reservations = 0
    for _ in range(max_rounds):
        pending, failures = [], Counter()
        for row in rows.values():
            if row["game_id"] not in postgame:
                failures["no_postgame_record"] += 1
                continue
            step, _, _, failure = teacher_state(row, postgame[row["game_id"]], store, config, taxonomy)
            if failure:
                failures[failure] += 1
            elif step:
                pending.append(step)
        if not pending:
            break
        registry = registry_for(rows, [request for _, request, _ in pending], snapshot_sha)
        for role, request, targets in pending:
            store.put_request(request)
            response = bedrock.submit(request, config[role], registry, run / "teacher" / "responses", cap_cents=cap,
                                      client=client, targets=targets, dry_run=dry_run)
            if dry_run:
                reservations += response["reservation_cents"]
                continue
            game = contract._game(rows[request["game_id"]])
            finish = response["finish_reason"] or "error"
            if request["stage"] == concepts.STAGE:
                artifact = concepts.ingest_concept_response(request, response["text"], game, response["model"], taxonomy, finish_reason=finish)
            else:
                artifact = research.ingest_response(request, response["text"], game, response["model"], finish_reason=finish)
            store.put_artifact(request["request_id"], artifact, response["finished_at_utc"])
        if dry_run:
            return {"dry_run": True, "requests": len(pending), "worst_case_cents": reservations, "cap_cents": cap}
    done = Counter()
    for row in rows.values():
        if row["game_id"] in postgame:
            step, gold, record, failure = teacher_state(row, postgame[row["game_id"]], store, config, taxonomy)
            done["complete" if record else ("failed" if failure else "pending")] += 1
    return {"games": dict(done), "spent_cents": bedrock.spent_cents(run / "teacher" / "responses"), "cap_cents": cap,
            "failures": dict(failures)}


# ----------------------------------------------------------------- assemble

def encode(records):
    return ("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records) + "\n").encode()


def stage_assemble(run, config, *, taxonomy=None, rows=None, postgame=None, outcomes=None, snapshot_sha256=None):
    run = Path(run)
    taxonomy = taxonomy or concepts.spec_taxonomy(concepts.head_spec(read_json(ROOT / config["forecaster_config"])))
    rows = rows if rows is not None else load_rows(run)
    postgame = postgame if postgame is not None else load_postgame(run)
    store = TeacherStore(run / "teacher")
    ready = []
    for game_id in (read_json(run / "games.json")["game_ids"] if (run / "games.json").exists() else list(rows)):
        if game_id in rows and game_id in postgame:
            _, gold, record, _ = teacher_state(rows[game_id], postgame[game_id], store, config, taxonomy)
            if record:
                ready.append((rows[game_id], gold, record))
    _require(len(ready) >= 16, f"Only {len(ready)} games have accepted gold and concept labels; the pilot needs 16")
    ready = ready[:config["pilot_games"]]
    if outcomes is None:
        from nba_dataset import Dataset
        labels = {r["game_id"]: r["home_win"] for r in Dataset(ROOT).labels(["train"], eligible_only=False)}
        outcomes = {row["game_id"]: labels[row["game_id"]] for row, _, _ in ready}
    files = {"inputs_train.jsonl": [row for row, _, _ in ready],
             "outcomes_train.jsonl": [{"game_id": row["game_id"], "split": "train", "home_win": int(outcomes[row["game_id"]])}
                                      for row, _, _ in ready],
             "gold_train.jsonl": [gold for _, gold, _ in ready],
             contract.CONCEPT_FILE: [record for _, _, record in ready]}
    bundle = run / "bundle"
    _require(not (bundle / "manifest.json").exists(), "Bundle already assembled; it is immutable")
    bundle.mkdir(parents=True, exist_ok=True)
    for name, records in files.items():
        (bundle / name).write_bytes(encode(records))
    manifest = {"schema_version": 2, "phase": "train_only", "pilot": True, "game_ids": [row["game_id"] for row, _, _ in ready],
                "artifacts": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in files},
                "snapshot_manifest_sha256": snapshot_sha256 or read_json(run / "games.json")["snapshot_manifest_sha256"],
                "created_at_utc": now_utc(), "concept_pool_sha256": concepts.pool_sha256(taxonomy)}
    raw = json.dumps(manifest, sort_keys=True, indent=2).encode()
    (bundle / "manifest.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    loaded, loaded_rows, _, loaded_gold = contract.load_bundle(bundle, digest)
    contract.load_concepts(bundle, loaded, loaded_rows, loaded_gold, taxonomy)
    write_json(run / "bundle_manifest_sha256.json", {"manifest_sha256": digest, "games": len(ready)})
    return {"bundle": str(bundle), "manifest_sha256": digest, "games": len(ready)}


def stage_status(run, config):
    run = Path(run)
    result = {"selected": len(read_json(run / "games.json")["game_ids"]) if (run / "games.json").exists() else 0,
              "rows": len(list((run / "rows").glob("*.json"))),
              "postgame": (run / "postgame" / "manifest.json").exists(),
              "bundle": read_json(run / "bundle_manifest_sha256.json") if (run / "bundle_manifest_sha256.json").exists() else None,
              "blockers": readiness(config)}
    if (run / "teacher" / "responses").exists():
        import bam_bedrock as bedrock
        result["teacher_spent_cents"] = bedrock.spent_cents(run / "teacher" / "responses")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["check", "select", "research", "infer", "ingest", "rows", "postgame",
                                          "teacher", "assemble", "status"])
    parser.add_argument("--run", type=Path, default=ROOT / "runs/concept_pilot")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--round", type=Path, help="research round directory (infer/ingest)")
    parser.add_argument("--snapshot-path", type=Path, help="downloaded checkpoint directory (infer)")
    parser.add_argument("--dry-run", action="store_true", help="teacher: write requests and worst-case cost only")
    args = parser.parse_args()
    config = load_config(args.config)
    stages = {"check": lambda: {"blockers": readiness(config)},
              "select": lambda: stage_select(args.run, config),
              "research": lambda: stage_research(args.run, config),
              "infer": lambda: stage_infer(args.round, args.snapshot_path),
              "ingest": lambda: stage_ingest(args.run, args.round),
              "rows": lambda: stage_rows(args.run, config),
              "postgame": lambda: stage_postgame(args.run),
              "teacher": lambda: stage_teacher(args.run, config, dry_run=args.dry_run),
              "assemble": lambda: stage_assemble(args.run, config),
              "status": lambda: stage_status(args.run, config)}
    print(json.dumps(stages[args.stage](), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
