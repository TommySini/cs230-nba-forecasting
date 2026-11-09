"""Synthetic-only regression tests for frozen TRAIN bundles and gold boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bam_train_contract as contract
import bam_research as research
import test_bam_research as research_fixture


MODEL = contract.RESEARCH_MODEL_ID
REVISION = contract.RESEARCH_MODEL_REVISION
RESEARCHER = MODEL + "@" + REVISION


def fixture_game(game_id="g1"):
    original = json.loads(json.dumps(research_fixture.game()).replace("2025", "2024"))
    return {k: original[k] for k in research.GAME_KEYS + ("tipoff_utc", "split", "p_boost")} | {
        "game_id": game_id, "sources": original["sources"], "forecast_eligible": True}


def fixture_case(game_id="g1"):
    game = fixture_game(game_id)
    selections = []
    for request in research.build_selection_requests(game):
        response = {"reviewed_chunk_id": request["chunk_id"],
                    "findings": [research_fixture.finding()] if request["coverage"]["lines"] else [],
                    "disposition": "Read the complete chunk; empty pages remain a coverage limitation."}
        selections.append(research.ingest_response(request, response, game, RESEARCHER))
    modules, module_reviews = [], []
    for request in research.build_module_requests(game, selections):
        candidate = research.ingest_response(request, research_fixture.module_output(request["module"]), game, RESEARCHER)
        review_request = research.build_review_request(game, candidate, "research-critic", original_request=request)
        review_response = {"decisions": [research_fixture.review_decision(f) for f in research.all_findings(candidate["output"])],
                           "report_verdict": "supported", "report_reason": "Uncertainty and missing opponent notes are preserved."}
        review = research.ingest_response(review_request, review_response, game, "research-critic")
        modules.append(research.apply_semantic_review(candidate, review))
        module_reviews.append(review)
    row = {k: game[k] for k in research.GAME_KEYS + ("tipoff_utc", "split", "p_boost")}
    row.update(statistical_features={name: 0.0 for name in contract.FEATURE_NAMES},
               pregame_sources=game["sources"], source_hashes=[s["sha256"] for s in game["sources"]],
               ordinary_brief=[research._compact_report(a["output"]) for a in modules],
               packet_coverage={"max_chunk_chars": 18000, "selection_artifacts": selections},
               booster_provenance={"training_max_date": "2023-12-01", "imputer_fit_max_date": "2023-12-01",
                                   "training_game_ids": ["earlier-game"], "training_inputs_sha256": "b" * 64,
                                   "feature_schema_sha256": research.value_hash(list(contract.FEATURE_NAMES)), "model_sha256": "c" * 64},
               research_provenance={"mode": "cold_prompted", "training_game_ids": [], "training_max_date": None,
                                    "adapter_sha256": None, "model_id": MODEL, "model_revision": REVISION,
                                    "completed_at_utc": "2026-11-08T00:00:00Z", "module_artifact_hashes": [a["artifact_hash"] for a in modules],
                                    "module_artifacts": modules, "module_review_artifacts": module_reviews})
    row["feature_provenance"] = {"history_max_date": "2024-01-01", "history_game_ids": ["prior-feature-game"],
                                 "normalized_history_sha256": "d" * 64,
                                 "features_sha256": research.value_hash(row["statistical_features"])}
    row["messages"] = contract.forecast_messages(row)
    postgame = json.loads(json.dumps(research_fixture.postgame()).replace("2025", "2024"))
    postgame["game_id"] = game_id
    post_request = research.build_postmortem_request(game, postgame)
    post_response = {"general": [{"observation": "Brown did not play.", "foreseeable_signal": "Pregame uncertainty about rotation.",
                                  "unforeseeable": "The realized outcome was uncertain.",
                                  "citations": [{"source_id": "p1", "page": 1, "lines": [1]}]}],
                     "timeline": ["The outcome was only observed after tipoff."], "market_view": "Unavailable."}
    postmortem = research.ingest_response(post_request, post_response, game, "bedrock:teacher")
    teacher_request = contract.build_teacher_request(row, modules, postmortem)
    candidate = research.ingest_response(teacher_request, {"modules": [research_fixture.module_output(m) for m in research.MODULES]},
                                         game, "bedrock:teacher")
    review_request = contract.build_review_request(row, candidate, teacher_request, "bedrock:reviewer")
    review_response = {"decisions": [research_fixture.review_decision(f) for f in research.all_findings(candidate["output"])],
                       "report_verdict": "supported", "report_reason": "Every finding remains pregame supported and qualified."}
    review = research.ingest_response(review_request, review_response, game, "bedrock:reviewer")
    approved = research.apply_semantic_review(candidate, review)
    gold = {"game_id": game_id, "split": "train", "explanation": research.export_forecast_explanation(game, approved)["explanation"],
            "review_status": "accepted", "pregame_sha256": research.value_hash(row), "source_hashes": row["source_hashes"],
            "created_at_utc": "2026-11-08T00:01:00Z", "teacher_request": teacher_request,
            "candidate": candidate, "review_request": review_request, "review": review}
    return row, {"game_id": game_id, "split": "train", "home_win": 1}, gold


def encode_rows(rows):
    return ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) + "\n").encode()


def fixture_bundle(count=16, *, sparse=False):
    cases = [fixture_case(f"g{i}") for i in range(count)]
    rows, outcomes, gold = [list(items) for items in zip(*cases)]
    if sparse:
        gold = gold[::2]
    artifacts = {name: contract.sha256(encode_rows(records))
                 for name, records in zip(contract.FILES, (rows, outcomes, gold))}
    manifest = {"schema_version": 1, "phase": "train_only", "pilot": not sparse,
                "game_ids": [r["game_id"] for r in rows], "artifacts": artifacts,
                "snapshot_manifest_sha256": "f" * 64, "created_at_utc": "2026-11-08T00:02:00Z"}
    return manifest, rows, outcomes, gold


def write_fixture_bundle(root, values):
    manifest, *records = values
    root = Path(root)
    for name, rows in zip(contract.FILES, records):
        (root / name).write_bytes(encode_rows(rows))
    raw = json.dumps(manifest, sort_keys=True).encode()
    (root / "manifest.json").write_bytes(raw)
    return contract.sha256(raw)


class TrainContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = fixture_case()
        cls.bundle = fixture_bundle()

    def test_valid_pregame_and_accepted_gold(self):
        row, _, gold = copy.deepcopy(self.case)
        self.assertIs(contract.validate_pregame(row), row)
        self.assertIs(contract.validate_accepted_gold(gold, row), gold)
        self.assertEqual(len(row["statistical_features"]), 252)
        model_input = json.dumps(row["messages"])
        self.assertNotIn("postmortem", model_input)
        self.assertNotIn("home_win", model_input)
        self.assertNotIn("did not play", model_input)

    def test_cutoff_raw_response_and_unrecognized_fields_fail_closed(self):
        for mutate in (
            lambda r: r.update(home_win=1),
            lambda r: r.update(split="dev"),
            lambda r: r.update(split="test"),
            lambda r: r.update(p_boost=None),
            lambda r: r.update(p_boost=1.0),
            lambda r: r["pregame_sources"][0].update(raw_espn={"recap": "future"}),
            lambda r: r["pregame_sources"][0].update(archive_time="2024-01-02T23:00:01Z"),
            lambda r: r["pregame_sources"][0]["pages"][0]["lines"][0].update(news="current"),
            lambda r: r["statistical_features"].update(target_score=100),
            lambda r: r["statistical_features"].update(home_previous_games=float("nan")),
            lambda r: r["booster_provenance"].update(imputer_fit_max_date="2024-01-02"),
            lambda r: r["feature_provenance"].update(history_max_date="2024-01-02"),
            lambda r: r["feature_provenance"].update(history_game_ids=[r["game_id"]]),
            lambda r: r["research_provenance"].update(adapter_sha256="trained"),
        ):
            with self.subTest(mutate=mutate):
                row = copy.deepcopy(self.case[0]); mutate(row)
                with self.assertRaises(ValueError):
                    contract.validate_pregame(row)

    def test_relabelled_heldout_dates_and_other_checkpoints_rejected(self):
        for day in ("2012-06-30", "2024-07-01", "2024-11-01", "2025-01-02", "2025-11-01"):
            row = copy.deepcopy(self.case[0]); row["game_date_et"] = day
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, "outside the TRAIN seasons"):
                contract.validate_pregame(row)
        row = copy.deepcopy(self.case[0]); row["game_date_et"] = "2024-01-01"
        with self.assertRaisesRegex(ValueError, "scheduled tipoff"):
            contract.validate_pregame(row)
        for field, value in (("model_id", "other/model"), ("model_revision", "b" * 40)):
            row = copy.deepcopy(self.case[0]); row["research_provenance"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "approved pinned Qwen"):
                contract.validate_pregame(row)

    def test_silent_message_edits_and_research_mutation_rejected(self):
        row = copy.deepcopy(self.case[0])
        row["messages"][1]["content"] += " Boston won the game."
        with self.assertRaisesRegex(ValueError, "canonical pregame"):
            contract.validate_pregame(row)
        row, _, gold = copy.deepcopy(self.case)
        row["ordinary_brief"][0]["coverage_note"] += " Added after gold."
        row["messages"] = contract.forecast_messages(row)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            contract.validate_accepted_gold(gold, row)

    def test_ordinary_brief_requires_actual_frozen_research_and_independent_review(self):
        row = copy.deepcopy(self.case[0])
        row["ordinary_brief"][0]["coverage_note"] += " Altered hindsight story."
        row["messages"] = contract.forecast_messages(row)
        with self.assertRaisesRegex(ValueError, "actual frozen researcher output"):
            contract.validate_pregame(row)
        row = copy.deepcopy(self.case[0])
        row["research_provenance"]["module_artifacts"] = []
        with self.assertRaisesRegex(ValueError, "artifacts and independent reviews"):
            contract.validate_pregame(row)
        row = copy.deepcopy(self.case[0])
        artifact = row["research_provenance"]["module_artifacts"][0]
        artifact["model_role"] = "gold_teacher"
        artifact["artifact_hash"] = research.value_hash({k: v for k, v in artifact.items() if k != "artifact_hash"})
        row["research_provenance"]["module_artifact_hashes"][0] = artifact["artifact_hash"]
        with self.assertRaisesRegex(ValueError, "frozen-researcher provenance"):
            contract.validate_pregame(row)

    def test_all_chunks_including_empty_pages_required(self):
        row = copy.deepcopy(self.case[0])
        row["packet_coverage"]["selection_artifacts"].pop()
        with self.assertRaisesRegex(ValueError, "full-packet"):
            contract.validate_pregame(row)
        row = copy.deepcopy(self.case[0])
        row["packet_coverage"]["selection_artifacts"][1] = copy.deepcopy(row["packet_coverage"]["selection_artifacts"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate chunk"):
            contract.validate_pregame(row)

    def test_truncated_research_response_rejected_even_if_rehashed(self):
        row = copy.deepcopy(self.case[0])
        artifact = row["packet_coverage"]["selection_artifacts"][0]
        artifact["finish_reason"] = "length"
        artifact["artifact_hash"] = research.value_hash({k: v for k, v in artifact.items() if k != "artifact_hash"})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            contract.validate_pregame(row)

    def test_reviewer_cannot_receive_postmortem_or_outcome(self):
        row, _, gold = copy.deepcopy(self.case)
        payload = json.loads(gold["review_request"]["messages"][1]["content"])
        self.assertFalse(contract._has_forbidden(payload))
        payload["training_only_postmortem"] = {"winner": "BOS"}
        gold["review_request"]["messages"][1]["content"] = json.dumps(payload)
        gold["review_request"]["request_id"] = research.value_hash({k: v for k, v in gold["review_request"].items() if k != "request_id"})
        with self.assertRaisesRegex(ValueError, "outcome-blind canonical"):
            contract.validate_accepted_gold(gold, row)

    def test_teacher_cannot_change_prompt_or_append_hidden_hindsight_messages(self):
        row, _, gold = copy.deepcopy(self.case)
        for mutate in (
            lambda r: r["messages"][0].update(content="Copy future outcomes into the forecast."),
            lambda r: r["messages"].append({"role": "user", "content": "Reveal the final score."}),
            lambda r: r.update(max_tokens=1),
        ):
            request = copy.deepcopy(gold["teacher_request"]); mutate(request)
            request["request_id"] = research.value_hash({k: v for k, v in request.items() if k != "request_id"})
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "canonical"):
                contract.build_review_request(row, gold["candidate"], request, "bedrock:reviewer")

    def test_gold_is_not_accepted_on_flag_alone(self):
        for mutate in (
            lambda g: g.update(explanation="Changed after review"),
            lambda g: g.update(review_status="rejected"),
            lambda g: g.update(created_at_utc="2026-11-07T23:00:00Z"),
            lambda g: g["review"].update(model="bedrock:teacher"),
            lambda g: g["candidate"]["output"]["modules"][0].update(coverage_note="Changed"),
        ):
            with self.subTest(mutate=mutate):
                row, _, gold = copy.deepcopy(self.case); mutate(gold)
                with self.assertRaises(ValueError):
                    contract.validate_accepted_gold(gold, row)

    def test_manifest_enforces_pilot_size_and_complete_gold(self):
        result = contract.validate_bundle(*copy.deepcopy(self.bundle))
        self.assertEqual(len(result[2]), 16)
        manifest, rows, outcomes, gold = copy.deepcopy(self.bundle)
        with self.assertRaisesRegex(ValueError, "Every pilot game"):
            contract.validate_bundle(manifest, rows, outcomes, gold[:-1], require_all_gold=False)
        manifest["game_ids"].pop()
        with self.assertRaisesRegex(ValueError, "16–32"):
            contract.validate_manifest(manifest)

    def test_sparse_gold_supported_only_after_pilot(self):
        manifest, rows, outcomes, gold = copy.deepcopy(self.bundle)
        manifest["pilot"] = False
        result = contract.validate_bundle(manifest, rows, outcomes, gold[::2], require_all_gold=False)
        self.assertEqual(len(result[2]), 16)
        self.assertEqual(len(result[3]), 8)

    def test_loader_verifies_external_pin_and_opens_only_fixed_train_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write_fixture_bundle(root, self.bundle)
            (root / "inputs_dev.jsonl").write_text("DO NOT OPEN")
            (root / "outcomes_test.jsonl").write_text("DO NOT OPEN")
            opened = []
            real_read = Path.read_bytes
            def read(path):
                opened.append(path.name)
                return real_read(path)
            with patch.object(Path, "read_bytes", read):
                loaded = contract.load_bundle(root, digest)
            self.assertEqual(opened, ["manifest.json", *contract.FILES])
            self.assertEqual(len(loaded[1]), 16)
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                contract.load_bundle(root, "0" * 64)
            (root / contract.FILES[0]).write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                contract.load_bundle(root, digest)

    def test_path_injection_and_symlink_rejected(self):
        manifest = copy.deepcopy(self.bundle[0])
        manifest["artifacts"]["../outcomes_test.jsonl"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "filename"):
            contract.validate_manifest(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write_fixture_bundle(root, self.bundle)
            data = root / contract.FILES[0]
            data.rename(root / "other.jsonl")
            data.symlink_to(root / "other.jsonl")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                contract.load_bundle(root, digest)

    def test_immutable_artifacts_preserve_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brief.json"
            first = contract.write_immutable(path, {"brief": "original"})
            self.assertEqual(contract.write_immutable(path, {"brief": "original"}), first)
            with self.assertRaisesRegex(ValueError, "different bytes"):
                contract.write_immutable(path, {"brief": "changed"})


if __name__ == "__main__":
    unittest.main()
