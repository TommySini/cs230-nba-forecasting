"""End-to-end pilot orchestration on synthetic games: research rounds, rows, Bedrock teacher stages, bundle."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bam_bedrock as bedrock
import bam_concepts as concepts
import bam_pilot as pilot
import bam_research as research
import bam_train_contract as contract
import test_bam_research as research_fixture
import test_bam_train_contract as contract_fixture

TAXONOMY = concepts.load_taxonomy()


def role(provider, model, bedrock_id):
    return {"api": {"provider": provider, "model": model, "context_window_tokens": 128000, "max_output_tokens": 16000},
            "bedrock_model_id": bedrock_id, "region": "us-west-2",
            "price_per_million_input_usd": 0.15, "price_per_million_output_usd": 0.6}


def ready_config():
    config = pilot.load_config()
    config["research_critic"] = {"model_id": "fixture/critic", "model_revision": "c" * 40}
    config["teacher"] = role("openai", "gpt-oss-120b", "openai.gpt-oss-120b-1:0")
    config["reviewer"] = role("deepseek", "deepseek-v3.2", "deepseek.v3-2")
    return config


def answer_research(request):
    """What a compliant local model would return for each research stage."""
    if request["stage"] == "selection":
        return {"reviewed_chunk_id": request["chunk_id"],
                "findings": [research_fixture.finding()] if request["coverage"]["lines"] else [],
                "disposition": "Read the complete chunk."}
    if request["stage"] in ("module", "module_partial"):
        return research_fixture.module_output(request["module"])
    return {"decisions": [dict(research_fixture.review_decision({"id": i})) for i in request["expected_ids"]],
            "report_verdict": "supported", "report_reason": "Every finding is supported and qualified."}


class FakeBedrock:
    def __init__(self, fail_stage=None):
        self.calls, self.fail_stage = [], fail_stage

    def converse(self, **params):
        payload = json.loads(params["messages"][0]["content"][0]["text"])
        self.calls.append(params["modelId"])
        if "normalized_postgame_sources" in payload:
            text = {"general": [{"observation": "Brown did not play.", "foreseeable_signal": "Pregame rotation uncertainty.",
                                 "unforeseeable": "The outcome was uncertain.", "citations": [{"source_id": "p1", "page": 1, "lines": [1]}]}],
                    "timeline": ["Observed after tipoff."], "market_view": "Unavailable."}
        elif "concept_definitions" in payload:
            items = [{"name": n, "score": 0, "evidence_present": False, "decisive": False, "citations": [],
                      "rationale": "Not addressed."} for n in concepts.candidate_names(TAXONOMY)]
            items[0] = {"name": items[0]["name"], "score": .5, "evidence_present": True, "decisive": True,
                        "citations": [{"source_id": "s1", "page": 1, "lines": [1]}], "rationale": "Brown is questionable."}
            text = {"concepts": items}
        elif "candidate" in payload:
            text = {"decisions": [research_fixture.review_decision(f) for f in research.all_findings(payload["candidate"])],
                    "report_verdict": "supported", "report_reason": "Supported and qualified."}
        else:
            text = {"modules": [research_fixture.module_output(m) for m in research.MODULES]}
        return {"output": {"message": {"content": [{"text": json.dumps(text)}]}}, "stopReason": "end_turn",
                "usage": {"inputTokens": 1000, "outputTokens": 200}}


class FakeNumeric:
    def features(self, game_id):
        features = {name: 0.0 for name in contract.FEATURE_NAMES}
        return features, {"history_max_date": "2024-01-01", "history_game_ids": ["prior-feature-game"],
                          "normalized_history_sha256": "d" * 64, "features_sha256": research.value_hash(features)}

    def booster(self, season):
        return {"training_max_date": "2023-12-01", "imputer_fit_max_date": "2023-12-01", "training_game_ids": ["earlier-game"],
                "training_inputs_sha256": "b" * 64, "feature_schema_sha256": research.value_hash(list(contract.FEATURE_NAMES)),
                "model_sha256": "c" * 64}


def run_research_rounds(run, config, games):
    for _ in range(10):
        progress = pilot.stage_research(run, config, games)
        if not progress["pending_requests"]:
            return progress
        for role_name, directory in progress["rounds_written"].items():
            spec = json.loads((Path(directory) / "request.json").read_text())
            output = Path(directory) / "output"; output.mkdir(parents=True)
            (output / "responses.jsonl").write_text("".join(json.dumps(
                {"request_id": r["request_id"], "status": "complete", "finish_reason": "stop", "text": json.dumps(answer_research(r))}) + "\n"
                for r in spec["requests"]))
            pilot.stage_ingest(run, directory, games)
    raise AssertionError("research did not converge")


class PilotEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = {f"g{i}": contract_fixture.fixture_game(f"g{i}") for i in range(16)}
        cls.config = ready_config()

    def postgame(self):
        records = {}
        for game_id in self.games:
            record = json.loads(json.dumps(research_fixture.postgame()).replace("2025", "2024"))
            record["game_id"] = game_id
            records[game_id] = record
        return records

    def test_synthetic_pilot_reaches_a_valid_schema_two_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            progress = run_research_rounds(run, self.config, self.games)
            self.assertEqual(progress["games"], {"approved": 16})
            self.assertEqual(pilot.stage_rows(run, self.config, self.games, numeric=FakeNumeric())["rows"], 16)
            rows, postgame = pilot.load_rows(run), self.postgame()
            dry = pilot.stage_teacher(run, self.config, dry_run=True, rows=rows, postgame=postgame)
            self.assertTrue(dry["dry_run"]); self.assertEqual(dry["requests"], 16); self.assertGreater(dry["worst_case_cents"], 0)
            client = FakeBedrock()
            result = pilot.stage_teacher(run, self.config, client=client, rows=rows, postgame=postgame)
            self.assertEqual(result["games"], {"complete": 16})
            self.assertEqual(len(client.calls), 16 * 4)  # postmortem, gold, review, concepts
            self.assertEqual(client.calls.count("deepseek.v3-2"), 16)  # reviews only
            again = pilot.stage_teacher(run, self.config, client=FakeBedrock(), rows=rows, postgame=postgame)
            self.assertEqual(again["spent_cents"], result["spent_cents"])  # cached answers are never paid twice
            outcomes = {game_id: i % 2 for i, game_id in enumerate(self.games)}
            bundle = pilot.stage_assemble(run, self.config, rows=rows, postgame=postgame, outcomes=outcomes, snapshot_sha256="f" * 64)
            self.assertEqual(bundle["games"], 16)
            manifest, loaded, _, gold = contract.load_bundle(bundle["bundle"], bundle["manifest_sha256"])
            self.assertEqual(manifest["schema_version"], 2)
            labels = contract.load_concepts(bundle["bundle"], manifest, loaded, gold, TAXONOMY)
            self.assertEqual(len(labels), 16)
            with self.assertRaisesRegex(ValueError, "immutable"):
                pilot.stage_assemble(run, self.config, rows=rows, postgame=postgame, outcomes=outcomes, snapshot_sha256="f" * 64)

    def test_unready_config_blocks_paid_teacher_work(self):
        config = pilot.load_config()
        blockers = pilot.readiness(config)
        self.assertTrue(any("research_critic" in b for b in blockers))
        self.assertTrue(any(b.startswith("teacher") for b in blockers))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "not ready"):
            pilot.stage_teacher(directory, config, rows={}, postgame={})
        same = ready_config(); same["reviewer"] = copy.deepcopy(same["teacher"])
        self.assertIn("teacher and reviewer must be different models", pilot.readiness(same))

    def test_candidate_selection_is_deterministic_and_spread(self):
        rows = [{"game_id": f"x{i}", "split": "train", "season": 2023, "p_boost": (i % 10) / 10 + .05, "forecast_eligible": True,
                 "game_date_et": f"2024-01-{i % 28 + 1:02d}",
                 "sources": [{"publisher": ["CLE", "DET", "MEM"][i % 3], "source_kind": "pdf"}]} for i in range(60)]
        config = dict(self.config, candidate_games=12)
        first, second = pilot.choose_candidates(rows, config), pilot.choose_candidates(rows, config)
        self.assertEqual([r["game_id"] for r in first], [r["game_id"] for r in second])
        self.assertEqual(len({r["sources"][0]["publisher"] for r in first}), 3)
        self.assertLess(min(r["p_boost"] for r in first), .2); self.assertGreater(max(r["p_boost"] for r in first), .8)


class BedrockTransportTests(unittest.TestCase):
    def setUp(self):
        self.row, _, self.gold = contract_fixture.fixture_case()
        self.spec = role("openai", "gpt-oss-120b", "openai.gpt-oss-120b-1:0")
        self.request = concepts.build_concept_request(self.gold["teacher_request"], TAXONOMY)
        self.registry = pilot.registry_for({self.row["game_id"]: self.row}, [self.request], "f" * 64)

    def test_cached_answer_cap_and_stop_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeBedrock()
            record = bedrock.submit(self.request, self.spec, self.registry, directory, cap_cents=100, client=client)
            self.assertEqual((record["status"], record["finish_reason"]), ("complete", "end_turn"))
            self.assertEqual(record["cost_cents"], 1)  # ceil(1000*0.15 + 200*0.6 per million -> 0.027 cents)
            self.assertEqual(bedrock.submit(self.request, self.spec, self.registry, directory, cap_cents=100, client=client), record)
            self.assertEqual(len(client.calls), 1)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "Spend cap"):
            bedrock.submit(self.request, self.spec, self.registry, directory, cap_cents=0, client=FakeBedrock())
        self.assertEqual(bedrock.STOP_REASONS["max_tokens"], "length")

    def test_provider_errors_are_retried_then_recorded_without_cost(self):
        class Throttled(Exception):
            response = {"Error": {"Code": "ThrottlingException"}}
        class Failing:
            calls = 0
            def converse(self, **params):
                Failing.calls += 1
                raise Throttled("slow down")
        with tempfile.TemporaryDirectory() as directory:
            record = bedrock.submit(self.request, self.spec, self.registry, directory, cap_cents=100, client=Failing(),
                                    sleep=lambda seconds: None)
            self.assertEqual((record["status"], record["cost_cents"], Failing.calls), ("error", 0, 4))
            self.assertIn("ThrottlingException", record["error"])

    def test_unauthorized_requests_never_reach_bedrock(self):
        client = FakeBedrock()
        registry = dict(self.registry, authorized_training_requests={})
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            bedrock.submit(self.request, self.spec, registry, directory, cap_cents=100, client=client)
        self.assertEqual(client.calls, [])

    def test_converse_params_merge_user_turns(self):
        envelope = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "A"}, {"role": "user", "content": "B"}],
                    "max_output_tokens": 50}
        params = bedrock.converse_params(envelope, self.spec)
        self.assertEqual(params["messages"], [{"role": "user", "content": [{"text": "A"}, {"text": "B"}]}])
        self.assertEqual(params["inferenceConfig"], {"maxTokens": 50, "temperature": 0})


if __name__ == "__main__":
    unittest.main()
