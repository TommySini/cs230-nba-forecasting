"""Concept-bottleneck head: taxonomy, TRAIN teacher labels, schema-2 bundles and joint training (synthetic only)."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bam_api_teacher as api
import bam_concepts as concepts
import bam_research as research
import bam_train_contract as contract
import test_bam_train_contract as contract_fixture

try:
    import torch
except ImportError:  # GPU/torch tests run where the training group is installed
    torch = None

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = concepts.load_taxonomy()


def label(name, score=0.0, present=False, decisive=False):
    return {"name": name, "score": score, "evidence_present": present, "decisive": decisive,
            "citations": [{"source_id": "s1", "page": 1, "lines": [1]}] if present else [],
            "rationale": "Pregame notes address this." if present else "Not addressed before the cutoff."}


def response(overrides=None):
    items = [label(name) for name in concepts.candidate_names(TAXONOMY)]
    for i, item in enumerate(items):
        if item["name"] in (overrides or {}):
            items[i] = overrides[item["name"]]
    return {"concepts": items}


def concept_record(row, gold, raw=None, model="bedrock:teacher", created="2026-11-08T00:01:30Z"):
    request = concepts.build_concept_request(gold["teacher_request"], TAXONOMY)
    raw = raw if raw is not None else response({"star_availability": label("star_availability", .6, True, True),
                                                "rest_advantage": label("rest_advantage", -.2, True)})
    candidate = concepts.ingest_concept_response(request, raw, contract._game(row), model, TAXONOMY)
    return {"game_id": row["game_id"], "split": "train", "concept_pool_sha256": concepts.pool_sha256(TAXONOMY),
            "created_at_utc": created, "request": request, "candidate": candidate}


class TaxonomyTests(unittest.TestCase):
    def test_repository_taxonomy_and_pinned_config_agree(self):
        self.assertEqual(len(TAXONOMY["candidates"]), 24)
        self.assertEqual(len(TAXONOMY["selected"]), 20)
        config = json.loads((ROOT / "config/bam_train_only_concept_pilot.json").read_text())
        spec = concepts.head_spec(config)
        self.assertEqual(spec["taxonomy_sha256"], concepts.taxonomy_sha256(TAXONOMY))
        self.assertEqual(concepts.spec_taxonomy(spec), TAXONOMY)

    def test_malformed_taxonomies_fail_closed(self):
        for mutate in (lambda t: t["candidates"].append(dict(t["candidates"][0])),
                       lambda t: t["selected"].append("not_a_candidate"),
                       lambda t: t["selected"].append(t["selected"][0]),
                       lambda t: t["candidates"][0].update(name="Bad Name"),
                       lambda t: t.update(extra=True)):
            taxonomy = copy.deepcopy(TAXONOMY)
            mutate(taxonomy)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                concepts.validate_taxonomy(taxonomy)

    def test_selection_ranks_decisive_evidenced_concepts_on_train(self):
        names = concepts.candidate_names(TAXONOMY)
        def output(decisive=(), present=()):
            return [label(n, .5 if n in present else 0., n in present, n in decisive) for n in names]
        labels = [output(decisive=("roster_changes", "rest_advantage"), present=("roster_changes", "rest_advantage", "head_to_head")),
                  output(decisive=("roster_changes",), present=("roster_changes", "head_to_head")),
                  output(decisive=("free_throw_edge",), present=())]  # decisive without evidence does not count
        result = concepts.select_concepts(labels, TAXONOMY, 3)
        self.assertEqual(result["selected"], ["roster_changes", "rest_advantage", "head_to_head"])
        self.assertEqual(result["selection"]["method"], "train_decisive_frequency")
        self.assertEqual(concepts.pool_sha256(result), concepts.pool_sha256(TAXONOMY))
        with self.assertRaises(ValueError):
            concepts.select_concepts([output()[:-1]], TAXONOMY, 3)


class ConceptLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row, _, cls.gold = contract_fixture.fixture_case()

    def test_request_reuses_frozen_gold_context_only(self):
        request = concepts.build_concept_request(self.gold["teacher_request"], TAXONOMY)
        payload = json.loads(request["messages"][1]["content"])
        original = json.loads(self.gold["teacher_request"]["messages"][1]["content"])
        for key in concepts.CONTEXT_KEYS:
            self.assertEqual(payload[key], original[key])
        self.assertEqual(request["parent_hashes"], [self.gold["teacher_request"]["request_id"]])
        self.assertEqual(request["split"], "train")
        self.assertEqual(request["request_id"], research.value_hash({k: v for k, v in request.items() if k != "request_id"}))
        self.assertEqual([c["name"] for c in payload["concept_definitions"]], concepts.candidate_names(TAXONOMY))
        non_gold = dict(self.gold["teacher_request"], stage="postmortem")
        non_gold["request_id"] = research.value_hash({k: v for k, v in non_gold.items() if k != "request_id"})
        with self.assertRaises(ValueError):
            concepts.build_concept_request(non_gold, TAXONOMY)

    def test_valid_labels_become_selected_order_targets(self):
        record = concept_record(self.row, self.gold)
        self.assertEqual(record["candidate"]["status"], "mechanically_valid")
        scores, present = concepts.concept_targets(record["candidate"]["output"], TAXONOMY)
        selected = TAXONOMY["selected"]
        self.assertEqual(len(scores), 20)
        self.assertEqual(scores[selected.index("star_availability")], .6)
        self.assertEqual(present[selected.index("rest_advantage")], 1.)
        self.assertEqual(sum(present), 2.)
        self.assertIs(contract.validate_concept_record(record, self.row, self.gold, TAXONOMY), record)

    def test_malformed_or_ungrounded_labels_are_rejected(self):
        names = concepts.candidate_names(TAXONOMY)
        bad_citation = label("recent_form", .4, True)
        bad_citation["citations"] = [{"source_id": "s1", "page": 1, "lines": [999]}]
        for raw, finish in ((response({"recent_form": label("recent_form", 1.5, True)}), "stop"),
                            (response({"recent_form": dict(label("recent_form"), score=.3)}), "stop"),
                            (response({"recent_form": bad_citation}), "stop"),
                            ({"concepts": response()["concepts"][:-1]}, "stop"),
                            ({"concepts": response()["concepts"] + [label(names[0])]}, "stop"),
                            (response(), "length")):
            request = concepts.build_concept_request(self.gold["teacher_request"], TAXONOMY)
            artifact = concepts.ingest_concept_response(request, raw, contract._game(self.row), "bedrock:teacher", TAXONOMY,
                                                        finish_reason=finish)
            with self.subTest(finish=finish, raw=str(raw)[:80]):
                self.assertEqual(artifact["status"], "rejected")
                self.assertNotIn("output", artifact)

    def test_contract_rejects_changed_or_mismatched_labels(self):
        record = concept_record(self.row, self.gold)
        tampered = copy.deepcopy(record)
        tampered["candidate"]["output"][0]["score"] = .9
        tampered["candidate"]["artifact_hash"] = research.value_hash({k: v for k, v in tampered["candidate"].items() if k != "artifact_hash"})
        early = concept_record(self.row, self.gold, created="2026-11-08T00:00:30Z")
        other_pool = copy.deepcopy(TAXONOMY)
        other_pool["candidates"][0]["description"] = "Changed definition."
        for value, taxonomy in ((tampered, TAXONOMY), (early, TAXONOMY), (record, other_pool)):
            with self.subTest(), self.assertRaises(ValueError):
                contract.validate_concept_record(value, self.row, self.gold, taxonomy)

    def test_api_boundary_accepts_concept_stage_only_for_authorized_train_games(self):
        request = concepts.build_concept_request(self.gold["teacher_request"], TAXONOMY)
        game = contract._game(self.row)
        registry = {"games": {self.row["game_id"]: {"game_id": self.row["game_id"], "split": "train",
                                                    "game_date_et": self.row["game_date_et"], "cutoff_utc": self.row["cutoff_utc"]}},
                    "trusted_source_hashes": {self.row["game_id"]: research.value_hash(game["sources"])},
                    "authorized_training_requests": {request["request_id"]: {
                        key: request[key] for key in ("request_id", "game_id", "stage", "source_hash")} | {
                        "request_sha256": research.value_hash(request)}}}
        api.validate_training_request(request, registry)
        stripped = copy.deepcopy(request)
        payload = json.loads(stripped["messages"][1]["content"]); payload.pop("training_only_postmortem")
        stripped["messages"][1]["content"] = json.dumps(payload)
        stripped["request_id"] = research.value_hash({k: v for k, v in stripped.items() if k != "request_id"})
        registry["authorized_training_requests"][stripped["request_id"]] = {
            key: stripped[key] for key in ("request_id", "game_id", "stage", "source_hash")} | {"request_sha256": research.value_hash(stripped)}
        with self.assertRaises(ValueError):
            api.validate_training_request(stripped, registry)


class SchemaTwoBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest, cls.rows, cls.outcomes, cls.gold = contract_fixture.fixture_bundle()
        cls.records = [concept_record(row, gold) for row, gold in zip(cls.rows, cls.gold)]
        cls.manifest = dict(manifest, schema_version=2, concept_pool_sha256=concepts.pool_sha256(TAXONOMY))
        cls.manifest["artifacts"] = dict(manifest["artifacts"], **{contract.CONCEPT_FILE: contract.sha256(contract_fixture.encode_rows(cls.records))})

    def write(self, root, records=None, manifest=None):
        manifest = manifest or self.manifest
        digest = contract_fixture.write_fixture_bundle(root, (manifest, self.rows, self.outcomes, self.gold))
        (Path(root) / contract.CONCEPT_FILE).write_bytes(contract_fixture.encode_rows(records or self.records))
        return digest

    def test_schema_two_bundle_loads_labels_for_every_pilot_game(self):
        with tempfile.TemporaryDirectory() as directory:
            digest = self.write(directory)
            manifest, rows, _, gold = contract.load_bundle(directory, digest)
            labels = contract.load_concepts(directory, manifest, rows, gold, TAXONOMY)
            self.assertEqual(set(labels), {r["game_id"] for r in rows})

    def test_changed_missing_or_wrong_pool_labels_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            digest = self.write(directory, records=self.records[:-1])
            manifest, rows, _, gold = contract.load_bundle(directory, digest)
            with self.assertRaisesRegex(ValueError, "changed"):
                contract.load_concepts(directory, manifest, rows, gold, TAXONOMY)
        with tempfile.TemporaryDirectory() as directory:
            partial = dict(self.manifest, artifacts=dict(self.manifest["artifacts"], **{
                contract.CONCEPT_FILE: contract.sha256(contract_fixture.encode_rows(self.records[:-1]))}))
            digest = self.write(directory, records=self.records[:-1], manifest=partial)
            manifest, rows, _, gold = contract.load_bundle(directory, digest)
            with self.assertRaisesRegex(ValueError, "Every pilot game"):
                contract.load_concepts(directory, manifest, rows, gold, TAXONOMY)
        with tempfile.TemporaryDirectory() as directory:
            digest = self.write(directory)
            manifest, rows, _, gold = contract.load_bundle(directory, digest)
            other = copy.deepcopy(TAXONOMY); other["scale"] = "Different scale."
            with self.assertRaisesRegex(ValueError, "pool"):
                contract.load_concepts(directory, manifest, rows, gold, other)
            with self.assertRaisesRegex(ValueError, "schema-2"):
                contract.load_concepts(directory, dict(manifest, schema_version=1), rows, gold, TAXONOMY)

    def test_schema_one_manifests_reject_concept_fields(self):
        with self.assertRaises(ValueError):
            contract.validate_manifest(dict(self.manifest, schema_version=1))
        base = {k: v for k, v in self.manifest.items() if k != "concept_pool_sha256"}
        with self.assertRaises(ValueError):
            contract.validate_manifest(dict(base, schema_version=2))


@unittest.skipIf(torch is None, "torch is installed with the training/tabpfn dependency groups")
class ConceptHeadTrainingTests(unittest.TestCase):
    def setUp(self):
        import bam_train_only as worker
        import test_bam_train_only as fixture
        self.worker, self.fixture = worker, fixture
        torch.set_num_threads(1)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(worker.common, "_adapter_state", side_effect=lambda m: {n: p for n, p in m.named_parameters()}))
        self.stack.enter_context(patch.object(worker.common, "_apply_adapter", side_effect=lambda m, state: m.load_state_dict(state, strict=False)))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.addCleanup(self.stack.close)
        self.rows, self.labels, self.gold = fixture.fixture()
        self.tokenizer = fixture.Tokenizer()
        self.targets = {"0": ([.8, -.4, 0.], [1., 1., 0.]), "1": ([-.6, .5, 0.], [1., 1., 0.]), "2": ([.7, 0., .3], [1., 0., 1.])}

    def config(self, **overrides):
        config = self.fixture.config(concept_weight=.5, minimum_relative_concept_reduction=.05, **overrides)
        config["forecast"]["head"] = {"type": "concept_bottleneck", "taxonomy_path": "config/bam_concepts.json",
                                      "taxonomy_sha256": concepts.taxonomy_sha256(TAXONOMY), "mlp_hidden": [8, 4], "dropout": .1}
        return config

    def make(self, seed=42):
        model, _ = self.fixture.make_model(seed)
        torch.manual_seed(seed)
        return model, concepts.build_head(6, 3, mlp_hidden=[8, 4], dropout=.1)

    def run_core(self, directory, model, head, **kwargs):
        return self.worker.train_steps(directory, self.config(), self.rows, self.labels, self.gold, model, self.tokenizer, head,
                                       manifest_sha256="b" * 64, concepts_by_game=self.targets, **kwargs)

    def test_zero_residual_reproduces_boosting_and_all_three_objectives_train(self):
        with tempfile.TemporaryDirectory() as directory:
            model, head = self.make()
            self.assertTrue(concepts.residual_is_zero(head))
            result = self.run_core(directory, model, head)
            curve = json.loads((Path(directory) / "learning_curve_train.json").read_text())
            self.assertTrue(result["baseline"]["zero_head_equivalence_verified"])
            self.assertEqual(result["head_type"], "concept_bottleneck")
            first, later = curve["steps"][0]["optimization"], curve["steps"][-1]["optimization"]
            self.assertEqual(first["brier_probe_adapter_gradient_norm"], 0.)  # zero final layer blocks Brier at step 1
            self.assertGreater(first["prefix_adapter_gradient_norm"], 0.)     # concept loss already reaches the LLM
            self.assertGreater(later["brier_probe_adapter_gradient_norm"], 0.)
            self.assertTrue(result["brier_gradient_reached_adapters_after_zero_head"])
            self.assertFalse(concepts.residual_is_zero(head))
            last = result["last_inference_train"]
            self.assertLess(last["brier"], result["baseline"]["brier"])
            self.assertLess(last["concept_loss"], result["baseline"]["concept_loss"])
            self.assertIn("relative_concept_reduction", result)
            self.assertFalse(result["generalization_claim"])

    def test_concept_loss_masks_scores_without_evidence(self):
        scores = torch.tensor([.5, .9]); logits = torch.tensor([0., 0.])
        total, mse, bce = concepts.concept_loss(scores, logits, [.5, -1.], [1., 0.])
        self.assertAlmostEqual(mse.item(), 0.)
        self.assertAlmostEqual(bce.item(), torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.])).item())
        self.assertAlmostEqual(total.item(), mse.item() + bce.item())

    def test_resume_reproduces_concept_training_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, head = self.make()
            complete = self.run_core(root / "complete", model, head)
            state = copy.deepcopy(self.worker.common._joint_state(model, head))
            model, head = self.make()
            with self.assertRaisesRegex(RuntimeError, "checkpointed"):
                self.run_core(root / "resume", model, head, stop_after_steps=2)
            model, head = self.make(88)
            self.assertEqual(complete, self.run_core(root / "resume", model, head))
            for name, value in state.items():
                self.assertTrue(torch.equal(value, self.worker.common._joint_state(model, head)[name]), name)

    def test_head_type_config_and_targets_must_agree(self):
        model, head = self.make()
        linear = torch.nn.Linear(6, 1); torch.nn.init.zeros_(linear.weight); torch.nn.init.zeros_(linear.bias)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "head"):
                self.worker.train_steps(directory, self.config(), self.rows, self.labels, self.gold, model, self.tokenizer, linear,
                                        manifest_sha256="b" * 64, concepts_by_game=self.targets)
            with self.assertRaisesRegex(ValueError, "head"):
                self.worker.train_steps(directory, self.fixture.config(), self.rows, self.labels, self.gold, model, self.tokenizer, head,
                                        manifest_sha256="b" * 64)
            wide = {k: (v[0] + [0.], v[1] + [0.]) for k, v in self.targets.items()}
            with self.assertRaisesRegex(ValueError, "width"):
                self.worker.train_steps(directory, self.config(), self.rows, self.labels, self.gold, model, self.tokenizer, head,
                                        manifest_sha256="b" * 64, concepts_by_game=wide)
        with self.assertRaisesRegex(ValueError, "concept_weight"):
            self.worker.validate_config(self.fixture.config(minimum_relative_concept_reduction=.1) | {"forecast": self.config()["forecast"]})
        with self.assertRaisesRegex(ValueError, "require the concept_bottleneck"):
            self.worker.validate_config(self.fixture.config(concept_weight=1.))

    def test_smoke_runs_configured_head_on_synthetic_representative_prompts(self):
        from types import SimpleNamespace
        model, _ = self.fixture.make_model()
        model.config = SimpleNamespace(hidden_size=6)
        config = self.config()
        config["forecast"]["head"]["mlp_hidden"] = [8, 4]
        with tempfile.TemporaryDirectory() as directory, patch.object(
                self.worker.common, "_load_model", return_value=(model, self.tokenizer)):
            summary = self.worker.smoke(config, directory, games=3, steps=2, prompt_fraction=.5)
            self.assertTrue(summary["synthetic_only"])
            self.assertFalse(summary["nba_performance_claim"])
            self.assertEqual(summary["head_type"], "concept_bottleneck")
            self.assertTrue(summary["zero_head_equivalence_verified"])
            # Character-level toy tokenizer: sizing stays within budget (real tokenizers fill it closely).
            self.assertTrue(all(0 < n <= 128 for n in summary["prompt_tokens"]), summary["prompt_tokens"])
            self.assertTrue((Path(directory) / "smoke/smoke_result.json").exists())
        with self.assertRaisesRegex(ValueError, "bounded"):
            self.worker.smoke(config, "unused", games=20)

    def test_pilot_config_validates(self):
        config = json.loads((ROOT / "config/bam_train_only_concept_pilot.json").read_text())
        options = self.worker.validate_config(config)
        self.assertEqual(options["concept_weight"], 1.0)


@unittest.skipIf(torch is None, "torch is installed with the training/tabpfn dependency groups")
class DevLoopConceptTests(unittest.TestCase):
    """bam_training's TRAIN-fit / DEV-select forecaster loop with the concept head."""

    def setUp(self):
        import test_bam_training as fixture
        # Use the fixture's own module copy: its setUp patches that object.
        self.bam, self.fixture = fixture.bam, fixture
        fixture.OptimizerResumeIntegration.setUp(self)  # tiny model, toy tokenizer, adapter/prefix patches
        self.config = copy.deepcopy(self.config)
        self.config["forecast"]["concept_loss_weight"] = .5
        self.config["forecast"]["head"] = {"type": "concept_bottleneck", "taxonomy_path": "config/bam_concepts.json",
                                           "taxonomy_sha256": concepts.taxonomy_sha256(TAXONOMY), "mlp_hidden": [8, 4], "dropout": .1}

    def new(self, seed=23):
        torch.manual_seed(seed)
        model = self.Model()
        return model, self.bam._build_forecast_head(self.config, 4, "cpu")

    def labels_for(self, game_id, score):
        names = concepts.candidate_names(TAXONOMY)
        items = [label(n) for n in names]
        items[names.index("star_availability")] = label("star_availability", score, True, True)
        for item in items:
            item["score"] = float(item["score"])
        return {"game_id": game_id, "split": "train", "concept_pool_sha256": concepts.pool_sha256(TAXONOMY), "labels": items}

    def write_fixture(self, root, labels=None):
        rows = [self.fixture.forecast(str(i), "2024-01-01", "train", p=.2) for i in range(3)] + [
            self.fixture.forecast("d", "2024-02-01", "dev", p=.2)]
        self.bam.write_rows(root / "forecast_explanations_train.jsonl", [
            {"game_id": str(i), "review_status": "accepted", "explanation": "Synthetic source summary."} for i in range(3)])
        self.bam.write_rows(root / "forecast_concepts_train.jsonl",
                            labels if labels is not None else [self.labels_for(str(i), .5 - .4 * i) for i in range(3)])
        return rows, {str(i): 1 for i in range(3)}, {"d": 1}

    def test_concept_head_trains_selects_on_dev_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            full, partial = Path(directory) / "full", Path(directory) / "resume"
            rows, labels, dev = self.write_fixture(full); self.write_fixture(partial)
            model, head = self.new()
            self.assertTrue(concepts.residual_is_zero(head))
            complete = self.bam._forecast_steps(full, self.config, rows, labels, dev, model, self.tokenizer, head)
            self.assertEqual(complete["head_type"], "concept_bottleneck")
            self.assertIn("concept loss", complete["loss"])
            self.assertEqual(complete["candidates"][0]["name"], "zero_correction")
            self.assertFalse(concepts.residual_is_zero(head))
            expected = copy.deepcopy(self.bam._joint_state(model, head))
            model, head = self.new()
            with self.assertRaisesRegex(RuntimeError, "checkpointed"):
                self.bam._forecast_steps(partial, self.config, rows, labels, dev, model, self.tokenizer, head, stop_after_steps=3)
            model, head = self.new(99)
            self.assertEqual(complete, self.bam._forecast_steps(partial, self.config, rows, labels, dev, model, self.tokenizer, head))
            for key, value in expected.items():
                self.assertTrue(torch.equal(value, self.bam._joint_state(model, head)[key]), key)

    def test_concept_labels_are_train_only_and_pool_bound(self):
        dev_label = self.labels_for("d", .3)
        wrong_pool = dict(self.labels_for("0", .3), concept_pool_sha256="0" * 64)
        unordered = self.labels_for("0", .3); unordered["labels"] = list(reversed(unordered["labels"]))
        for bad in ([dev_label], [wrong_pool], [unordered], [self.labels_for("0", .3)] * 2, []):
            with tempfile.TemporaryDirectory() as directory, self.subTest(bad=str(bad)[:60]):
                rows, labels, dev = self.write_fixture(Path(directory), labels=bad)
                model, head = self.new()
                with self.assertRaises(ValueError):
                    self.bam._forecast_steps(Path(directory), self.config, rows, labels, dev, model, self.tokenizer, head)

    def test_head_type_and_concept_weight_must_match_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows, labels, dev = self.write_fixture(root)
            model, head = self.new()
            linear = torch.nn.Linear(4, 1); torch.nn.init.zeros_(linear.weight); torch.nn.init.zeros_(linear.bias)
            with self.assertRaisesRegex(ValueError, "head"):
                self.bam._forecast_steps(root, self.config, rows, labels, dev, model, self.tokenizer, linear)
            linear_config = copy.deepcopy(self.config); linear_config["forecast"].pop("head")
            with self.assertRaisesRegex(ValueError, "concept_loss_weight"):
                self.bam._forecast_steps(root, linear_config, rows, labels, dev, model, self.tokenizer, linear)
            nonzero = self.new()[1]; torch.nn.init.ones_(nonzero.residual.weight)
            other = root / "other"; self.write_fixture(other)
            with self.assertRaisesRegex(ValueError, "exactly zero"):
                self.bam._forecast_steps(other, self.config, rows, labels, dev, model, self.tokenizer, nonzero)

    def test_export_from_bundle_records_feeds_the_dev_pipeline(self):
        row, _, gold = contract_fixture.fixture_case()
        record = concept_record(row, gold)
        exported = concepts.forecast_label_rows([record], TAXONOMY)
        self.assertEqual(set(exported[0]), {"game_id", "split", "concept_pool_sha256", "labels"})
        self.assertEqual(concepts.validate_normalized_labels(exported[0]["labels"], TAXONOMY), record["candidate"]["output"])


if __name__ == "__main__":
    unittest.main()
