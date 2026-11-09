"""Synthetic accounting/launch-scope checks; no AWS or project dataset calls."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bam_aws_preflight import evaluate_preflight, ledger_available_cents

NOW = datetime(2026, 11, 8, 5, 0, tzinfo=timezone.utc)
CHECKED = "2026-11-08T04:00:00Z"
ACCOUNT = "123456789012"


def fixture(region="us-west-2"):
    """Entirely invented account evidence; NEVER suitable for a real launch."""
    base = {"checked_at": CHECKED, "reference": "synthetic-unit-test-evidence"}
    roles = {}
    for kind in ("controller", "gpu"):
        roles[kind] = {"role_arn": f"arn:aws:iam::{ACCOUNT}:role/cs230/{kind}",
                       "instance_profile_arn": f"arn:aws:iam::{ACCOUNT}:instance-profile/cs230/{kind}",
                       "train_only_policy_verified": True, "no_static_credentials": True,
                       "trust_service": "ec2.amazonaws.com", "policy_sha256": "a" * 64}
    models = {}
    for kind in ("teacher", "reviewer"):
        models[kind] = {**base, "region": region, "model_id": kind + "-model",
                        "inference_profile_id": "us." + kind + "-model",
                        "agreementAvailability": "AVAILABLE", "authorizationStatus": "AUTHORIZED",
                        "entitlementAvailability": "AVAILABLE", "regionAvailability": "AVAILABLE"}
    evidence = {"schema_version": 1, "account_id": ACCOUNT,
                "credits": {**base, "balance_kind": "earned_remaining", "currency": "USD",
                            "remaining_usd": "100", "expires_at": "2027-05-01T00:00:00Z",
                            "eligibility_reference": "synthetic-credit-details",
                            "eligible_services": ["EC2", "Bedrock-Anthropic", "S3", "Transfer", "Logs"],
                            "accrued_unbilled_usd": "0", "accrued_unbilled_reference": "synthetic-billing"},
                "execution": {**base, "region": region, "status": "deployed", **roles},
                "gpu_quota": {**base, "region": region, "quota_code": "L-DB2E81BA", "value": 8},
                "models": models}

    def quote(resource, service, **rates):
        return {**base, "resource_id": resource, "service": service,
                "region": region, "currency": "USD", **rates}

    plan = {"schema_version": 1, "account_id": ACCOUNT, "platform": "aws", "region": region,
            "job_id": "synthetic-test-only", "manifest_sha256": "b" * 64, "config_sha256": "c" * 64,
            "model": {"model_id": "Qwen/Qwen2.5-14B-Instruct", "model_revision": "d" * 40},
            "phase": "train_pilot", "splits": ["TRAIN"], "game_count": 16,
            "cash_spend_authorized_usd": 0, "planned_start_at": "2026-11-08T06:00:00Z",
            "max_total_runtime_seconds": 7200, "max_cost_usd": "40", "compute": {},
            "model_calls": {}, "overheads": {}}
    for kind, instance, rate in (("controller", "t3.small", "0.02"), ("gpu", "g6e.2xlarge", "2.5")):
        plan["compute"][kind] = {"instance_type": instance, "count": 1, "max_runtime_seconds": 3600,
                                 "terminate_at_deadline": True,
                                 "quote": quote(instance, "EC2", hourly_usd=rate)}
    for kind in ("teacher", "reviewer"):
        plan["model_calls"][kind] = {"model_id": kind + "-model", "inference_profile_id": "us." + kind + "-model",
                                    "max_requests": 16, "max_attempts_per_request": 2,
                                    "max_input_tokens": 2000, "max_output_tokens": 1000,
                                    "reasoning_in_output_budget": True,
                                    "input_count_basis": "synthetic-full-input-bound",
                                    "quote": quote("us." + kind + "-model", "Bedrock-Anthropic",
                                                   input_usd_per_million="3", output_usd_per_million="15")}
    for kind, service, amount in (("storage", "S3", "1"), ("transfer", "Transfer", "1"),
                                  ("logging", "Logs", "1"), ("cleanup", "EC2", "1"), ("tax", "Tax", "0")):
        plan["overheads"][kind] = {"max_cost_usd": amount, "bound_basis": "synthetic-bound-including-tail",
                                    "quote": quote(kind, service)}
    ledger = {"schema_version": 1, "cap_cents": 20000, "buffer_cents": 500,
              "reservation_multiplier": "1.15", "calls": {
                  "completed": {"call_key": "completed", "state": "completed", "reservation_cents": 800,
                                "estimated_cost_cents": 677},
                  "uncertain": {"call_key": "uncertain", "state": "uncertain", "reservation_cents": 182}},
              "events": [{"action": "original-history"}]}
    return evidence, plan, ledger


class PreflightTests(unittest.TestCase):
    def evaluate(self, evidence=None, plan=None, ledger=None):
        defaults = fixture()
        return evaluate_preflight(evidence if evidence is not None else defaults[0],
                                  plan if plan is not None else defaults[1],
                                  ledger if ledger is not None else defaults[2], now=NOW)

    def test_complete_synthetic_evidence_passes_but_never_reserves_or_authorizes_launch(self):
        evidence, plan, ledger = fixture()
        before = deepcopy((evidence, plan, ledger))
        result = self.evaluate(evidence, plan, ledger)
        self.assertTrue(result["prerequisites_satisfied"], result["blockers"])
        self.assertEqual(result["financials"]["bounded_reservation_cents"], 905)
        self.assertEqual(result["financials"]["ledger_available_cents"], 18641)
        self.assertEqual(result["financials"]["verified_credit_available_cents"], 9318)
        self.assertFalse(result["reservation_created"])
        self.assertFalse(result["launch_authorized"])
        self.assertEqual(result["binding"]["job_id"], plan["job_id"])
        self.assertEqual(result["binding"]["region"], plan["region"])
        self.assertEqual(len(result["plan_content_sha256"]), 64)
        self.assertEqual((evidence, plan, ledger), before)

    def test_explicit_other_region_passes_only_with_matching_regional_evidence(self):
        for region in ("us-east-1", "eu-west-1", "ap-southeast-2"):
            with self.subTest(region=region):
                evidence, plan, ledger = fixture(region)
                result = self.evaluate(evidence, plan, ledger)
                self.assertTrue(result["prerequisites_satisfied"], result["blockers"])
                self.assertEqual(result["binding"]["region"], region)
                self.assertFalse(result["launch_authorized"])

    def test_each_regional_evidence_record_must_match_the_plan(self):
        paths = [("execution",), ("gpu_quota",), ("models", "teacher"), ("models", "reviewer")]
        for path in paths:
            with self.subTest(path=path):
                evidence, plan, ledger = fixture("us-east-1")
                record = evidence
                for key in path:
                    record = record[key]
                record["region"] = "us-west-2"
                result = self.evaluate(evidence, plan, ledger)
                self.assertFalse(result["prerequisites_satisfied"])
                self.assertIn("plan region", "\n".join(result["blockers"]))
        for group, keys in (("compute", ("controller", "gpu")), ("model_calls", ("teacher", "reviewer")),
                            ("overheads", ("storage", "transfer", "logging", "cleanup", "tax"))):
            for key in keys:
                with self.subTest(quote=(group, key)):
                    evidence, plan, ledger = fixture("us-east-1")
                    plan[group][key]["quote"]["region"] = "us-west-2"
                    result = self.evaluate(evidence, plan, ledger)
                    self.assertFalse(result["prerequisites_satisfied"])
                    self.assertFalse(result["financials"]["cost_bounds_complete"])

    def test_missing_malformed_and_other_partition_regions_fail_closed(self):
        for region in (None, "", [], "US-EAST-1", "us-east-1 ", "us-gov-west-1", "cn-north-1",
                       "us-iso-east-1", "moon-east-1", "us-east-0"):
            with self.subTest(region=region):
                evidence, plan, ledger = fixture(region)
                result = self.evaluate(evidence, plan, ledger)
                self.assertFalse(result["prerequisites_satisfied"])
                self.assertTrue(any(x.startswith("scope:") for x in result["blockers"]))

    def test_current_account_scenario_exposes_each_independent_blocker(self):
        evidence, plan, ledger = fixture()
        evidence["credits"].pop("eligibility_reference")
        evidence["credits"]["potential_activity_rewards_usd"] = 100
        evidence["execution"]["status"] = "draft"
        evidence["gpu_quota"]["value"] = 0
        evidence["gpu_quota"]["pending_requested_value"] = 8
        evidence["models"]["teacher"]["agreementAvailability"] = "NOT_AVAILABLE"
        result = self.evaluate(evidence, plan, ledger)
        self.assertFalse(result["prerequisites_satisfied"])
        for name in ("credits:", "execution:", "gpu_quota:", "teacher:"):
            self.assertTrue(any(x.startswith(name) for x in result["blockers"]), result)

    def test_project_cap_and_unearned_rewards_cannot_cover_credit_shortfall(self):
        evidence, plan, _ = fixture()
        evidence["credits"]["remaining_usd"] = "10"
        evidence["credits"]["potential_activity_rewards_usd"] = 100
        result = self.evaluate(evidence, plan)
        self.assertGreater(result["financials"]["ledger_available_cents"], 900)
        self.assertIn("project cap does not authorize cash", "\n".join(result["blockers"]))
        evidence["credits"]["balance_kind"] = "potential_total"
        self.assertFalse(self.evaluate(evidence, plan)["prerequisites_satisfied"])

    def test_unknown_service_eligibility_and_taxes_fail(self):
        evidence, plan, _ = fixture()
        evidence["credits"]["eligible_services"].remove("Bedrock-Anthropic")
        self.assertIn("Bedrock-Anthropic", "\n".join(self.evaluate(evidence, plan)["blockers"]))
        evidence = fixture()[0]
        plan["overheads"]["tax"]["max_cost_usd"] = "1"
        self.assertIn("Tax", "\n".join(self.evaluate(evidence, plan)["blockers"]))

    def test_credit_expiry_must_cover_cleanup_and_unbilled_costs_are_retained(self):
        evidence, plan, _ = fixture()
        evidence["credits"]["expires_at"] = "2026-11-08T07:30:00Z"
        self.assertIn("runtime and cleanup", "\n".join(self.evaluate(evidence, plan)["blockers"]))
        evidence = fixture()[0]
        evidence["credits"]["accrued_unbilled_usd"] = "90"
        self.assertFalse(self.evaluate(evidence, plan)["prerequisites_satisfied"])

    def test_stale_future_and_naive_timestamps_fail(self):
        for value in ("2026-11-07T04:00:00Z", "2026-11-08T06:00:00Z", "2026-11-08T04:00:00"):
            evidence = fixture()[0]
            evidence["credits"]["checked_at"] = value
            self.assertFalse(self.evaluate(evidence=evidence)["prerequisites_satisfied"])

    def test_heldout_scope_and_root_or_wrong_account_roles_fail(self):
        for split in (["DEV"], ["TRAIN", "TEST"], ["train"]):
            plan = fixture()[1]
            plan["splits"] = split
            self.assertFalse(self.evaluate(plan=plan)["prerequisites_satisfied"])
        for arn in (f"arn:aws:iam::{ACCOUNT}:root", "arn:aws:iam::999999999999:role/cs230/gpu"):
            evidence = fixture()[0]
            evidence["execution"]["gpu"]["role_arn"] = arn
            self.assertFalse(self.evaluate(evidence=evidence)["prerequisites_satisfied"])

    def test_cost_bound_covers_every_report_retry_and_reasoning(self):
        original = self.evaluate()["financials"]["bounded_reservation_cents"]
        for field, value in (("max_requests", 0), ("max_requests", 15), ("max_attempts_per_request", 0),
                             ("reasoning_in_output_budget", False), ("input_count_basis", "")):
            plan = fixture()[1]
            plan["model_calls"]["teacher"][field] = value
            result = self.evaluate(plan=plan)
            self.assertFalse(result["prerequisites_satisfied"])
            self.assertFalse(result["financials"]["cost_bounds_complete"])
        plan = fixture()[1]
        plan["model_calls"]["teacher"]["max_attempts_per_request"] = 3
        self.assertGreater(self.evaluate(plan=plan)["financials"]["bounded_reservation_cents"], original)

    def test_missing_storage_or_unbounded_stop_plan_fails(self):
        plan = fixture()[1]
        del plan["overheads"]["storage"]
        plan["compute"]["gpu"]["terminate_at_deadline"] = False
        result = self.evaluate(plan=plan)
        self.assertFalse(result["prerequisites_satisfied"])
        self.assertTrue(any(x.startswith("storage_cost:") for x in result["blockers"]))
        self.assertTrue(any(x.startswith("gpu_cost:") for x in result["blockers"]))

    def test_ledger_totals_are_recomputed_and_uncertain_reserves_not_released(self):
        ledger = fixture()[2]
        ledger["totals"] = {"available_for_new_calls_cents": 999999}
        ledger["calls"]["uncertain"]["reservation_cents"] = 19000
        self.assertEqual(ledger_available_cents(ledger), -177)
        result = self.evaluate(ledger=ledger)
        self.assertFalse(result["prerequisites_satisfied"])
        self.assertEqual(ledger["calls"]["uncertain"]["state"], "uncertain")

    def test_wrong_ledger_cap_or_incomplete_record_fails_closed(self):
        for mutate in (lambda x: x.update(cap_cents=15000),
                       lambda x: x["calls"]["completed"].pop("estimated_cost_cents"),
                       lambda x: x["calls"]["uncertain"].update(reservation_cents=True)):
            ledger = fixture()[2]
            mutate(ledger)
            self.assertFalse(self.evaluate(ledger=ledger)["prerequisites_satisfied"])

    def test_malformed_numbers_do_not_turn_into_free_work(self):
        for amount in (True, "NaN", "Infinity", "-1", None):
            plan = fixture()[1]
            plan["overheads"]["storage"]["max_cost_usd"] = amount
            self.assertFalse(self.evaluate(plan=plan)["prerequisites_satisfied"])

    def test_same_model_cannot_review_its_own_gold(self):
        evidence = fixture()[0]
        evidence["models"]["reviewer"] = deepcopy(evidence["models"]["teacher"])
        self.assertIn("different models", "\n".join(self.evaluate(evidence=evidence)["blockers"]))

    def test_cli_reads_original_ledger_and_does_not_write_even_on_blocked_run(self):
        evidence, plan, ledger = fixture()
        # The CLI uses the actual time; give it unambiguously stale evidence.
        evidence["credits"]["checked_at"] = "2000-01-01T00:00:00Z"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ledger.json"
            path.write_text(json.dumps(ledger))
            Path(str(path) + ".lock").touch()
            config = {"cap_usd": 200, "compute_platform": "aws", "ledger_path": str(path)}
            for name, obj in (("evidence", evidence), ("plan", plan), ("config", config)):
                (root / (name + ".json")).write_text(json.dumps(obj))
            before = path.read_bytes()
            script = Path(__file__).resolve().parents[1] / "scripts/bam_aws_preflight.py"
            run = subprocess.run([sys.executable, str(script), "--evidence", str(root / "evidence.json"),
                                  "--plan", str(root / "plan.json"), "--budget-config", str(root / "config.json")],
                                 capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 2, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report["ledger_path"], str(path.resolve()))
            self.assertFalse(report["ledger_mutated"])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
