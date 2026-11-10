"""Launcher: termination armed first, role-scoped S3 paths, and receipts the TRAIN worker accepts."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bam_aws_launch as launch
from bam_training import value_hash

ROOT = Path(__file__).resolve().parents[1]


def config():
    value = launch.load_config()
    value["bucket"] = "cs230-artifacts-example"
    return value


class UserDataTests(unittest.TestCase):
    def test_shutdown_is_armed_before_any_work_and_instances_terminate(self):
        for job, extra in (("smoke", {}), ("research", {"round_name": "000-researcher"}), ("train", {"manifest_sha256": "a" * 64})):
            script = launch.user_data(config(), job, code_key="abc123.tar.gz", run_id="pilot-1", **extra)
            lines = script.splitlines()
            self.assertEqual(lines[:2], ["#!/bin/bash", "set -euo pipefail"])
            self.assertTrue(lines[2].startswith(f"shutdown -h +{config()['jobs'][job]['max_minutes']}"), job)
            self.assertIn("trap '", script)
            params = launch.run_instances_params(config(), job, script, ami_id="ami-1",
                                                 instance_profile_arn="arn:aws:iam::123456789012:instance-profile/cs230/p", run_id="pilot-1")
            self.assertEqual(params["InstanceInitiatedShutdownBehavior"], "terminate")
            self.assertEqual(params["MetadataOptions"]["HttpTokens"], "required")
            self.assertTrue(params["BlockDeviceMappings"][0]["Ebs"]["Encrypted"])
            self.assertNotIn("AWS_SECRET", script); self.assertNotIn("aws_access_key", script.lower())

    def test_jobs_only_touch_prefixes_their_iam_role_allows(self):
        prefix = "s3://cs230-artifacts-example/train/pilot-20261108/"
        research = launch.user_data(config(), "research", code_key="c.tar.gz", run_id="r1", round_name="001-critic")
        self.assertIn(prefix + "pregame/r1/rounds/001-critic/request.json", research)
        self.assertIn(prefix + "research/r1/rounds/001-critic/output", research)
        self.assertNotIn("targets/", research); self.assertNotIn("metrics/", research)
        self.assertIn("--group research", research); self.assertIn("--python 3.13", research)
        train = launch.user_data(config(), "train", code_key="c.tar.gz", run_id="r1", manifest_sha256="b" * 64)
        for part in ("targets/r1/bundle", "targets/r1/admission.json", "checkpoints/r1", "metrics/r1"):
            self.assertIn(prefix + part, train)
        self.assertNotIn("research/", train.replace("--group research", ""))
        self.assertIn("--manifest-sha256 " + "b" * 64, train)

    def test_unsafe_identifiers_and_missing_manifest_are_rejected(self):
        with self.assertRaises(ValueError):
            launch.user_data(config(), "smoke", code_key="c.tar.gz", run_id="x; rm -rf /")
        with self.assertRaises(ValueError):
            launch.user_data(config(), "train", code_key="c.tar.gz", run_id="r1")
        self.assertTrue(launch.readiness(launch.load_config()))  # repository config has no bucket yet
        self.assertEqual(launch.readiness(config()), [])


class AdmissionTests(unittest.TestCase):
    def test_receipt_passes_the_train_worker_admission_check(self):
        import bam_train_only as worker
        forecaster = json.loads((ROOT / launch.FORECASTER_CONFIG).read_text())
        binding = {"job_id": "pilot-1", "manifest_sha256": "a" * 64, "config_sha256": value_hash(forecaster),
                   "model": {k: forecaster[k] for k in ("model_id", "model_revision")}, "region": "us-west-2"}
        report = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(), "binding": binding,
                  "prerequisites_satisfied": True, "blockers": [], "reservation_created": False, "launch_authorized": False,
                  "financials": {"cost_bounds_complete": True, "bounded_reservation_cents": 500,
                                 "verified_credit_available_cents": 9500, "ledger_available_cents": 18000}}
        with tempfile.TemporaryDirectory() as directory:
            preflight = Path(directory) / "preflight.json"; preflight.write_text(json.dumps(report))
            receipt = launch.admission_receipt(preflight, forecaster, manifest_sha256="a" * 64, job_id="pilot-1",
                                               reservation_id="ledger-7", reserved_cents=600, region="us-west-2")
            admission = Path(directory) / "admission.json"; admission.write_text(json.dumps(receipt))
            self.assertEqual(worker._read_execution_admission(admission, preflight, "a" * 64, forecaster, "pilot-1"), receipt)
            with self.assertRaisesRegex(ValueError, "binding"):
                launch.admission_receipt(preflight, forecaster, manifest_sha256="c" * 64, job_id="pilot-1",
                                         reservation_id="ledger-7", reserved_cents=600, region="us-west-2")
            blocked = dict(report, prerequisites_satisfied=False, blockers=["gpu_quota: below 8 vCPUs"])
            preflight.write_text(json.dumps(blocked))
            with self.assertRaisesRegex(ValueError, "blockers"):
                launch.admission_receipt(preflight, forecaster, manifest_sha256="a" * 64, job_id="pilot-1",
                                         reservation_id="ledger-7", reserved_cents=600, region="us-west-2")
            stale = dict(report, checked_at=(datetime.now(timezone.utc) - timedelta(hours=30)).isoformat())
            preflight.write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "too old"):
                launch.admission_receipt(preflight, forecaster, manifest_sha256="a" * 64, job_id="pilot-1",
                                         reservation_id="ledger-7", reserved_cents=600, region="us-west-2")


if __name__ == "__main__":
    unittest.main()
