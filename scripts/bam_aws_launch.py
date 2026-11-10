"""Launch bounded GPU jobs for the concept pilot on EC2 (Oregon IAM stack cs230-train-pilot-20261108).

Jobs (config/bam_aws.json):
  smoke     forecaster role: synthetic concept-head run on the real model (bam_train_only --smoke)
  research  researcher role: one bam_pilot research round with vLLM (pregame/ in, research/ out)
  train     forecaster role: the admitted TRAIN pilot (targets/ in, checkpoints/ and metrics/ out)

Every instance arms `shutdown -h +max_minutes` before any setup and is launched with
InstanceInitiatedShutdownBehavior=terminate, so it terminates even if the job hangs.
Code comes from a committed `git archive` uploaded to code/; no credentials are placed on
the instance (it uses its /cs230/ instance profile).

Commands: check | upload-code | plan | admit | launch. `plan` and `admit` make no AWS calls;
`launch --dry-run` asks EC2 to check permissions without starting anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/bam_aws.json"
FORECASTER_CONFIG = "config/bam_train_only_concept_pilot.json"
STACK_PROFILES = {"researcher": "ResearchProfile", "forecaster": "TrainingProfile", "controller": "ControllerProfile"}
GROUPS = {"researcher": ("research", "3.13"), "forecaster": ("training", "3.12")}
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(config.get("schema_version") == 1 and set(config["jobs"]) == {"smoke", "research", "train"}, "Unknown AWS config")
    for job in config["jobs"].values():
        _require(job["role"] in GROUPS and 10 <= job["max_minutes"] <= 24 * 60, "Each job needs a GPU role and 10 min-24 h limit")
    return config


def readiness(config):
    blockers = []
    if not config.get("bucket"):
        blockers.append("bucket: create the stack's ArtifactBucketName bucket in Oregon and enter it in config/bam_aws.json")
    if config.get("region") != "us-west-2":
        blockers.append("region: the deployed IAM stack and its bucket restriction are in us-west-2")
    return blockers


def s3_uri(config, *parts):
    return f"s3://{config['bucket']}/{config['prefix']}/" + "/".join(parts)


def _safe(value, name):
    _require(isinstance(value, str) and SAFE.fullmatch(value), f"{name} must be a simple identifier")
    return value


def user_data(config, job, *, code_key, run_id, round_name=None, manifest_sha256=None):
    """Bash for the instance: termination armed first, then code, environment, job, upload."""
    spec = config["jobs"][job]
    group, python = GROUPS[spec["role"]]
    run_id = _safe(run_id, "run_id")
    code = s3_uri(config, "code", _safe(code_key, "code_key"))
    log_target = s3_uri(config, "research" if spec["role"] == "researcher" else "metrics", run_id, "logs", f"{job}.log")
    lines = ["#!/bin/bash", "set -euo pipefail",
             f"shutdown -h +{int(spec['max_minutes'])}  # armed before any work; instance terminates on shutdown",
             "exec > >(tee /var/log/bam-job.log) 2>&1",
             f"trap 'aws s3 cp /var/log/bam-job.log {shlex.quote(log_target)} || true; shutdown -h now' EXIT",
             "export HOME=/root PATH=/root/.local/bin:$PATH HF_HOME=/opt/hf",
             "mkdir -p /opt/repo /opt/out && cd /opt/repo",
             f"aws s3 cp {shlex.quote(code)} /opt/code.tar.gz && tar -xzf /opt/code.tar.gz -C /opt/repo",
             "curl -LsSf https://astral.sh/uv/install.sh | sh",
             f"uv venv --python {python} .venv && uv pip install --python .venv/bin/python --group {group}"]
    if job == "smoke":
        lines += [f".venv/bin/python scripts/bam_train_only.py --smoke --config {FORECASTER_CONFIG} --output /opt/out",
                  f"aws s3 cp --recursive /opt/out {shlex.quote(s3_uri(config, 'metrics', run_id, 'smoke'))}"]
    elif job == "research":
        round_name = _safe(round_name, "round")
        local = f"runs/{run_id}/research/rounds/{round_name}"
        lines += [f"mkdir -p {local}",
                  f"aws s3 cp {shlex.quote(s3_uri(config, 'pregame', run_id, 'rounds', round_name, 'request.json'))} {local}/request.json",
                  ".venv/bin/python - <<'PY'",
                  "import json; from huggingface_hub import snapshot_download",
                  f"model = json.load(open('{local}/request.json'))['model']",
                  "snapshot_download(model['model_id'], revision=model['model_revision'], local_dir='/opt/model')",
                  "PY",
                  f".venv/bin/python scripts/bam_pilot.py infer --run runs/{run_id} --round {local} --snapshot-path /opt/model",
                  f"aws s3 cp --recursive {local}/output {shlex.quote(s3_uri(config, 'research', run_id, 'rounds', round_name, 'output'))}"]
    elif job == "train":
        _require(isinstance(manifest_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", manifest_sha256), "Train job needs the bundle manifest SHA256")
        targets = s3_uri(config, "targets", run_id)
        lines += [f"aws s3 cp --recursive {shlex.quote(targets + '/bundle')} /opt/bundle",
                  f"aws s3 cp {shlex.quote(targets + '/admission.json')} /opt/admission.json",
                  f"aws s3 cp {shlex.quote(targets + '/preflight.json')} /opt/preflight.json",
                  f".venv/bin/python scripts/bam_train_only.py --config {FORECASTER_CONFIG} --bundle /opt/bundle "
                  f"--manifest-sha256 {manifest_sha256} --output /opt/out --admission /opt/admission.json "
                  f"--preflight-report /opt/preflight.json --job-id {run_id}",
                  f"aws s3 cp --recursive /opt/out {shlex.quote(s3_uri(config, 'checkpoints', run_id))}",
                  "for f in result_train.json learning_curve_train.json baseline_train.json model_files.json head_config.json; do "
                  f"aws s3 cp /opt/out/$f {shlex.quote(s3_uri(config, 'metrics', run_id))}/$f; done"]
    else:
        raise ValueError("Unknown job")
    return "\n".join(lines) + "\n"


def run_instances_params(config, job, script, *, ami_id, instance_profile_arn, run_id):
    spec = config["jobs"][job]
    params = {"ImageId": ami_id, "InstanceType": spec["instance_type"], "MinCount": 1, "MaxCount": 1,
              "IamInstanceProfile": {"Arn": instance_profile_arn}, "UserData": script,
              "InstanceInitiatedShutdownBehavior": "terminate",
              "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
              "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": spec["volume_gb"], "VolumeType": "gp3",
                                                                          "DeleteOnTermination": True, "Encrypted": True}}],
              "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                  {"Key": "Project", "Value": "cs230"}, {"Key": "Job", "Value": job}, {"Key": "RunId", "Value": run_id},
                  {"Key": "MaxMinutes", "Value": str(spec["max_minutes"])}]}]}
    if config.get("subnet_id"):
        params["SubnetId"] = config["subnet_id"]
    if config.get("security_group_ids"):
        params["SecurityGroupIds"] = list(config["security_group_ids"])
    return params


def admission_receipt(preflight_path, forecaster_config, *, manifest_sha256, job_id, reservation_id, reserved_cents, region, now=None):
    """Launcher receipt bam_train_only checks before loading anything (see _read_execution_admission)."""
    from bam_training import value_hash
    raw = Path(preflight_path).read_bytes()
    preflight = json.loads(raw)
    _require(preflight.get("prerequisites_satisfied") is True and preflight.get("blockers") == [],
             "Preflight has blockers; resolve them before admission")
    binding = preflight.get("binding", {})
    model = {key: forecaster_config[key] for key in ("model_id", "model_revision")}
    _require(binding == {"job_id": job_id, "manifest_sha256": manifest_sha256, "config_sha256": value_hash(forecaster_config),
                         "model": model, "region": region}, "Preflight binding differs from this job, bundle, config or region")
    _require(type(reserved_cents) is int and reserved_cents > 0 and isinstance(reservation_id, str) and reservation_id.strip(),
             "A positive ledger reservation is required")
    checked = datetime.fromisoformat(preflight["checked_at"].replace("Z", "+00:00"))
    now = now or datetime.now(timezone.utc)
    expires = min(checked + timedelta(hours=23), now + timedelta(hours=12))
    _require(expires > now, "Preflight is too old; run it again")
    return {"platform": "aws_ec2", "status": "admitted", "reservation_id": reservation_id, "job_id": job_id,
            "config_sha256": value_hash(forecaster_config), "model": model, "manifest_sha256": manifest_sha256,
            "region": region, "max_runtime_seconds": forecaster_config["train_only"]["max_runtime_seconds"],
            "external_termination_armed": True, "reserved_cents": reserved_cents,
            "preflight_report_sha256": hashlib.sha256(raw).hexdigest(), "expires_at_utc": expires.isoformat()}


def code_archive(output):
    """`git archive` of the committed HEAD; refuses a dirty tree so the instance runs reviewed code."""
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout
    _require(not status.strip(), "Commit tracked changes before uploading code")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(output), "HEAD"], cwd=ROOT, check=True)
    return commit


def _clients(config):
    import boto3
    session = boto3.Session(region_name=config["region"])
    return session.client("ec2"), session.client("ssm"), session.client("cloudformation"), session.client("s3")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "upload-code", "plan", "admit", "launch"])
    parser.add_argument("--job", choices=["smoke", "research", "train"])
    parser.add_argument("--run-id")
    parser.add_argument("--round")
    parser.add_argument("--code-key")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--preflight")
    parser.add_argument("--reservation-id")
    parser.add_argument("--reserved-cents", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "check":
        print(json.dumps({"blockers": readiness(config)}, indent=2))
        return
    if args.command == "admit":
        forecaster = json.loads((ROOT / FORECASTER_CONFIG).read_text())
        receipt = admission_receipt(args.preflight, forecaster, manifest_sha256=args.manifest_sha256, job_id=args.run_id,
                                    reservation_id=args.reservation_id, reserved_cents=args.reserved_cents, region=config["region"])
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    blockers = readiness(config)
    _require(not blockers, "AWS config not ready: " + " | ".join(blockers))
    if args.command == "plan":
        print(user_data(config, args.job, code_key=args.code_key, run_id=args.run_id, round_name=args.round,
                        manifest_sha256=args.manifest_sha256))
        return
    ec2, ssm, cloudformation, s3 = _clients(config)
    if args.command == "upload-code":
        archive = ROOT / "runs" / "code.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        commit = code_archive(archive)
        key = f"{commit}.tar.gz"
        s3.upload_file(str(archive), config["bucket"], f"{config['prefix']}/code/{key}")
        print(json.dumps({"code_key": key, "commit": commit}))
        return
    role = config["jobs"][args.job]["role"]
    profile = cloudformation.describe_stack_resource(StackName=config["stack_name"], LogicalResourceId=STACK_PROFILES[role])
    profile_name = profile["StackResourceDetail"]["PhysicalResourceId"]
    import boto3
    account = boto3.client("sts", region_name=config["region"]).get_caller_identity()["Account"]
    profile_arn = f"arn:aws:iam::{account}:instance-profile/cs230/{profile_name}"
    ami_id = ssm.get_parameter(Name=config["ami_ssm_parameter"])["Parameter"]["Value"]
    script = user_data(config, args.job, code_key=args.code_key, run_id=args.run_id, round_name=args.round,
                       manifest_sha256=args.manifest_sha256)
    params = run_instances_params(config, args.job, script, ami_id=ami_id, instance_profile_arn=profile_arn, run_id=args.run_id)
    if args.dry_run:
        try:
            ec2.run_instances(DryRun=True, **params)
        except Exception as error:  # DryRunOperation means the call would have succeeded
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            print(json.dumps({"dry_run": code, "would_succeed": code == "DryRunOperation", "message": str(error)[:300]}))
            return
    instance = ec2.run_instances(**params)["Instances"][0]
    print(json.dumps({"instance_id": instance["InstanceId"], "job": args.job, "run_id": args.run_id,
                      "max_minutes": config["jobs"][args.job]["max_minutes"], "region": config["region"]}))


if __name__ == "__main__":
    main()
