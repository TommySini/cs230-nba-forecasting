"""Read-only funding/prerequisite checks for the TRAIN pilot; never launches jobs.

Inputs are dated evidence supplied by the account operator, not self-verifying
claims. Account checks, price collection, ledger reservation, and launch remain
separate operations. A passing report is NOT an atomic admission/reservation.
The CLI reads only the ledger named in the existing budget configuration and
never creates, copies, resets, reserves, or reconciles a ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import fcntl
import hashlib
import json
from pathlib import Path
import re

from bam_budget import BUFFER_CENTS, MARGIN, STATES

G_QUOTA = "L-DB2E81BA"
MAX_EVIDENCE_AGE = timedelta(hours=24)
REQUIRED_OVERHEADS = {"storage", "transfer", "logging", "cleanup", "tax"}


def is_commercial_region(value):
    """Check region-name format only; live evidence must establish availability.

    GovCloud, China and isolated partitions cannot use the existing arn:aws
    role contract. This deliberately makes no capacity or opt-in assertion.
    """
    return isinstance(value, str) and re.fullmatch(
        r"(?:af|ap|ca|eu|il|me|mx|sa|us)-"
        r"(?:central|east|north|northeast|northwest|south|southeast|southwest|west)-[1-9][0-9]*",
        value,
    ) is not None


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name}: explicit numeric value required")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name}: invalid number") from exc
    if not number.is_finite() or number < 0 or (positive and number <= 0):
        raise ValueError(f"{name}: finite {'positive' if positive else 'nonnegative'} value required")
    return number


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: integer >= {minimum} required")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: evidence reference or value required")
    return value


def _time(value, name):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name}: timezone-aware ISO timestamp required") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name}: timezone-aware ISO timestamp required")
    return parsed.astimezone(timezone.utc)


def _fresh(record, name, now):
    checked = _time(record.get("checked_at"), name + ".checked_at")
    if checked > now or now - checked > MAX_EVIDENCE_AGE:
        raise ValueError(f"{name}: evidence is future-dated or older than 24 hours")
    _text(record.get("reference"), name + ".reference")


def _cents(amount, *, floor=False):
    return int((amount * 100).to_integral_value(rounding=ROUND_FLOOR if floor else ROUND_CEILING))


def content_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ledger_available_cents(ledger):
    """Recompute from original records; never trust a supplied totals field."""
    if (ledger.get("schema_version") != 1 or ledger.get("cap_cents") != 20000
            or ledger.get("buffer_cents") != BUFFER_CENTS
            or ledger.get("reservation_multiplier") != str(MARGIN)):
        raise ValueError("ledger: existing $200 cap, $5 buffer and 1.15 margin required")
    calls = ledger.get("calls")
    if not isinstance(calls, dict) or not isinstance(ledger.get("events"), list):
        raise ValueError("ledger: original calls and event history required")
    committed = 0
    for key, call in calls.items():
        if not isinstance(call, dict) or call.get("call_key") != key or call.get("state") not in STATES:
            raise ValueError("ledger: invalid durable call record")
        reserved = _integer(call.get("reservation_cents"), "ledger.reservation_cents", minimum=0)
        if call["state"] == "completed":
            committed += _integer(call.get("estimated_cost_cents"), "ledger.estimated_cost_cents", minimum=0)
        elif call["state"] != "released":
            committed += reserved
    return ledger["cap_cents"] - BUFFER_CENTS - committed


def evaluate_preflight(evidence, plan, ledger, *, now=None):
    """Collect independent blockers and deterministic worst-case cost bounds.

    Credit evidence must describe an earned, remaining pool eligible for EVERY
    nonzero service in this job. A promotional potential amount, a project cap,
    or a budget ledger is not credit evidence. Accrued/unbilled charges plus all
    outstanding experiment reservations are conservatively deducted again from
    credits; this may double-count reservations already reflected in the balance,
    but never assumes they are free. Current phase authorizes no cash shortfall.
    """
    if not all(isinstance(x, dict) for x in (evidence, plan, ledger)):
        raise ValueError("evidence, plan, and ledger must be JSON objects")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    blockers = []
    costs = []
    services = set()
    financials = {}
    region = plan.get("region")

    def check(label, fn):
        try:
            return fn()
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            blockers.append(f"{label}: {exc}")
            return None

    def scope():
        if evidence.get("schema_version") != 1 or plan.get("schema_version") != 1:
            raise ValueError("evidence and plan schema_version must be 1")
        account = _text(evidence.get("account_id"), "account_id")
        if not re.fullmatch(r"\d{12}", account) or plan.get("account_id") != account:
            raise ValueError("account identities must match")
        if (plan.get("platform") != "aws" or not is_commercial_region(region)
                or plan.get("phase") != "train_pilot" or plan.get("splits") != ["TRAIN"]):
            raise ValueError("AWS TRAIN-only pilot with an explicit commercial region required; DEV and TEST excluded")
        if not 16 <= _integer(plan.get("game_count"), "game_count") <= 32:
            raise ValueError("pilot must contain 16-32 games")
        if plan.get("cash_spend_authorized_usd") != 0:
            raise ValueError("this phase permits no assumed out-of-pocket shortfall")
        _text(plan.get("job_id"), "job_id")
        for name in ("manifest_sha256", "config_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(plan.get(name))):
                raise ValueError(f"{name}: pinned identity digest required")
        model = plan.get("model", {})
        if model.get("model_id") != "Qwen/Qwen2.5-14B-Instruct":
            raise ValueError("pilot forecaster must use Qwen/Qwen2.5-14B-Instruct")
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get("model_revision"))):
            raise ValueError("forecaster must pin an immutable model revision")
    check("scope", scope)
    account = evidence.get("account_id", "")

    def window():
        start = _time(plan.get("planned_start_at"), "planned_start_at")
        if start < now or start - now > MAX_EVIDENCE_AGE:
            raise ValueError("planned start must be within the next 24 hours")
        seconds = _integer(plan.get("max_total_runtime_seconds"), "max_total_runtime_seconds")
        if seconds > 86400:
            raise ValueError("pilot runtime including cleanup must be <=24 hours")
        return start + timedelta(seconds=seconds)
    finish = check("runtime", window)

    def execution():
        roles = evidence.get("execution", {})
        _fresh(roles, "execution", now)
        if roles.get("region") != region or roles.get("status") != "deployed":
            raise ValueError("read deployed role policies for the plan region before launch")
        for kind in ("controller", "gpu"):
            role = roles.get(kind, {})
            if not re.fullmatch(r"arn:aws:iam::" + re.escape(account) + r":role/cs230/[\w+=,.@/-]+", str(role.get("role_arn"))):
                raise ValueError(f"{kind}: project-scoped execution role ARN required; root/user identities rejected")
            if not re.fullmatch(r"arn:aws:iam::" + re.escape(account) + r":instance-profile/cs230/[\w+=,.@/-]+", str(role.get("instance_profile_arn"))):
                raise ValueError(f"{kind}: matching project instance profile required")
            if (role.get("train_only_policy_verified") is not True
                    or role.get("no_static_credentials") is not True
                    or role.get("trust_service") != "ec2.amazonaws.com"):
                raise ValueError(f"{kind}: reviewed TRAIN-only EC2 policy and absence of static credentials required")
            if not re.fullmatch(r"[0-9a-f]{64}", str(role.get("policy_sha256"))):
                raise ValueError(f"{kind}: deployed policy digest required")
    check("execution", execution)

    def quota():
        q = evidence.get("gpu_quota", {})
        _fresh(q, "gpu_quota", now)
        if q.get("region") != region or q.get("quota_code") != G_QUOTA:
            raise ValueError("On-Demand G/VT quota evidence for the plan region required")
        if _number(q.get("value"), "gpu_quota.value") < 8:
            raise ValueError("effective G/VT quota is below 8 vCPUs; pending requests do not count")
    check("gpu_quota", quota)

    model_ids = {}
    for kind in ("teacher", "reviewer"):
        def model_check(kind=kind):
            m = evidence.get("models", {}).get(kind, {})
            _fresh(m, kind, now)
            if m.get("region") != region:
                raise ValueError("model availability for the plan region required")
            for field, expected in (("agreementAvailability", "AVAILABLE"),
                                    ("authorizationStatus", "AUTHORIZED"),
                                    ("entitlementAvailability", "AVAILABLE"),
                                    ("regionAvailability", "AVAILABLE")):
                if m.get(field) != expected:
                    raise ValueError(f"{field} must be {expected}; got {m.get(field)!r}")
            model_ids[kind] = _text(m.get("model_id"), kind + ".model_id")
            _text(m.get("inference_profile_id"), kind + ".inference_profile_id")
        check(kind, model_check)
    if len(model_ids) == 2 and model_ids["teacher"] == model_ids["reviewer"]:
        blockers.append("models: teacher and outcome-blind reviewer must be different models")

    def quote(record, label, resource):
        q = record.get("quote", {})
        _fresh(q, label + ".quote", now)
        if q.get("resource_id") != resource or q.get("region") != region or q.get("currency") != "USD":
            raise ValueError("quote must pin requested resource, plan region and USD")
        _text(q.get("service"), label + ".quote.service")
        return q

    for kind in ("controller", "gpu"):
        def compute(kind=kind):
            c = plan.get("compute", {}).get(kind, {})
            instance = _text(c.get("instance_type"), kind + ".instance_type")
            if kind == "gpu" and instance != "g6e.2xlarge":
                raise ValueError("this pilot gate is pinned to one g6e.2xlarge")
            if kind == "controller" and not re.fullmatch(r"[cmrt][0-9][a-z0-9]*\.[a-z0-9]+", instance):
                raise ValueError("controller must use a CPU-only instance family")
            if c.get("count") != 1:
                raise ValueError("one instance per compute role required")
            seconds = _integer(c.get("max_runtime_seconds"), kind + ".max_runtime_seconds")
            if seconds > _integer(plan.get("max_total_runtime_seconds"), "max_total_runtime_seconds"):
                raise ValueError("instance runtime exceeds total bound")
            if c.get("terminate_at_deadline") is not True:
                raise ValueError("termination deadline required; stopping is insufficient")
            q = quote(c, kind, instance)
            costs.append(_number(q.get("hourly_usd"), kind + ".hourly_usd", positive=True) * seconds / 3600)
            services.add(q["service"])
        check(kind + "_cost", compute)

    for kind in ("teacher", "reviewer"):
        def tokens(kind=kind):
            c = plan.get("model_calls", {}).get(kind, {})
            m = evidence.get("models", {}).get(kind, {})
            model = _text(c.get("model_id"), kind + ".model_id")
            if model != m.get("model_id") or c.get("inference_profile_id") != m.get("inference_profile_id"):
                raise ValueError("request model and inference profile must match availability evidence")
            requests = _integer(c.get("max_requests"), kind + ".max_requests")
            if requests < _integer(plan.get("game_count"), "game_count"):
                raise ValueError("teacher/reviewer bounds must cover every pilot game")
            attempts = _integer(c.get("max_attempts_per_request"), kind + ".max_attempts_per_request")
            inputs = _integer(c.get("max_input_tokens"), kind + ".max_input_tokens")
            outputs = _integer(c.get("max_output_tokens"), kind + ".max_output_tokens")
            if c.get("reasoning_in_output_budget") is not True:
                raise ValueError("reasoning tokens must be included in output bound")
            _text(c.get("input_count_basis"), kind + ".input_count_basis")
            q = quote(c, kind, c.get("inference_profile_id"))
            in_rate = _number(q.get("input_usd_per_million"), kind + ".input_rate", positive=True)
            out_rate = _number(q.get("output_usd_per_million"), kind + ".output_rate", positive=True)
            costs.append(requests * attempts * (inputs * in_rate + outputs * out_rate) / 1000000)
            services.add(q["service"])
        check(kind + "_cost", tokens)

    for kind in sorted(REQUIRED_OVERHEADS):
        def overhead(kind=kind):
            c = plan.get("overheads", {}).get(kind, {})
            amount = _number(c.get("max_cost_usd"), kind + ".max_cost_usd")
            q = quote(c, kind, kind)
            _text(c.get("bound_basis"), kind + ".bound_basis")
            if amount > 0:
                services.add(q["service"])
            costs.append(amount)
        check(kind + "_cost", overhead)

    required = _cents(sum(costs, Decimal(0)) * MARGIN)
    financials["bounded_reservation_cents"] = required
    financials["reservation_margin"] = str(MARGIN)
    financials["cost_bounds_complete"] = not any("_cost:" in b for b in blockers)

    def budget():
        available = ledger_available_cents(ledger)
        financials["ledger_available_cents"] = available
        ceiling = _cents(_number(plan.get("max_cost_usd"), "max_cost_usd", positive=True), floor=True)
        if required > ceiling:
            raise ValueError("bounded cost plus margin exceeds this run's spend ceiling")
        if required > available:
            raise ValueError("existing completed estimates/reservations and buffer leave insufficient project budget")
    check("budget", budget)

    def credits():
        c = evidence.get("credits", {})
        _fresh(c, "credits", now)
        if c.get("balance_kind") != "earned_remaining" or c.get("currency") != "USD":
            raise ValueError("earned remaining USD credit evidence required; potential rewards are not funds")
        _text(c.get("eligibility_reference"), "credits.eligibility_reference")
        expiry = _time(c.get("expires_at"), "credits.expires_at")
        if finish is None or expiry <= finish:
            raise ValueError("credits must remain valid through runtime and cleanup")
        eligible = c.get("eligible_services")
        if not isinstance(eligible, list) or any(not isinstance(x, str) for x in eligible):
            raise ValueError("explicit eligible service list required")
        missing = services - set(eligible)
        if missing:
            raise ValueError("credit eligibility unverified for: " + ", ".join(sorted(missing)))
        remaining = _number(c.get("remaining_usd"), "credits.remaining_usd")
        unbilled = _number(c.get("accrued_unbilled_usd"), "credits.accrued_unbilled_usd")
        _text(c.get("accrued_unbilled_reference"), "credits.accrued_unbilled_reference")
        # Validate records first, including uncertain charges, then retain them.
        ledger_available_cents(ledger)
        outstanding = sum(x["reservation_cents"] for x in ledger["calls"].values()
                          if x["state"] not in ("completed", "released"))
        available = _cents(remaining, floor=True) - _cents(unbilled) - outstanding - BUFFER_CENTS
        financials["verified_credit_available_cents"] = available
        if required > available:
            raise ValueError("verified eligible credits are insufficient; project cap does not authorize cash")
    check("credits", credits)

    hashes = {name + "_content_sha256": check(name + "_hash", lambda value=value: content_sha256(value))
              for name, value in (("evidence", evidence), ("plan", plan), ("ledger", ledger))}
    return {"schema_version": 1, "checked_at": now.isoformat(),
            "prerequisites_satisfied": not blockers, "blockers": blockers,
            "financials": financials, "ledger_mutated": False,
            "reservation_created": False, "launch_authorized": False,
            "binding": {key: plan.get(key) for key in ("job_id", "manifest_sha256", "config_sha256", "model", "region")},
            **hashes,
            "next_step": "Refresh evidence and reserve atomically in the original single-writer ledger before any launch."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--budget-config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.budget_config.read_text())
    if config.get("cap_usd") != 200 or config.get("compute_platform") != "aws":
        parser.error("existing $200 AWS budget configuration required")
    ledger_path = Path(config["ledger_path"]).resolve()
    # Use the same lock as BudgetLedger; do not create missing accounting files.
    with Path(str(ledger_path) + ".lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        ledger_bytes = ledger_path.read_bytes()
        ledger = json.loads(ledger_bytes)
        result = evaluate_preflight(json.loads(args.evidence.read_text()),
                                    json.loads(args.plan.read_text()), ledger)
        result["ledger_path"] = str(ledger_path)
        result["ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["prerequisites_satisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
