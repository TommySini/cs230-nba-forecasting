"""Amazon Bedrock transport for TRAIN-only teacher and reviewer requests.

Every request first passes bam_api_teacher.api_payload (manifest-verified TRAIN game,
authorized request, outcome-blind review). Responses are written once per scientific
request ID and reused on retry, so a resumed run never pays twice. A run-level spend
cap is enforced with a worst-case reservation (estimated input plus the full output
budget) before each call; actual cost comes from Bedrock's token usage.

boto3 is imported only when a real client is created; tests inject a fake client.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import bam_api_teacher as api
import bam_research as research


ROLE_KEYS = {"api", "bedrock_model_id", "region", "price_per_million_input_usd", "price_per_million_output_usd", "note"}
STOP_REASONS = {"end_turn": "end_turn", "stop_sequence": "end_turn", "max_tokens": "length"}
RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "InternalServerException"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def role_spec(spec):
    """Validated Bedrock role (teacher or reviewer); unset decisions fail closed."""
    _require(isinstance(spec, dict) and set(spec) <= ROLE_KEYS, "Unexpected Bedrock role fields")
    api.api_spec(spec.get("api") or {})
    _require(isinstance(spec.get("bedrock_model_id"), str) and spec["bedrock_model_id"].strip(), "Bedrock model ID required")
    _require(isinstance(spec.get("region"), str) and spec["region"].strip(), "Bedrock region required")
    for key in ("price_per_million_input_usd", "price_per_million_output_usd"):
        value = spec.get(key)
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"Current {key} required")
    return spec


def identity(spec):
    return api.api_identity(spec["api"])


def converse_params(envelope, spec):
    """Bedrock Converse arguments; consecutive user messages become one user turn."""
    messages = envelope["messages"]
    _require(messages[0]["role"] == "system" and all(m["role"] == "user" for m in messages[1:]), "System plus user messages required")
    return {"modelId": spec["bedrock_model_id"], "system": [{"text": messages[0]["content"]}],
            "messages": [{"role": "user", "content": [{"text": m["content"]} for m in messages[1:]]}],
            "inferenceConfig": {"maxTokens": envelope["max_output_tokens"], "temperature": 0}}


def cost_cents(spec, input_tokens, output_tokens):
    return math.ceil((input_tokens * spec["price_per_million_input_usd"]
                      + output_tokens * spec["price_per_million_output_usd"]) / 1e6 * 100)


def reservation_cents(envelope, spec):
    """Worst case: about three characters per input token, plus the full output budget."""
    characters = sum(len(m["content"]) for m in envelope["messages"])
    return cost_cents(spec, math.ceil(characters / 3), envelope["max_output_tokens"])


def spent_cents(directory):
    directory = Path(directory)
    return sum(json.loads(p.read_text())["cost_cents"] for p in sorted(directory.glob("*.json"))) if directory.exists() else 0


def make_client(region):
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-runtime", region_name=region, config=Config(read_timeout=900, retries={"max_attempts": 1}))


def _error_name(error):
    response = getattr(error, "response", None) or {}
    return response.get("Error", {}).get("Code") or type(error).__name__


def submit(request, spec, registry, directory, *, cap_cents, client=None, targets=None, dry_run=False, attempts=4, sleep=time.sleep):
    """Send one authorized request (or return its cached response).

    Returns the stored record: text, finish_reason, usage and cost. Errors that survive
    retries are stored too (no cost) so the caller can inspect them; they never count as
    a model answer.
    """
    role_spec(spec)
    directory = Path(directory)
    path = directory / f"{request['request_id']}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        _require(saved["request_sha256"] == research.value_hash(request) and saved["model"] == identity(spec),
                 "Cached Bedrock response belongs to a different request or model")
        return saved
    envelope = api.api_payload(request, spec["api"], registry, targets=targets)
    reserve = reservation_cents(envelope, spec)
    spent = spent_cents(directory)
    _require(spent + reserve <= cap_cents, f"Spend cap: {spent} + worst case {reserve} cents exceeds {cap_cents}")
    if dry_run:
        return {"request_id": request["request_id"], "status": "dry_run", "reservation_cents": reserve}
    client = client or make_client(spec["region"])
    params = converse_params(envelope, spec)
    started = datetime.now(timezone.utc).isoformat()
    record = {"request_id": request["request_id"], "request_sha256": research.value_hash(request), "stage": request["stage"],
              "game_id": request["game_id"], "model": identity(spec), "bedrock_model_id": spec["bedrock_model_id"],
              "region": spec["region"], "submitted_at_utc": started}
    for attempt in range(attempts):
        try:
            response = client.converse(**params)
            break
        except Exception as error:  # noqa: BLE001 - provider errors are recorded, not swallowed
            name = _error_name(error)
            if name in RETRYABLE and attempt + 1 < attempts:
                sleep(min(60, 2 ** attempt * 5))
                continue
            record.update(status="error", error=f"{name}: {str(error)[:500]}", text="", finish_reason=None,
                          usage={"inputTokens": 0, "outputTokens": 0}, cost_cents=0,
                          finished_at_utc=datetime.now(timezone.utc).isoformat())
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            return record
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    usage = response.get("usage", {})
    record.update(status="complete", text="".join(b["text"] for b in blocks if "text" in b),
                  reasoning_blocks=sum("reasoningContent" in b for b in blocks),
                  stop_reason=response.get("stopReason"), finish_reason=STOP_REASONS.get(response.get("stopReason"), "error"),
                  usage={"inputTokens": int(usage.get("inputTokens", 0)), "outputTokens": int(usage.get("outputTokens", 0))},
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    record["cost_cents"] = cost_cents(spec, record["usage"]["inputTokens"], record["usage"]["outputTokens"])
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record
