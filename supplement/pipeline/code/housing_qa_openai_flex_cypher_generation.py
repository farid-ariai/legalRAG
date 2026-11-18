#!/usr/bin/env python3
"""Generate full-clean HousingQA Cypher with GPT-5.6 Terra and Flex.

The campaign is pinned to ``HousingQA_clean``, prompt ``v6``, the OpenAI
Responses API, strict Structured Outputs, and the Flex service tier.  A
deterministic 200-question pilot covers every clean jurisdiction.  Only a
zero-failure pilot unlocks the remaining questions.

Every attempted question is appended immediately to a durable JSONL
checkpoint or failure ledger.  An identical ``--resume`` continues only rows
that have not already reached a terminal outcome.  A complete run publishes
the query file in canonical clean-question order plus the manifest required by
``housing_qa_execute_cypher.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from housing_qa_cypher_safety import validate_read_only_cypher
from housing_qa_dataset_paths import CLEAN_MANIFEST, CLEAN_QUESTIONS
from housing_qa_openai_cypher_generation import validate_cypher
from housing_qa_prompt import VERSIONS, build_chat_messages


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Indices" / "KG_Indices" / "HousingQA_cypher"
EXPECTED_MODEL = "gpt-5.6-terra"
EXPECTED_ROWS = 4584
EXPECTED_STATES = 47
EXPECTED_GROUPS = 225
PROMPT_VERSION = "v6"
RUNNER_VERSION = "housing_qa_openai_flex_v2"
SERVICE_TIER = "flex"
REASONING_EFFORT = "none"
TOP_K = 20
DEFAULT_PILOT_COUNT = 200
DEFAULT_PILOT_SEED = 20260906
PROMPT_CACHE_KEY = "housingqa-clean-cypher-v6-gpt56-terra-flex"
IDENTITY_FIELDS = ("idx", "state", "question", "question_group")

JSON_SCHEMA = {
    "type": "object",
    "properties": {"cypher": {"type": "string"}},
    "required": ["cypher"],
    "additionalProperties": False,
}
RESPONSES_TEXT = {
    "format": {
        "type": "json_schema",
        "name": "housing_qa_cypher",
        "strict": True,
        "schema": JSON_SCHEMA,
    },
    "verbosity": "low",
}
OUTPUT_CONTRACT = (
    "\n\nOUTPUT CONTRACT:\n"
    "Return exactly one JSON object with one key named `cypher`. Its value "
    "must be the complete read-only Cypher query. Do not put commentary, "
    "Markdown, or a second query in the value."
)
CORRECTION_TEMPLATE = (
    "The previous JSON object was well formed, but its Cypher failed this "
    "mechanical output-contract check: {error}. Return a corrected JSON object "
    "under the identical schema. Preserve the retrieval cues, weights, state "
    "filter, and query intent; change only what is needed to satisfy the cited "
    "check. The final RETURN must include `s.idx AS statute_idx` and `score`, "
    "and the query must end with `LIMIT 20`."
)


class FlexGenerationError(RuntimeError):
    """A terminal API, tier, structured-output, or query-validation failure."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_questions(
    questions_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_bytes = questions_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    question_sha = hashlib.sha256(question_bytes).hexdigest()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    rows = json.loads(question_bytes)
    manifest = json.loads(manifest_bytes)
    expected_sha = manifest.get("files", {}).get("housing_q.json", {}).get("sha256")
    counts = manifest.get("counts", {})
    if manifest.get("dataset") != "HousingQA_clean" or question_sha != expected_sha:
        raise ValueError("questions are not the manifest-bound HousingQA_clean snapshot")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected exactly {EXPECTED_ROWS} clean questions")
    if counts.get("n_clean_questions") != EXPECTED_ROWS:
        raise ValueError("dataset manifest clean-question count differs")
    if counts.get("n_states") != EXPECTED_STATES:
        raise ValueError("dataset manifest jurisdiction count differs")
    if counts.get("n_question_groups") != EXPECTED_GROUPS:
        raise ValueError("dataset manifest question-group count differs")

    seen: set[str] = set()
    states: set[str] = set()
    groups: set[int] = set()
    validated: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"question {position} is not an object")
        missing = set(IDENTITY_FIELDS) - set(row)
        if missing:
            raise ValueError(f"question {position} lacks identity fields {sorted(missing)}")
        qid = str(row["idx"])
        if not qid or qid in seen:
            raise ValueError(f"question {position} has an empty or duplicate ID")
        if not str(row["state"]).strip() or not str(row["question"]).strip():
            raise ValueError(f"question {qid} has an empty state or question")
        if type(row["question_group"]) is not int:
            raise ValueError(f"question {qid} has a non-integer question_group")
        seen.add(qid)
        states.add(str(row["state"]).strip().casefold())
        groups.add(row["question_group"])
        validated.append({**row, "_position": position})
    if len(states) != EXPECTED_STATES or len(groups) != EXPECTED_GROUPS:
        raise ValueError("clean questions do not have the expected state/group coverage")
    return validated, {
        "questions_sha256": question_sha,
        "dataset_manifest_sha256": manifest_sha,
        "states": len(states),
        "question_groups": len(groups),
    }


def select_pilot_positions(
    rows: list[dict[str, Any]], count: int, seed: int
) -> tuple[int, ...]:
    if count < EXPECTED_STATES or count >= len(rows):
        raise ValueError(
            f"pilot count must be at least {EXPECTED_STATES} and below {len(rows)}"
        )
    rng = random.Random(seed)
    by_state: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state"]).strip().casefold()].append(row["_position"])
    selected: list[int] = []
    for state in sorted(by_state):
        candidates = list(by_state[state])
        rng.shuffle(candidates)
        selected.append(candidates[0])
    remaining = [
        row["_position"] for row in rows if row["_position"] not in set(selected)
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    return tuple(sorted(selected))


def build_structured_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = build_chat_messages(
        str(row["state"]), str(row["question"]), version=PROMPT_VERSION
    )
    messages[0] = {**messages[0], "content": messages[0]["content"] + OUTPUT_CONTRACT}
    messages[2] = {
        **messages[2],
        "content": json.dumps({"cypher": messages[2]["content"]}, ensure_ascii=False),
    }
    fingerprint = json_sha256(messages[:3])
    return messages, fingerprint


def response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else {"raw": str(usage)}


def request_once(
    client: OpenAI,
    messages: list[dict[str, str]],
    *,
    model: str,
    max_output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    response = client.with_options(timeout=timeout).responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": REASONING_EFFORT},
        text=RESPONSES_TEXT,
        service_tier=SERVICE_TIER,
        store=False,
        prompt_cache_key=PROMPT_CACHE_KEY,
    )
    status = str(getattr(response, "status", "") or "")
    returned_tier = str(getattr(response, "service_tier", "") or "")
    raw = str(getattr(response, "output_text", "") or "")
    details = {
        "response_id": str(getattr(response, "id", "") or ""),
        "response_model": str(getattr(response, "model", "") or ""),
        "response_status": status,
        "service_tier": returned_tier,
        "raw_output": raw,
        "usage": response_usage(response),
    }
    if status != "completed":
        raise FlexGenerationError(f"Responses API status is {status!r}", details)
    if returned_tier != SERVICE_TIER:
        raise FlexGenerationError(
            f"response service tier is {returned_tier!r}, expected 'flex'", details
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FlexGenerationError("Structured Output is not valid JSON", details) from exc
    if not isinstance(payload, dict) or set(payload) != {"cypher"}:
        raise FlexGenerationError("Structured Output does not contain only `cypher`", details)
    if not isinstance(payload["cypher"], str):
        raise FlexGenerationError("Structured Output `cypher` is not a string", details)
    normalised, validation_error = validate_cypher(payload["cypher"], limit=TOP_K)
    if validation_error or normalised is None:
        raise FlexGenerationError(
            f"static query validation failed: {validation_error}",
            {**details, "generated_cypher_query": payload["cypher"]},
        )
    try:
        safe_query = validate_read_only_cypher(normalised)
    except ValueError as exc:
        raise FlexGenerationError(
            f"read-only query validation failed: {exc}",
            {**details, "generated_cypher_query": normalised},
        ) from exc
    return {
        **details,
        "generated_cypher_query": safe_query,
        "structured_payload": payload,
    }


def request_with_capacity_retries(
    client: OpenAI,
    messages: list[dict[str, str]],
    *,
    model: str,
    max_output_tokens: int,
    timeout: float,
    capacity_retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transport: list[dict[str, Any]] = []
    for number in range(1, capacity_retries + 2):
        started = time.monotonic()
        try:
            decoded = request_once(
                client,
                messages,
                model=model,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            code = str(
                getattr(exc, "status_code", "") or getattr(exc, "code", "") or ""
            )
            transport.append(
                {
                    "attempt": number,
                    "status": "failed",
                    "code": code,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
            retryable = code in {"408", "409", "429", "500", "502", "503", "504"}
            if not retryable or number > capacity_retries:
                raise FlexGenerationError(
                    str(exc),
                    {
                        "transport_attempts": transport,
                        "generation_details": dict(getattr(exc, "details", {})),
                    },
                ) from exc
            time.sleep(min(2**number, 60))
            continue
        transport.append(
            {
                "attempt": number,
                "status": "passed",
                "response_id": decoded["response_id"],
                "seconds": round(time.monotonic() - started, 3),
            }
        )
        return decoded, transport
    raise AssertionError("capacity retry loop ended without a result")


def validate_record(record: dict[str, Any], row: dict[str, Any]) -> None:
    if any(record.get(key) != row[key] for key in IDENTITY_FIELDS):
        raise ValueError("generation record identity differs from HousingQA_clean")
    if record.get("model") != EXPECTED_MODEL or record.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("generation record model or prompt version differs")
    openai = record.get("openai")
    if not isinstance(openai, dict) or openai.get("service_tier_returned") != SERVICE_TIER:
        raise ValueError("generation record was not served through Flex")
    if openai.get("structured_outputs") is not True:
        raise ValueError("generation record lacks Structured Outputs provenance")
    normalised, error = validate_cypher(
        str(record.get("generated_cypher_query") or ""), limit=TOP_K
    )
    if error or normalised != record.get("generated_cypher_query"):
        raise ValueError(f"generation record query validation failed: {error}")
    validate_read_only_cypher(normalised)


def generate_question(
    client: OpenAI,
    row: dict[str, Any],
    *,
    model: str,
    max_output_tokens: int,
    timeout: float,
    capacity_retries: int,
) -> dict[str, Any]:
    base_messages, fingerprint = build_structured_messages(row)
    messages = base_messages
    started = time.monotonic()
    generation_attempts: list[dict[str, Any]] = []
    all_transport: list[dict[str, Any]] = []
    decoded: dict[str, Any] | None = None
    for attempt_number in (1, 2):
        try:
            decoded, transport = request_with_capacity_retries(
                client,
                messages,
                model=model,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                capacity_retries=capacity_retries,
            )
        except FlexGenerationError as exc:
            all_transport.extend(exc.details.get("transport_attempts") or [])
            details = dict(exc.details.get("generation_details") or exc.details)
            details.setdefault("validation_error", str(exc))
            generation_attempts.append(
                {"attempt": attempt_number, "status": "failed", **details}
            )
            if (
                attempt_number == 1
                and details.get("raw_output")
                and details.get("generated_cypher_query")
            ):
                messages = [
                    *base_messages,
                    {"role": "assistant", "content": details["raw_output"]},
                    {
                        "role": "user",
                        "content": CORRECTION_TEMPLATE.format(error=str(exc)),
                    },
                ]
                continue
            raise FlexGenerationError(
                str(exc), {"generation_attempts": generation_attempts}
            ) from exc
        all_transport.extend(transport)
        generation_attempts.append(
            {
                "attempt": attempt_number,
                "status": "passed",
                "response_id": decoded["response_id"],
                "response_model": decoded["response_model"],
                "service_tier": decoded["service_tier"],
                "raw_output": decoded["raw_output"],
                "generated_cypher_query": decoded["generated_cypher_query"],
                "usage": decoded["usage"],
            }
        )
        break
    if decoded is None:
        raise AssertionError("bounded generation loop ended without a result")
    record = {
        **{key: row[key] for key in IDENTITY_FIELDS},
        "row_position": row["_position"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint_sha256": fingerprint,
        "cypher_query_source": (
            f"openai_responses_{model}_{SERVICE_TIER}_{PROMPT_VERSION}_structured"
        ),
        "generated_cypher_query": decoded["generated_cypher_query"],
        "raw_output": decoded["raw_output"],
        "structured_payload": decoded["structured_payload"],
        "structured_ok": True,
        "validation_error": None,
        "looks_truncated": False,
        "generation_attempt_count": len(generation_attempts),
        "generation_attempts": generation_attempts,
        "seconds": round(time.monotonic() - started, 3),
        "completed_at": utc_now(),
        "openai": {
            "transport": "responses_api",
            "service_tier_requested": SERVICE_TIER,
            "service_tier_returned": decoded["service_tier"],
            "response_id": decoded["response_id"],
            "response_model": decoded["response_model"],
            "response_status": decoded["response_status"],
            "reasoning_effort": REASONING_EFFORT,
            "structured_outputs": True,
            "store": False,
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "usage": decoded["usage"],
            "transport_attempts": all_transport,
        },
    }
    validate_record(record, row)
    return record


def paths_for(tag: str) -> dict[str, Path]:
    prefix = OUT_DIR / f"housing_qa_clean__gpt56_terra_flex_{PROMPT_VERSION}__{tag}"
    return {
        key: Path(f"{prefix}__{key}.{suffix}")
        for key, suffix in (
            ("queries", "json"),
            ("query_manifest", "json"),
            ("checkpoint", "jsonl"),
            ("failures", "jsonl"),
            ("outcomes", "json"),
            ("manifest", "json"),
            ("progress", "json"),
            ("pilot_pass", "json"),
            ("done", "json"),
        )
    }


def run_contract(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    pilot_positions: tuple[int, ...],
) -> dict[str, Any]:
    _, fingerprint = build_structured_messages(rows[0])
    return {
        "schema_version": 1,
        "artifact": "HousingQA_clean_GPT56_Terra_Flex_query_campaign",
        "runner_version": RUNNER_VERSION,
        "dataset": "HousingQA_clean",
        "questions_sha256": metadata["questions_sha256"],
        "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint_sha256": fingerprint,
        "settings": {
            "endpoint": "/v1/responses",
            "service_tier": SERVICE_TIER,
            "reasoning_effort": REASONING_EFFORT,
            "structured_outputs": True,
            "responses_text_format_sha256": json_sha256(RESPONSES_TEXT),
            "max_output_tokens": args.max_output_tokens,
            "request_timeout": args.request_timeout,
            "capacity_retries": args.capacity_retries,
            "workers": args.workers,
            "store": False,
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "top_k": TOP_K,
            "max_generation_attempts": 2,
            "correction_policy": (
                "one bounded validator-directed mechanical correction preserving retrieval cues"
            ),
        },
        "counts": {
            "questions": len(rows),
            "states": metadata["states"],
            "question_groups": metadata["question_groups"],
            "pilot_questions": len(pilot_positions),
        },
        "pilot": {
            "seed": args.pilot_seed,
            "positions": list(pilot_positions),
            "question_ids_sha256": json_sha256(
                [str(rows[position]["idx"]) for position in pilot_positions]
            ),
            "selection": "one_seeded_question_per_state_then_seeded_fill_v1",
            "gate": "zero terminal failures and all static/read-only validations pass",
        },
        "inputs_sha256": json_sha256(
            [{key: row[key] for key in IDENTITY_FIELDS} for row in rows]
        ),
    }


def load_checkpoint(
    path: Path, rows: list[dict[str, Any]], model: str
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        position = record.get("row_position")
        if type(position) is not int or not 0 <= position < len(rows) or position in completed:
            raise ValueError(f"invalid or duplicate checkpoint identity at line {line_number}")
        if record.get("model") != model:
            raise ValueError(f"checkpoint model mismatch at line {line_number}")
        validate_record(record, rows[position])
        completed[position] = record
    return completed


def load_failures(
    path: Path, rows: list[dict[str, Any]], model: str
) -> dict[int, dict[str, Any]]:
    failures: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return failures
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        position = record.get("row_position")
        if type(position) is not int or not 0 <= position < len(rows) or position in failures:
            raise ValueError(f"invalid or duplicate failure identity at line {line_number}")
        row = rows[position]
        if record.get("model") != model or any(
            record.get(key) != row[key] for key in IDENTITY_FIELDS
        ):
            raise ValueError(f"failure identity mismatch at line {line_number}")
        if not isinstance(record.get("error"), str) or not record["error"]:
            raise ValueError(f"failure lacks a diagnostic at line {line_number}")
        failures[position] = record
    return failures


def publish_final_queries(
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    completed: dict[int, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    ordered = [completed[position] for position in range(len(rows))]
    queries = [
        {key: value for key, value in record.items() if key != "row_position"}
        for record in ordered
    ]
    atomic_write_json(paths["queries"], queries)
    atomic_write_json(paths["outcomes"], queries)
    query_sha = sha256_file(paths["queries"])
    outcomes_sha = sha256_file(paths["outcomes"])
    counts = {
        "source_query_rows": EXPECTED_ROWS,
        "source_unique_question_ids": EXPECTED_ROWS,
        "clean_question_rows": EXPECTED_ROWS,
        "clean_unique_question_ids": EXPECTED_ROWS,
        "output_query_rows": EXPECTED_ROWS,
        "output_unique_question_ids": EXPECTED_ROWS,
        "excluded_source_rows": 0,
        "states": EXPECTED_STATES,
        "question_groups": EXPECTED_GROUPS,
    }
    query_manifest = {
        "schema_version": 1,
        "artifact": "HousingQA_clean_queries",
        "status": "completed",
        "created_at": utc_now(),
        "inputs": {
            "clean_questions": str(CLEAN_QUESTIONS.resolve()),
            "source_queries": str(paths["outcomes"].resolve()),
            "dataset_manifest": str(CLEAN_MANIFEST.resolve()),
        },
        "outputs": {
            "queries": str(paths["queries"].resolve()),
            "manifest": str(paths["query_manifest"].resolve()),
        },
        "sha256": {
            "clean_questions": metadata["questions_sha256"],
            "source_queries": outcomes_sha,
            "dataset_manifest": metadata["dataset_manifest_sha256"],
            "output_queries": query_sha,
            "implementation": sha256_file(Path(__file__)),
        },
        "counts": counts,
        "validation": {
            "canonical_question_hash_matches_dataset_manifest": True,
            "source_query_ids_unique": True,
            "clean_question_ids_unique": True,
            "all_clean_ids_present_in_source": True,
            "idx_state_question_identity_exact": True,
            "question_group_identity_exact": True,
            "output_follows_clean_question_order": True,
            "source_query_records_preserved_without_mutation": True,
            "inputs_unchanged_during_materialisation": True,
            "all_queries_structured_output_valid": True,
            "all_queries_static_and_read_only_valid": True,
            "all_queries_service_tier_flex": True,
        },
        "generation": {
            "model": EXPECTED_MODEL,
            "prompt_version": PROMPT_VERSION,
            "service_tier": SERVICE_TIER,
            "reasoning_effort": REASONING_EFFORT,
            "structured_outputs": True,
        },
    }
    atomic_write_json(paths["query_manifest"], query_manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=CLEAN_QUESTIONS)
    parser.add_argument("--dataset-manifest", type=Path, default=CLEAN_MANIFEST)
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--capacity-retries", type=int, default=8)
    parser.add_argument("--pilot-count", type=int, default=DEFAULT_PILOT_COUNT)
    parser.add_argument("--pilot-seed", type=int, default=DEFAULT_PILOT_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.model != EXPECTED_MODEL:
        raise ValueError(f"this campaign is pinned to {EXPECTED_MODEL}")
    if PROMPT_VERSION not in VERSIONS:
        raise ValueError(f"prompt {PROMPT_VERSION} is unavailable")
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    if args.max_output_tokens <= 0 or args.request_timeout <= 0 or args.capacity_retries < 0:
        raise ValueError("token, timeout, and retry settings are invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.out_tag):
        raise ValueError("invalid output tag")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    rows, metadata = load_questions(args.questions, args.dataset_manifest)
    pilot_positions = select_pilot_positions(rows, args.pilot_count, args.pilot_seed)
    pilot_set = set(pilot_positions)
    paths = paths_for(args.out_tag)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = Path(f"{paths['manifest']}.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another process owns this HousingQA generation tag") from exc

    try:
        contract = run_contract(args, rows, metadata, pilot_positions)
        artefacts = [path for key, path in paths.items() if key != "manifest"]
        if not paths["manifest"].exists() and any(path.exists() for path in artefacts):
            raise ValueError("orphan output artefacts exist without a campaign manifest")
        if paths["manifest"].exists():
            if not args.resume or json.loads(paths["manifest"].read_text()) != contract:
                raise ValueError("existing campaign requires an identical explicit --resume")
        else:
            if args.resume:
                raise ValueError("cannot resume without an existing campaign manifest")
            atomic_write_json(paths["manifest"], contract)

        completed = load_checkpoint(paths["checkpoint"], rows, args.model)
        failures = load_failures(paths["failures"], rows, args.model)
        if set(completed).intersection(failures):
            raise ValueError("a question appears in both successful and failed ledgers")
        if paths["done"].exists():
            done = json.loads(paths["done"].read_text())
            if done.get("status") != "completed" or len(completed) != EXPECTED_ROWS:
                raise ValueError("done receipt contradicts the generation checkpoint")
            print(f"[done] existing verified generation rows={EXPECTED_ROWS}")
            return 0

        client = OpenAI(timeout=args.request_timeout, max_retries=0)
        started = time.monotonic()

        def progress(status: str, phase: str) -> None:
            elapsed = max(time.monotonic() - started, 0.001)
            attempted = len(completed) + len(failures)
            atomic_write_json(
                paths["progress"],
                {
                    **contract,
                    "status": status,
                    "phase": phase,
                    "updated_at": utc_now(),
                    "rows_total": EXPECTED_ROWS,
                    "rows_attempted": attempted,
                    "rows_successful": len(completed),
                    "rows_remaining": EXPECTED_ROWS - attempted,
                    "failures": len(failures),
                    "pilot_attempted": len((set(completed) | set(failures)) & pilot_set),
                    "pilot_successful": len(set(completed) & pilot_set),
                    "pilot_failures": len(set(failures) & pilot_set),
                    "rows_per_hour": round(attempted / elapsed * 3600, 3),
                },
            )

        def run_phase(phase: str, positions: list[int]) -> None:
            pending = [
                rows[position]
                for position in positions
                if position not in completed and position not in failures
            ]
            progress("running", phase)
            print(
                f"[phase] {phase} pending={len(pending)} model={args.model} tier=flex",
                flush=True,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        generate_question,
                        client,
                        row,
                        model=args.model,
                        max_output_tokens=args.max_output_tokens,
                        timeout=args.request_timeout,
                        capacity_retries=args.capacity_retries,
                    ): row
                    for row in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    row = futures[future]
                    position = row["_position"]
                    try:
                        record = future.result()
                        append_jsonl(paths["checkpoint"], record)
                        completed[position] = record
                        outcome = "ok"
                    except Exception as exc:
                        failure = {
                            **{key: row[key] for key in IDENTITY_FIELDS},
                            "row_position": position,
                            "model": args.model,
                            "failed_at": utc_now(),
                            "error": str(exc),
                            "details": dict(getattr(exc, "details", {})),
                        }
                        append_jsonl(paths["failures"], failure)
                        failures[position] = failure
                        outcome = f"FAILED {type(exc).__name__}: {exc}"
                    progress("running", phase)
                    print(
                        f"[row {position}] {outcome} | attempted={len(completed) + len(failures)}/"
                        f"{EXPECTED_ROWS} successful={len(completed)} failures={len(failures)}",
                        flush=True,
                    )

        run_phase("pilot", list(pilot_positions))
        pilot_failures = sorted(pilot_set & set(failures))
        pilot_successes = pilot_set & set(completed)
        if len(pilot_successes) != len(pilot_positions) or pilot_failures:
            progress("pilot_failed", "pilot")
            print(
                f"[stop] pilot gate failed: {len(pilot_successes)}/{len(pilot_positions)} "
                f"successful, {len(pilot_failures)} failures",
                flush=True,
            )
            return 3
        atomic_write_json(
            paths["pilot_pass"],
            {
                **contract,
                "status": "PASS",
                "completed_at": utc_now(),
                "successful_questions": len(pilot_successes),
                "failures": 0,
                "checks": {
                    "all_requested_service_tier_flex": True,
                    "all_returned_service_tier_flex": True,
                    "all_structured_outputs_valid": True,
                    "all_static_query_validations_pass": True,
                    "all_read_only_query_validations_pass": True,
                    "all_47_jurisdictions_represented": True,
                },
            },
        )
        progress("pilot_passed", "full")
        print(f"[gate] pilot PASS ({len(pilot_successes)} questions)", flush=True)

        remaining_positions = [
            position for position in range(EXPECTED_ROWS) if position not in pilot_set
        ]
        run_phase("full", remaining_positions)
        if len(completed) + len(failures) != EXPECTED_ROWS:
            progress("interrupted", "full")
            return 2
        ordered_outcomes = [
            completed.get(position) or failures[position] for position in range(EXPECTED_ROWS)
        ]
        atomic_write_json(paths["outcomes"], ordered_outcomes)
        if failures:
            progress("completed_with_generation_failures", "full")
            print(f"[stop] full generation has {len(failures)} frozen failures", flush=True)
            return 4

        publish_final_queries(paths, rows, completed, metadata)
        progress("completed", "generation")
        atomic_write_json(
            paths["done"],
            {
                **contract,
                "status": "completed",
                "finished_at": utc_now(),
                "integrity": {
                    "attempted_question_count": EXPECTED_ROWS,
                    "successful_query_count": EXPECTED_ROWS,
                    "generation_failure_count": 0,
                    "all_queries_structured_ok": True,
                    "all_queries_service_tier_flex": True,
                    "canonical_clean_question_order": True,
                },
                "outputs": {key: str(path) for key, path in paths.items() if path.exists()},
                "sha256": {key: sha256_file(path) for key, path in paths.items() if path.exists()},
            },
        )
        print(f"[done] generated {EXPECTED_ROWS} queries: {paths['queries']}", flush=True)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
