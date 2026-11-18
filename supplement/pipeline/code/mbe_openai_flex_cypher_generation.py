#!/usr/bin/env python3
"""Generate MBE Cypher with GPT-5.6 Terra using the frozen prompt and Flex.

This is an additive OpenAI Responses API transport for ``mbe_v1``.  It reuses
the exact prompt messages, JSON schema, validator, bounded CASE-cap correction,
question whitelist and provenance checks used by the local-model campaign.
Every successful or failed row is checkpointed immediately and an identical
``--resume`` continues only unattempted rows.  Validator failures are retained
as model outcomes rather than regenerated, because retrying them with a changed
prompt would invalidate the same-prompt comparison.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

import mbe_api_cypher_generation as generation
from mbe_prompt_v1 import CORRECTION_TEMPLATE, MAX_CASE_CLAUSES, PROMPT_VERSION


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Indices" / "KG_Indices" / "MBE_cypher"
EXPECTED_MODEL = "gpt-5.6-terra"
EXPECTED_ROWS = 1071
RUNNER_VERSION = "mbe_openai_flex_v1"
SERVICE_TIER = "flex"
REASONING_EFFORT = "none"
PROMPT_CACHE_KEY = "mbe-cypher-mbe-v1-gpt56-terra-flex"
IDENTITY_FIELDS = generation.IDENTITY_FIELDS
JSON_SCHEMA = generation.RESPONSE_FORMAT["json_schema"]["schema"]
RESPONSES_TEXT = {"format": {
    "type": "json_schema",
    "name": generation.RESPONSE_FORMAT["json_schema"]["name"],
    "strict": True,
    "schema": JSON_SCHEMA,
}, "verbosity": "low"}


class FlexGenerationError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def response_usage(response: Any) -> tuple[int | None, int | None, dict[str, Any] | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    payload = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    return payload.get("input_tokens"), payload.get("output_tokens"), payload


def request_once(
    client: OpenAI, messages: list[dict[str, str]], *, model: str,
    max_output_tokens: int, timeout: float,
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
    actual_tier = str(getattr(response, "service_tier", "") or "")
    prompt_tokens, completion_tokens, usage = response_usage(response)
    raw = str(getattr(response, "output_text", "") or "")
    details = {
        "response_id": str(getattr(response, "id", "") or ""),
        "response_model": str(getattr(response, "model", "") or ""),
        "response_status": status,
        "service_tier": actual_tier,
        "raw_output": raw,
        "finish_reason": "stop" if status == "completed" else status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage": usage,
    }
    if status != "completed":
        raise FlexGenerationError(f"Responses API status is {status!r}", details)
    if actual_tier != SERVICE_TIER:
        raise FlexGenerationError(f"response service tier is {actual_tier!r}, expected 'flex'", details)
    synthetic = {"choices": [{"finish_reason": "stop", "message": {"content": raw}}],
                 "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}
    try:
        decoded = generation.decode_response(synthetic)
    except generation.GenerationError as exc:
        raise FlexGenerationError(str(exc), {**details, **exc.details}) from exc
    return {**decoded, **details, "finish_reason": "stop"}


def request_with_capacity_retries(
    client: OpenAI, messages: list[dict[str, str]], *, model: str,
    max_output_tokens: int, timeout: float, capacity_retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transport: list[dict[str, Any]] = []
    for number in range(1, capacity_retries + 2):
        started = time.monotonic()
        try:
            decoded = request_once(client, messages, model=model,
                                   max_output_tokens=max_output_tokens, timeout=timeout)
        except Exception as exc:
            code = str(getattr(exc, "status_code", "") or getattr(exc, "code", "") or "")
            transport.append({"attempt": number, "status": "failed", "code": code,
                              "error_type": type(exc).__name__, "message": str(exc)[:2000],
                              "seconds": round(time.monotonic() - started, 3)})
            retryable = code in {"408", "409", "429", "500", "502", "503", "504"}
            if not retryable or number > capacity_retries:
                raise FlexGenerationError(str(exc), {
                    "transport_attempts": transport,
                    "generation_details": dict(getattr(exc, "details", {})),
                }) from exc
            time.sleep(min(2 ** number, 60))
            continue
        transport.append({"attempt": number, "status": "passed",
                          "response_id": decoded["response_id"],
                          "seconds": round(time.monotonic() - started, 3)})
        return decoded, transport
    raise AssertionError("capacity retry loop ended without a result")


def make_record(
    row: dict[str, Any], decoded: dict[str, Any], attempts: list[dict[str, Any]],
    transport: list[dict[str, Any]], model: str, fingerprint: str, seconds: float,
) -> dict[str, Any]:
    raw_query = decoded["raw_cypher"]
    record = {
        **{key: row[key] for key in IDENTITY_FIELDS},
        "input_warnings": row.get("input_warnings", []),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint_sha256": fingerprint,
        "cypher_query": decoded["cypher_query"],
        "raw_cypher": raw_query,
        "raw_output": decoded["raw_output"],
        "structured_ok": True,
        "finish_reason": decoded["finish_reason"],
        "case_clause_count": generation.case_clause_count(decoded["cypher_query"]),
        "apostrophe_repairs": decoded["apostrophe_repairs"],
        "normalisations": {
            "rule": "single_quoted_inner_word_apostrophe_v1",
            "count": decoded["apostrophe_repairs"],
            "raw_sha256": hashlib.sha256(raw_query.encode()).hexdigest(),
            "canonical_sha256": hashlib.sha256(decoded["cypher_query"].encode()).hexdigest(),
        },
        "generation_attempt_count": len(attempts),
        "generation_attempts": attempts,
        "completion_tokens_total": sum(int(item.get("completion_tokens") or 0) for item in attempts),
        "seconds": round(seconds, 3),
        "completed_at": generation.utc_now(),
        "openai": {
            "transport": "responses_api",
            "service_tier_requested": SERVICE_TIER,
            "service_tier_returned": decoded["service_tier"],
            "response_id": decoded["response_id"],
            "response_model": decoded["response_model"],
            "reasoning_effort": REASONING_EFFORT,
            "structured_outputs": True,
            "store": False,
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "usage": decoded["usage"],
            "transport_attempts": transport,
        },
    }
    generation.validate_record_provenance(record, row)
    return record


def generate_question(
    client: OpenAI, row: dict[str, Any], *, model: str, max_output_tokens: int,
    timeout: float, capacity_retries: int,
) -> dict[str, Any]:
    base, fingerprint = generation.build_messages(row)
    messages = base
    generation_attempts: list[dict[str, Any]] = []
    all_transport: list[dict[str, Any]] = []
    started = time.monotonic()
    for attempt_number in (1, 2):
        try:
            decoded, transport = request_with_capacity_retries(
                client, messages, model=model, max_output_tokens=max_output_tokens,
                timeout=timeout, capacity_retries=capacity_retries,
            )
        except FlexGenerationError as exc:
            details = dict(exc.details.get("generation_details") or exc.details)
            details.setdefault("validation_error", f"{type(exc).__name__}: {exc}")
            generation_attempts.append({"attempt": attempt_number, "status": "failed", **details})
            match = re.fullmatch(r"query has (\d+) CASE clauses; maximum is 10",
                                 str(details.get("validation_error") or ""))
            if attempt_number == 1 and match and details.get("raw_output"):
                instruction = CORRECTION_TEMPLATE.format(count=match.group(1))
                messages = [*base, {"role": "assistant", "content": details["raw_output"]},
                            {"role": "user", "content": instruction}]
                continue
            raise FlexGenerationError(str(exc), {"generation_attempts": generation_attempts}) from exc
        all_transport.extend(transport)
        generation_attempts.append({"attempt": attempt_number, "status": "passed", **decoded})
        return make_record(row, decoded, generation_attempts, all_transport, model,
                           fingerprint, time.monotonic() - started)
    raise AssertionError("bounded generation loop ended without a result")


def paths_for(tag: str) -> dict[str, Path]:
    prefix = OUT_DIR / f"mbe_openai__{PROMPT_VERSION}__{tag}"
    return {key: Path(f"{prefix}__{key}.{suffix}") for key, suffix in (
        ("queries", "json"), ("checkpoint", "jsonl"), ("failures", "jsonl"),
        ("outcomes", "json"), ("manifest", "json"), ("progress", "json"),
        ("done", "json"),
    )}


def run_contract(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, fingerprint = generation.build_messages(rows[0])
    return {
        "runner_version": RUNNER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint_sha256": fingerprint,
        "model": args.model,
        "settings": {
            "endpoint": "/v1/responses",
            "service_tier": SERVICE_TIER,
            "reasoning_effort": REASONING_EFFORT,
            "structured_outputs": True,
            "responses_text_format_sha256": generation.json_sha256(RESPONSES_TEXT),
            "max_output_tokens": args.max_output_tokens,
            "request_timeout": args.request_timeout,
            "capacity_retries": args.capacity_retries,
            "workers": args.workers,
            "store": False,
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "max_case_clauses": MAX_CASE_CLAUSES,
            "max_generation_attempts": 2,
        },
        "source_sha256": generation.sha256_file(args.questions),
        "inputs_sha256": generation.json_sha256(rows),
        "row_indices": [row["row_index"] for row in rows],
        "rows": len(rows),
        "full_run": len(rows) == EXPECTED_ROWS,
    }


def load_checkpoint(path: Path, rows: list[dict[str, Any]], model: str) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    expected = {row["row_index"]: row for row in rows}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        row_index = record.get("row_index")
        if row_index not in expected or row_index in completed or record.get("model") != model:
            raise ValueError(f"invalid or duplicate checkpoint identity at line {line_number}")
        generation.validate_record_provenance(record, expected[row_index])
        if record.get("openai", {}).get("service_tier_returned") != SERVICE_TIER:
            raise ValueError("checkpoint was not served through Flex")
        completed[row_index] = record
    return completed


def load_failures(path: Path, rows: list[dict[str, Any]], model: str) -> dict[int, dict[str, Any]]:
    failures: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return failures
    expected = {row["row_index"]: row for row in rows}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        row_index = record.get("row_index")
        if row_index not in expected or row_index in failures or record.get("model") != model:
            raise ValueError(f"invalid or duplicate failure identity at line {line_number}")
        row = expected[row_index]
        if any(record.get(key) != row[key] for key in IDENTITY_FIELDS):
            raise ValueError(f"failure input identity mismatch at line {line_number}")
        if not isinstance(record.get("error"), str) or not record["error"]:
            raise ValueError(f"failure has no diagnostic at line {line_number}")
        failures[row_index] = record
    return failures


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    generation.append_jsonl(path, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "Dataset" / "barexam_qa.csv")
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--capacity-retries", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--row-indices", help="Optional comma-separated positional rows for a validation run")
    args = parser.parse_args(argv)
    if args.model != EXPECTED_MODEL:
        raise ValueError(f"this campaign is pinned to {EXPECTED_MODEL}")
    if not 1 <= args.workers <= 4 or args.max_output_tokens <= 0 or args.request_timeout <= 0:
        raise ValueError("require 1–4 workers and positive token/timeout settings")
    if args.capacity_retries < 0 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.out_tag):
        raise ValueError("invalid retry count or output tag")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    all_rows = generation.load_questions(args.questions)
    if len(all_rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} MBE questions, found {len(all_rows)}")
    rows = all_rows
    if args.row_indices:
        selected = [int(value.strip()) for value in args.row_indices.split(",") if value.strip()]
        if not selected or len(selected) != len(set(selected)) or any(index < 0 or index >= EXPECTED_ROWS for index in selected):
            raise ValueError("row indices must be distinct known positional rows")
        selected_set = set(selected)
        rows = [row for row in all_rows if row["row_index"] in selected_set]
    paths = paths_for(args.out_tag)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = Path(f"{paths['manifest']}.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another process owns this OpenAI generation tag") from exc
    try:
        contract = run_contract(args, rows)
        if not paths["manifest"].exists() and any(path.exists() for path in paths.values()):
            raise ValueError("orphan output artefacts exist without a manifest")
        if paths["manifest"].exists():
            if not args.resume or json.loads(paths["manifest"].read_text()) != contract:
                raise ValueError("existing run requires an identical explicit --resume")
        else:
            if args.resume:
                raise ValueError("cannot resume without an existing manifest")
            generation.atomic_write_json(paths["manifest"], contract)
        completed = load_checkpoint(paths["checkpoint"], rows, args.model)
        failures = load_failures(paths["failures"], rows, args.model)
        if set(completed).intersection(failures):
            raise ValueError("a row appears in both successful and failed checkpoints")
        if paths["done"].exists():
            done = json.loads(paths["done"].read_text())
            if len(completed) + len(failures) != len(rows) or done.get("status") not in {
                    "completed", "completed_with_generation_failures"}:
                raise ValueError("done receipt contradicts the checkpoint")
            print(f"[done] existing verified generation rows={len(rows)}")
            return 0
        client = OpenAI(timeout=args.request_timeout, max_retries=0)
        attempted = set(completed) | set(failures)
        pending = [row for row in rows if row["row_index"] not in attempted]
        started = time.monotonic()

        def progress(status: str) -> None:
            elapsed = max(time.monotonic() - started, 0.001)
            generation.atomic_write_json(paths["progress"], {
                **contract, "status": status, "updated_at": generation.utc_now(),
                "rows_total": len(rows), "rows_attempted": len(completed) + len(failures),
                "rows_successful": len(completed),
                "rows_remaining": len(rows) - len(completed) - len(failures),
                "failures": len(failures),
                "rows_per_hour": round(len(completed) / elapsed * 3600, 3),
            })

        progress("running")
        print(f"[start] rows={len(rows)} pending={len(pending)} model={args.model} tier=flex", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate_question, client, row, model=args.model,
                                   max_output_tokens=args.max_output_tokens,
                                   timeout=args.request_timeout,
                                   capacity_retries=args.capacity_retries): row for row in pending}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    record = future.result()
                    append_jsonl(paths["checkpoint"], record)
                    completed[row["row_index"]] = record
                    outcome = f"ok cases={record['case_clause_count']} tokens={record['completion_tokens_total']}"
                except Exception as exc:
                    append_jsonl(paths["failures"], {
                        **{key: row[key] for key in IDENTITY_FIELDS}, "model": args.model,
                        "failed_at": generation.utc_now(), "error": str(exc),
                        "details": dict(getattr(exc, "details", {})),
                    })
                    failures = load_failures(paths["failures"], rows, args.model)
                    outcome = f"FAILED {type(exc).__name__}: {exc}"
                progress("running")
                print(f"[row {row['row_index']}] {outcome} | "
                      f"attempted={len(completed) + len(failures)}/{len(rows)} "
                      f"successful={len(completed)} failures={len(failures)}", flush=True)
        if len(completed) + len(failures) != len(rows):
            progress("interrupted")
            return 2
        successful = [completed[row["row_index"]] for row in rows if row["row_index"] in completed]
        for record in successful:
            row = all_rows[record["row_index"]]
            generation.validate_record_provenance(record, row)
        outcomes = [completed.get(row["row_index"]) or failures[row["row_index"]] for row in rows]
        generation.atomic_write_json(paths["queries"], successful)
        generation.atomic_write_json(paths["outcomes"], outcomes)
        final_status = "completed" if not failures else "completed_with_generation_failures"
        progress(final_status)
        generation.atomic_write_json(paths["done"], {
            **contract, "status": final_status, "finished_at": generation.utc_now(),
            "failure_policy": "frozen_mbe_v1_case_cap_correction_only_other_validator_failures_zero_v1",
            "code_sha256": generation.sha256_file(Path(__file__)),
            "integrity": {"attempted_row_count": len(outcomes),
                          "successful_query_count": len(successful),
                          "generation_failure_count": len(failures),
                          "all_successes_structured_ok": True,
                          "all_successes_service_tier_flex": all(
                              record["openai"]["service_tier_returned"] == SERVICE_TIER
                              for record in successful)},
            "outputs": {key: str(path) for key, path in paths.items() if key != "done"},
            "sha256": {key: generation.sha256_file(path) for key, path in paths.items() if key != "done"},
        })
        print(f"[done] attempted={len(outcomes)} successful={len(successful)} "
              f"failures={len(failures)} queries={paths['queries']}")
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
