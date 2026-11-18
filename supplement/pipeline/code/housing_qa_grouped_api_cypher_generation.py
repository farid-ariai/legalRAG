#!/usr/bin/env python3
"""Generate Housing QA Cypher through an already-running vLLM API.

The clean Housing QA cohort contains 4,584 rows and 225 distinct
``question_group`` values. Question text is invariant within a group; state is
the varying input.  This runner therefore asks the model for one reusable
Cypher template per group containing the exact literal ``__STATE__``, then
materializes the template for every original dataset row.

The runner is designed for unattended experiments:

* successful groups are appended immediately to a JSONL checkpoint;
* rerunning the same command resumes without regenerating successful groups;
* progress, throughput, ETA, failures, and final integrity checks are persisted;
* no failed request is retried silently;
* the final all-rows file is schema-compatible with
  ``housing_qa_execute_cypher.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Indices" / "KG_Indices" / "HousingQA_cypher"
STATE_PLACEHOLDER = "__STATE__"
RUNNER_VERSION = "grouped_api_v4"
MAX_CASE_CLAUSES = 10
UNESCAPED_WORD_APOSTROPHE = re.compile(r"(?<!\\)(?<=[A-Za-z])'(?=[A-Za-z])")
RAW_CONDITION_NODE_COLLECTION = re.compile(
    r"collect\s*\(\s*(?:distinct\s+)?c\s*\)\s+AS\s+conditions",
    re.IGNORECASE,
)
SERIALISABLE_CONDITION_COLLECTION = """[condition IN collect(DISTINCT c)
   WHERE condition IS NOT NULL |
   {
     number: condition.number,
     description: condition.description
   }
  ] AS conditions"""

sys.path.insert(0, str(Path(__file__).resolve().parent))
from housing_qa_prompt import VERSIONS, extract_cypher, get_prompt  # noqa: E402
from housing_qa_dataset_paths import (  # noqa: E402
    CLEAN_N_GROUPS,
    CLEAN_N_QUESTIONS,
    CLEAN_N_STATES,
    CLEAN_QUESTIONS,
)


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "housing_qa_cypher_template",
        "schema": {
            "type": "object",
            "properties": {"cypher": {"type": "string"}},
            "required": ["cypher"],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=CLEAN_QUESTIONS)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", default="v5", choices=sorted(VERSIONS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument(
        "--group-ids",
        default=None,
        help="Comma-separated question_group ids for a deterministic targeted pilot.",
    )
    parser.add_argument("--out-tag", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-rows", type=int, default=CLEAN_N_QUESTIONS)
    parser.add_argument("--expected-groups", type=int, default=CLEAN_N_GROUPS)
    parser.add_argument("--expected-states", type=int, default=CLEAN_N_STATES)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    required = {"idx", "state", "question", "question_group"}
    for position, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {position} is missing {sorted(missing)}")
    return rows


def group_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        group_id = int(row["question_group"])
        question = str(row["question"]).strip()
        state = str(row["state"]).strip()
        entry = grouped.setdefault(
            group_id,
            {"question_group": group_id, "question": question, "states": set(), "row_count": 0},
        )
        if entry["question"] != question:
            raise ValueError(f"question text varies within question_group {group_id}")
        entry["states"].add(state)
        entry["row_count"] += 1
    return [
        {**entry, "states": sorted(entry["states"])}
        for _, entry in sorted(grouped.items())
    ]


def replace_state(text: str, old_state: str, replacement: str) -> tuple[str, int]:
    return re.subn(re.escape(old_state), replacement, text, flags=re.IGNORECASE)


def build_messages(question: str, prompt_version: str) -> tuple[list[dict[str, str]], str]:
    system, example_state, example_question, example_cypher = get_prompt(prompt_version)
    example_template, replacements = replace_state(
        example_cypher.strip(), example_state.strip(), STATE_PLACEHOLDER
    )
    if replacements < 1:
        raise ValueError(
            f"prompt {prompt_version} example does not contain its example state {example_state!r}"
        )
    template_contract = f"""

REUSABLE STATE-TEMPLATE MODE:
- Generate one query reusable across U.S. jurisdictions.
- In the state restriction, use the exact quoted literal '{STATE_PLACEHOLDER}'.
- The query must contain '{STATE_PLACEHOLDER}' exactly once.
- Do not name or infer a real state.
- Use at most {MAX_CASE_CLAUSES} CASE scoring clauses. Never enumerate synonyms,
  antonyms, generic adjectives, or speculative wording.
- Keep the complete query concise; target fewer than 1,200 output tokens.
- Never return a Neo4j node or relationship object. In particular, never return
  `c` or `collect(c)` directly. Return `conditions` only as a list of plain maps
  containing `number` and `description`, exactly as demonstrated in the example.
- Return exactly one JSON object with one key named `cypher`; its value is the
  complete read-only Cypher query, with no Markdown or commentary.
""".rstrip()
    messages = [
        {"role": "system", "content": system.strip() + template_contract},
        {
            "role": "user",
            "content": f"State: {STATE_PLACEHOLDER}\n\nQuestion: {example_question.strip()}",
        },
        {
            "role": "assistant",
            "content": json.dumps({"cypher": example_template}, ensure_ascii=False),
        },
        {
            "role": "user",
            "content": f"State: {STATE_PLACEHOLDER}\n\nQuestion: {question.strip()}",
        },
    ]
    prompt_fingerprint = hashlib.sha256(
        json.dumps(messages[:3], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return messages, prompt_fingerprint


def strip_string_literals(query: str) -> str:
    return re.sub(r"""(["'])(?:\\.|(?!\1).)*\1""", '""', query, flags=re.DOTALL)


def mask_cypher_literals_and_comments(query: str) -> str:
    """Mask quoted strings and comments with spaces while preserving offsets."""
    masked = list(query)
    index = 0
    quote = None
    line_comment = False
    block_comment = False
    while index < len(query):
        char = query[index]
        following = query[index + 1] if index + 1 < len(query) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            else:
                masked[index] = " "
            index += 1
            continue
        if block_comment:
            masked[index] = " "
            if char == "*" and following == "/":
                masked[index + 1] = " "
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            masked[index] = " "
            if char == "\\" and following:
                masked[index + 1] = " "
                index += 2
            elif char == quote:
                quote = None
                index += 1
            else:
                index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            masked[index] = " "
            index += 1
        elif char == "/" and following == "/":
            masked[index] = masked[index + 1] = " "
            line_comment = True
            index += 2
        elif char == "/" and following == "*":
            masked[index] = masked[index + 1] = " "
            block_comment = True
            index += 2
        else:
            index += 1
    return "".join(masked)


def normalise_word_apostrophes(query: str) -> tuple[str, int]:
    """Escape apostrophes inside words while preserving all other Cypher text."""
    return UNESCAPED_WORD_APOSTROPHE.subn(lambda _: "\\'", query)


def normalise_condition_collections(query: str) -> tuple[str, int]:
    """Replace raw Condition-node collections with JSON-compatible maps."""
    masked = mask_cypher_literals_and_comments(query)
    if not re.search(r"\(\s*c\s*:\s*Condition\s*\)", masked, re.IGNORECASE):
        return query, 0
    matches = list(RAW_CONDITION_NODE_COLLECTION.finditer(masked))
    repaired = query
    for match in reversed(matches):
        repaired = (
            repaired[:match.start()]
            + SERIALISABLE_CONDITION_COLLECTION
            + repaired[match.end():]
        )
    return repaired, len(matches)


def unsafe_returned_graph_values(query: str) -> list[str]:
    """Find bare graph entities or node-collection aliases in the final RETURN."""
    masked = mask_cypher_literals_and_comments(query)
    return_matches = list(re.finditer(r"\bRETURN\b", masked, re.IGNORECASE))
    if not return_matches:
        return []
    tail = masked[return_matches[-1].end():]
    terminator = re.search(
        r"\b(?:ORDER\s+BY|LIMIT|SKIP|UNION)\b",
        tail,
        re.IGNORECASE,
    )
    clause = tail[:terminator.start()] if terminator else tail

    graph_variables = set(
        re.findall(r"\(\s*([A-Za-z_]\w*)\s*:", masked)
    )
    graph_variables.update(
        alias
        for source, alias in re.findall(
            r"head\s*\(\s*collect\s*\(\s*([A-Za-z_]\w*)\s*\)\s*\)"
            r"\s+AS\s+([A-Za-z_]\w*)",
            masked,
            re.IGNORECASE,
        )
        if source in graph_variables
    )
    graph_collection_aliases = {
        alias
        for source, alias in re.findall(
            r"collect\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_]\w*)\s*\)"
            r"\s+AS\s+([A-Za-z_]\w*)",
            masked,
            re.IGNORECASE,
        )
        if source in graph_variables
    }

    safe_clause = clause
    for variable in sorted(graph_variables, key=len, reverse=True):
        escaped = re.escape(variable)
        safe_clause = re.sub(
            rf"\b(?:properties|labels|elementId|type)\s*\(\s*{escaped}\s*\)",
            "",
            safe_clause,
            flags=re.IGNORECASE,
        )
        safe_clause = re.sub(
            rf"\b{escaped}\s*\.\s*[A-Za-z_]\w*",
            "",
            safe_clause,
        )

    unsafe = []
    for variable in sorted(graph_variables | graph_collection_aliases):
        if re.search(rf"(?<![\w.]){re.escape(variable)}(?![\w.])", safe_clause):
            unsafe.append(variable)
    return unsafe


def validate_read_only_cypher(query: str) -> str:
    """Validate the shared safety and shape contract for executable Cypher."""
    body = str(query or "").strip().rstrip(";").strip()
    if not body:
        raise ValueError("empty query")
    if ";" in body:
        raise ValueError("multiple Cypher statements are not allowed")
    bad_apostrophe = UNESCAPED_WORD_APOSTROPHE.search(body)
    if bad_apostrophe:
        context = body[max(0, bad_apostrophe.start() - 30):bad_apostrophe.end() + 30]
        raise ValueError(f"unescaped apostrophe inside Cypher string near {context!r}")
    scrubbed = strip_string_literals(body)
    blocked = re.search(
        r"(?i)\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|"
        r"LOAD\s+CSV|GRANT|DENY|REVOKE|ALTER|RENAME|START\s+DATABASE|"
        r"STOP\s+DATABASE)\b",
        scrubbed,
    )
    if blocked:
        raise ValueError(f"write/admin clause rejected: {blocked.group(0)}")
    if re.search(r"(?i)\bCALL\s+(?!\{)", scrubbed):
        raise ValueError("procedure calls are not allowed")
    if RAW_CONDITION_NODE_COLLECTION.search(mask_cypher_literals_and_comments(body)):
        raise ValueError("conditions must be returned as maps, not raw Neo4j nodes")
    unsafe_graph_values = unsafe_returned_graph_values(body)
    if unsafe_graph_values:
        raise ValueError(
            "final RETURN contains raw graph values: "
            + ", ".join(unsafe_graph_values)
        )
    case_count = len(re.findall(r"(?i)\bCASE\b", scrubbed))
    if case_count > MAX_CASE_CLAUSES:
        raise ValueError(
            f"query has {case_count} CASE clauses; maximum is {MAX_CASE_CLAUSES}"
        )
    for required in (r"\bMATCH\b", r"\bRETURN\b", r"\bLIMIT\s+20\b"):
        if not re.search(required, scrubbed, flags=re.IGNORECASE):
            raise ValueError(f"query is missing required pattern {required}")
    return body


def validate_query_template(query: str) -> str:
    """Validate executable Cypher plus the grouped state-template contract."""
    body = validate_read_only_cypher(query)
    placeholder_count = body.count(STATE_PLACEHOLDER)
    if placeholder_count != 1:
        raise ValueError(
            f"expected exactly one {STATE_PLACEHOLDER} literal, found {placeholder_count}"
        )
    placeholder_literal = rf"['\"]{re.escape(STATE_PLACEHOLDER)}['\"]"
    state_filter = (
        rf"(?i)(?:toLower\(\s*s\.state\s*\)|s\.state)\s*=\s*"
        rf"(?:toLower\(\s*)?{placeholder_literal}\s*\)?"
    )
    if not re.search(state_filter, body):
        raise ValueError(
            f"query must compare s.state to the quoted {STATE_PLACEHOLDER} literal"
        )
    return body


def materialize_state(query_template: str, state: str) -> str:
    if query_template.count(STATE_PLACEHOLDER) != 1:
        raise ValueError("cannot materialize a template without exactly one state placeholder")
    escaped_state = str(state).strip().lower().replace("\\", "\\\\").replace("'", "\\'")
    return query_template.replace(STATE_PLACEHOLDER, escaped_state)


def api_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from exc


def verify_server(api_base: str, model: str) -> dict[str, Any]:
    response = api_json(f"{api_base.rstrip('/')}/models", None, timeout=15)
    available = {entry["id"] for entry in response.get("data", [])}
    if model not in available:
        raise RuntimeError(f"server exposes {sorted(available)}, not requested model {model!r}")
    return response


class GroupGenerationError(RuntimeError):
    """Generation failure carrying safe response diagnostics for checkpoint logs."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def model_chat_template_kwargs(model: str) -> dict[str, bool] | None:
    """Return model-family chat options accepted by the serving tokenizer."""
    if str(model).strip().lower().startswith("mistralai/"):
        return None
    return {"enable_thinking": False}


CASE_CAP_VALIDATION_ERROR = re.compile(
    r"query has \d+ CASE clauses; maximum is \d+",
    re.IGNORECASE,
)


def completion_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    """Build one model-family-compatible structured completion request."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "response_format": RESPONSE_FORMAT,
    }
    chat_template_kwargs = model_chat_template_kwargs(model)
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return payload


def decode_group_response(response: dict[str, Any]) -> dict[str, Any]:
    """Decode, normalise, and validate one structured model response."""
    choice = response["choices"][0]
    message = choice["message"]
    raw_output = message.get("content") or ""
    usage = response.get("usage") or {}
    diagnostics: dict[str, Any] = {
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "raw_character_count": len(raw_output),
        "raw_output": raw_output,
        "raw_head": raw_output[:2000],
        "raw_tail": raw_output[-2000:],
    }
    if choice.get("finish_reason") != "stop":
        raise GroupGenerationError(
            f"generation ended with finish_reason={choice.get('finish_reason')!r}",
            diagnostics,
        )
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        diagnostics["json_error"] = str(exc)
        raise GroupGenerationError(
            f"structured response was invalid JSON: {exc}", diagnostics
        ) from exc
    if not isinstance(parsed, dict) or set(parsed) != {"cypher"}:
        parsed_keys = sorted(parsed) if isinstance(parsed, dict) else []
        diagnostics["parsed_type"] = type(parsed).__name__
        diagnostics["parsed_keys"] = parsed_keys
        raise GroupGenerationError(
            "structured response must be an object containing only 'cypher'",
            diagnostics,
        )
    try:
        raw_query_template = extract_cypher(parsed["cypher"])
        diagnostics.update(
            raw_cypher_character_count=len(raw_query_template),
            raw_cypher=raw_query_template,
            raw_cypher_head=raw_query_template[:4000],
            raw_cypher_tail=raw_query_template[-4000:],
        )
        query_template, apostrophe_repairs = normalise_word_apostrophes(
            raw_query_template
        )
        query_template, condition_collection_repairs = normalise_condition_collections(
            query_template
        )
        diagnostics.update(
            apostrophe_repairs=apostrophe_repairs,
            condition_collection_repairs=condition_collection_repairs,
            normalised_cypher_head=query_template[:4000],
            normalised_cypher_tail=query_template[-4000:],
        )
        query_template = validate_query_template(query_template)
    except Exception as exc:
        diagnostics["validation_error"] = f"{type(exc).__name__}: {exc}"
        raise GroupGenerationError(
            f"generated Cypher failed validation: {exc}", diagnostics
        ) from exc
    return {
        "finish_reason": choice.get("finish_reason"),
        "raw_output": raw_output,
        "raw_query_template": raw_query_template,
        "query_template": query_template,
        "apostrophe_repairs": apostrophe_repairs,
        "condition_collection_repairs": condition_collection_repairs,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "diagnostics": diagnostics,
    }


def correction_messages(
    base_messages: list[dict[str, str]],
    raw_output: str,
    validation_error: str,
) -> tuple[list[dict[str, str]], str] | None:
    """Build one bounded correction turn for a CASE-cap validation failure."""
    match = CASE_CAP_VALIDATION_ERROR.search(validation_error)
    if not match:
        return None
    instruction = (
        f"Your previous response was rejected because {match.group(0)}. "
        "Return the complete corrected JSON object again. Count every CASE keyword "
        "and use at most 10 in total; combine or remove the least important scoring "
        "signal. Preserve the exact __STATE__ placeholder, read-only behaviour, "
        "plain-map Condition output, deterministic ordering, and LIMIT 20. Return "
        "only one JSON object with the single key 'cypher'."
    )
    return [
        *base_messages,
        {"role": "assistant", "content": raw_output},
        {"role": "user", "content": instruction},
    ], instruction


def generation_attempt_record(
    attempt: int,
    status: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return the auditable subset retained for each model attempt."""
    return {
        "attempt": attempt,
        "status": status,
        "finish_reason": diagnostics.get("finish_reason"),
        "completion_tokens": diagnostics.get("completion_tokens"),
        "prompt_tokens": diagnostics.get("prompt_tokens"),
        "raw_output": diagnostics.get("raw_output"),
        "raw_cypher": diagnostics.get("raw_cypher"),
        "validation_error": diagnostics.get("validation_error"),
    }


def generate_group(
    group: dict[str, Any],
    *,
    api_base: str,
    model: str,
    prompt_version: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    base_messages, prompt_fingerprint = build_messages(
        group["question"], prompt_version
    )
    messages = base_messages
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    correction_instruction = None
    decoded = None
    for attempt_number in (1, 2):
        payload = completion_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        response = api_json(
            f"{api_base.rstrip('/')}/chat/completions", payload, timeout=timeout
        )
        try:
            decoded = decode_group_response(response)
        except GroupGenerationError as exc:
            attempts.append(
                generation_attempt_record(attempt_number, "failed", exc.details)
            )
            correction = (
                correction_messages(
                    base_messages,
                    str(exc.details.get("raw_output") or ""),
                    str(exc.details.get("validation_error") or ""),
                )
                if attempt_number == 1
                else None
            )
            if correction is not None:
                messages, correction_instruction = correction
                continue
            final_details = dict(exc.details)
            final_details["generation_attempts"] = attempts
            if correction_instruction is not None:
                final_details["correction_instruction_sha256"] = hashlib.sha256(
                    correction_instruction.encode()
                ).hexdigest()
            raise GroupGenerationError(str(exc), final_details) from exc
        attempts.append(
            generation_attempt_record(
                attempt_number,
                "passed",
                decoded["diagnostics"],
            )
        )
        break
    if decoded is None:
        raise AssertionError("generation loop ended without a decoded response")

    raw_query_template = decoded["raw_query_template"]
    query_template = decoded["query_template"]
    apostrophe_repairs = decoded["apostrophe_repairs"]
    condition_collection_repairs = decoded["condition_collection_repairs"]
    completion_tokens = decoded["completion_tokens"]
    prompt_tokens = decoded["prompt_tokens"]
    return {
        "question_group": int(group["question_group"]),
        "question": group["question"],
        "states": group["states"],
        "row_count": int(group["row_count"]),
        "state_placeholder": STATE_PLACEHOLDER,
        "generated_cypher_query_template_raw": raw_query_template,
        "generated_cypher_query_template": query_template,
        "generated_cypher_query": query_template,
        "raw_output": decoded["raw_output"],
        "prompt_version": prompt_version,
        "prompt_fingerprint_sha256": prompt_fingerprint,
        "model": model,
        "finish_reason": decoded["finish_reason"],
        "structured_ok": True,
        "placeholder_count": query_template.count(STATE_PLACEHOLDER),
        "case_clause_count": len(
            re.findall(r"(?i)\bCASE\b", strip_string_literals(query_template))
        ),
        "apostrophe_repairs": apostrophe_repairs,
        "condition_collection_repairs": condition_collection_repairs,
        "generation_mode": (
            "vllm_api_chat_json_schema"
            if len(attempts) == 1
            else "vllm_api_chat_json_schema_bounded_case_correction"
        ),
        "generation_attempt_count": len(attempts),
        "generation_attempts": attempts,
        "correction_instruction_sha256": (
            hashlib.sha256(correction_instruction.encode()).hexdigest()
            if correction_instruction is not None
            else None
        ),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens_total": sum(
            int(attempt["completion_tokens"] or 0) for attempt in attempts
        ),
        "prompt_tokens_total": sum(
            int(attempt["prompt_tokens"] or 0) for attempt in attempts
        ),
        "seconds": round(time.monotonic() - started, 3),
        "completed_at": utc_now(),
    }


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            completed[int(record["question_group"])] = record
        except Exception as exc:
            raise ValueError(f"invalid checkpoint line {line_number} in {path}: {exc}") from exc
    return completed


def progress_payload(
    *,
    status: str,
    started_at: str,
    started_monotonic: float,
    total: int,
    completed: int,
    failed: int,
    model: str,
    prompt_version: str,
    workers: int,
    last_group: int | None = None,
    last_error: str | None = None,
) -> dict[str, Any]:
    elapsed = max(0.001, time.monotonic() - started_monotonic)
    rate = completed / elapsed
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0 else None
    return {
        "runner_version": RUNNER_VERSION,
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "model": model,
        "prompt_version": prompt_version,
        "workers": workers,
        "groups_total": total,
        "groups_completed": completed,
        "groups_failed_this_run": failed,
        "groups_remaining": remaining,
        "groups_per_hour": round(rate * 3600, 3),
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "last_group": last_group,
        "last_error": last_error,
        "pid": os.getpid(),
    }


def materialize_all_rows(
    rows: list[dict[str, Any]], completed: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in sorted(rows, key=lambda item: int(item["idx"])):
        group = completed[int(row["question_group"])]
        query = materialize_state(group["generated_cypher_query_template"], row["state"])
        output.append(
            {
                "idx": int(row["idx"]),
                "state": row["state"],
                "question": row["question"],
                "question_group": int(row["question_group"]),
                "generated_cypher_query": query,
                "generated_cypher_query_template_raw": group.get(
                    "generated_cypher_query_template_raw",
                    group["generated_cypher_query_template"],
                ),
                "generated_cypher_query_template": group["generated_cypher_query_template"],
                "raw_output": group["raw_output"],
                "prompt_version": group["prompt_version"],
                "model": group["model"],
                "finish_reason": group["finish_reason"],
                "structured_ok": group["structured_ok"],
                "generation_mode": group["generation_mode"],
                "state_substituted": True,
                "state_placeholder": STATE_PLACEHOLDER,
                "apostrophe_repairs": int(group.get("apostrophe_repairs", 0)),
                "condition_collection_repairs": int(
                    group.get("condition_collection_repairs", 0)
                ),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    rows = load_questions(args.questions)
    groups = group_questions(rows)
    states = {str(row["state"]).strip().lower() for row in rows}
    if args.limit_groups is not None and args.group_ids:
        raise ValueError("--limit-groups and --group-ids are mutually exclusive")
    full_run = args.limit_groups is None and not args.group_ids
    if full_run:
        observed = (len(rows), len(groups), len(states))
        expected = (args.expected_rows, args.expected_groups, args.expected_states)
        if observed != expected:
            raise ValueError(
                f"dataset shape changed: observed rows/groups/states={observed}, expected={expected}"
            )
    elif args.limit_groups is not None:
        groups = groups[: args.limit_groups]
        kept = {group["question_group"] for group in groups}
        rows = [row for row in rows if int(row["question_group"]) in kept]
    else:
        selected = {int(value.strip()) for value in args.group_ids.split(",") if value.strip()}
        known = {int(group["question_group"]) for group in groups}
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown --group-ids: {sorted(unknown)}")
        groups = [group for group in groups if int(group["question_group"]) in selected]
        rows = [row for row in rows if int(row["question_group"]) in selected]

    verify_server(args.api_base, args.model)

    tag = args.out_tag or f"clean__{safe_name(args.model)}"
    prefix = OUT_DIR / f"housing_qa_grouped_api__{args.prompt_version}__{tag}"
    checkpoint_path = Path(f"{prefix}__checkpoint.jsonl")
    failures_path = Path(f"{prefix}__failures.jsonl")
    progress_path = Path(f"{prefix}__progress.json")
    manifest_path = Path(f"{prefix}__manifest.json")
    by_group_path = Path(f"{prefix}__by_group__queries.json")
    all_rows_path = Path(f"{prefix}__all_rows__queries.json")
    done_path = Path(f"{prefix}__done.json")

    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"{checkpoint_path} exists; use --resume or choose a different --out-tag"
        )
    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    expected_group_ids = {int(group["question_group"]) for group in groups}
    unknown = set(completed) - expected_group_ids
    if unknown:
        raise ValueError(f"checkpoint contains groups not in this run: {sorted(unknown)}")

    pending = [group for group in groups if group["question_group"] not in completed]
    started_at = utc_now()
    started_monotonic = time.monotonic()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    manifest = {
        "runner_version": RUNNER_VERSION,
        "status": "running",
        "started_at": started_at,
        "command": command,
        "working_directory": str(Path.cwd()),
        "questions": str(args.questions.resolve()),
        "api_base": args.api_base,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "request_timeout": args.request_timeout,
        "rows": len(rows),
        "groups": len(groups),
        "states": len({str(row["state"]).strip().lower() for row in rows}),
        "resumed_groups": len(completed),
        "output_prefix": str(prefix),
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        progress_path,
        progress_payload(
            status="running",
            started_at=started_at,
            started_monotonic=started_monotonic,
            total=len(groups),
            completed=len(completed),
            failed=0,
            model=args.model,
            prompt_version=args.prompt_version,
            workers=args.workers,
        ),
    )
    print(
        f"[start] rows={len(rows)} groups={len(groups)} states={manifest['states']} "
        f"pending={len(pending)} workers={args.workers} model={args.model}",
        flush=True,
    )

    failures = 0
    last_error = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_group = {
            executor.submit(
                generate_group,
                group,
                api_base=args.api_base,
                model=args.model,
                prompt_version=args.prompt_version,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
            ): group
            for group in pending
        }
        for future in concurrent.futures.as_completed(future_to_group):
            group = future_to_group[future]
            group_id = int(group["question_group"])
            try:
                record = future.result()
                append_jsonl(checkpoint_path, record)
                completed[group_id] = record
                outcome = (
                    f"ok {record['seconds']:.1f}s "
                    f"tokens={record.get('completion_tokens')}"
                )
                last_error = None
            except Exception as exc:
                failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
                failure_record = {
                    "question_group": group_id,
                    "question": group["question"],
                    "error": last_error,
                    "failed_at": utc_now(),
                }
                details = getattr(exc, "details", None)
                if isinstance(details, dict):
                    failure_record["response_diagnostics"] = details
                append_jsonl(
                    failures_path,
                    failure_record,
                )
                outcome = f"FAILED {last_error}"

            progress = progress_payload(
                status="running",
                started_at=started_at,
                started_monotonic=started_monotonic,
                total=len(groups),
                completed=len(completed),
                failed=failures,
                model=args.model,
                prompt_version=args.prompt_version,
                workers=args.workers,
                last_group=group_id,
                last_error=last_error,
            )
            atomic_write_json(progress_path, progress)
            eta = progress["eta_seconds"]
            eta_text = "n/a" if eta is None else f"{eta / 3600:.2f}h"
            print(
                f"[group {group_id}] {outcome} | "
                f"{len(completed)}/{len(groups)} complete, failures={failures}, ETA={eta_text}",
                flush=True,
            )

    missing = sorted(expected_group_ids - set(completed))
    if missing:
        status = "completed_with_failures"
        manifest.update(
            status=status,
            finished_at=utc_now(),
            groups_completed=len(completed),
            groups_missing=missing,
            failures_this_run=failures,
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            progress_path,
            progress_payload(
                status=status,
                started_at=started_at,
                started_monotonic=started_monotonic,
                total=len(groups),
                completed=len(completed),
                failed=failures,
                model=args.model,
                prompt_version=args.prompt_version,
                workers=args.workers,
                last_error=last_error,
            ),
        )
        print(f"[incomplete] missing groups: {missing}", flush=True)
        return 2

    by_group = [completed[group_id] for group_id in sorted(expected_group_ids)]
    all_rows = materialize_all_rows(rows, completed)
    atomic_write_json(by_group_path, by_group)
    atomic_write_json(all_rows_path, all_rows)

    integrity = {
        "group_count": len(by_group),
        "row_count": len(all_rows),
        "state_count": len({str(row["state"]).strip().lower() for row in all_rows}),
        "unique_idx_count": len({int(row["idx"]) for row in all_rows}),
        "all_structured_ok": all(row["structured_ok"] for row in by_group),
        "all_templates_have_one_placeholder": all(
            row["generated_cypher_query_template"].count(STATE_PLACEHOLDER) == 1
            for row in by_group
        ),
        "materialized_queries_have_no_placeholder": all(
            STATE_PLACEHOLDER not in row["generated_cypher_query"] for row in all_rows
        ),
    }
    expected_integrity = {
        "group_count": len(groups),
        "row_count": len(rows),
        "state_count": len({str(row["state"]).strip().lower() for row in rows}),
        "unique_idx_count": len(rows),
        "all_structured_ok": True,
        "all_templates_have_one_placeholder": True,
        "materialized_queries_have_no_placeholder": True,
    }
    if integrity != expected_integrity:
        raise RuntimeError(
            f"final integrity mismatch: observed={integrity}, expected={expected_integrity}"
        )

    finished_at = utc_now()
    done = {
        "runner_version": RUNNER_VERSION,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "model": args.model,
        "prompt_version": args.prompt_version,
        "integrity": integrity,
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "by_group": str(by_group_path),
            "all_rows": str(all_rows_path),
            "progress": str(progress_path),
            "manifest": str(manifest_path),
        },
        "sha256": {
            "by_group": sha256_file(by_group_path),
            "all_rows": sha256_file(all_rows_path),
        },
    }
    atomic_write_json(done_path, done)
    manifest.update(done)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        progress_path,
        progress_payload(
            status="completed",
            started_at=started_at,
            started_monotonic=started_monotonic,
            total=len(groups),
            completed=len(completed),
            failed=failures,
            model=args.model,
            prompt_version=args.prompt_version,
            workers=args.workers,
        ),
    )
    print(
        f"[done] {len(by_group)} templates -> {len(all_rows)} rows; "
        f"states={integrity['state_count']} output={all_rows_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
