#!/usr/bin/env python3
"""Generate full-scope MBE queries through an existing vLLM API, with provenance.

Only question text and available choices reach the model. Successful rows are
checkpointed durably; a CASE-cap error permits exactly one recorded correction.
No other failure is retried within a run. Resume validates the immutable run
contract and every checkpoint before attempting only missing rows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mbe_prompt_v1 import (
    CORRECTION_TEMPLATE, EXAMPLE_CYPHER, EXAMPLE_USER, MAX_CASE_CLAUSES,
    PROMPT_VERSION, SYSTEM_PROMPT,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Indices" / "KG_Indices" / "MBE_cypher"
RUNNER_VERSION = "mbe_api_v1"
EXPECTED_ROWS = 1071
IDENTITY_FIELDS = ("row_index", "idx", "question", "choices", "input_sha256")
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "mbe_cypher_query", "strict": True,
    "schema": {"type": "object", "properties": {"cypher_query": {"type": "string"}},
               "required": ["cypher_query"], "additionalProperties": False},
}}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Read every source row without exposing answer or metadata columns."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"idx", "prompt", "question", "choice_a", "choice_b", "choice_c", "choice_d"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("question CSV is missing required input columns")
        output, seen = [], set()
        for position, source in enumerate(reader):
            idx = str(source.get("idx") or "").strip()
            stem = str(source.get("question") or "").strip()
            if not idx or idx in seen or not stem:
                raise ValueError(f"invalid or duplicate QA identity/question at row {position}")
            seen.add(idx)
            question = "\n\n".join(text for text in (
                str(source.get("prompt") or "").strip(), stem) if text)
            choices, warnings = [], []
            for label in "ABCD":
                value = str(source.get(f"choice_{label.lower()}") or "").strip()
                if value:
                    choices.append(f"{label}. {value}")
                else:
                    warnings.append(f"missing_choice_{label.lower()}")
            if len(choices) < 2:
                raise ValueError(f"fewer than two available choices at row {position}")
            safe = {"question": question, "choices": "\n".join(choices)}
            output.append({"row_index": position, "idx": idx, **safe,
                           "input_sha256": json_sha256(safe), "input_warnings": warnings})
    if not output:
        raise ValueError("question CSV is empty")
    return output


def build_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """Build the frozen shared prompt, excluding all non-whitelisted row fields."""
    base = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_USER},
            {"role": "assistant", "content": json.dumps({"cypher_query": EXAMPLE_CYPHER})}]
    fingerprint = json_sha256({"version": PROMPT_VERSION, "base_messages": base,
                              "correction_template": CORRECTION_TEMPLATE,
                              "response_format": RESPONSE_FORMAT})
    return [*base, {"role": "user", "content": (
        f"Question:\n{row['question']}\n\nAnswer choices:\n{row['choices']}")}], fingerprint


def mask_cypher(query: str) -> str:
    """Mask literals/comments without changing offsets; reject malformed quoting."""
    out, index = list(query), 0
    while index < len(query):
        if query.startswith("//", index):
            end = query.find("\n", index)
            end = len(query) if end < 0 else end
        elif query.startswith("/*", index):
            end = query.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated Cypher comment")
            end += 2
        elif query[index] in "\"'":
            quote, end = query[index], index + 1
            while end < len(query):
                if query[end] == "\\":
                    end += 2
                elif query[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
            else:
                raise ValueError("unterminated Cypher string")
        else:
            index += 1
            continue
        out[index:end] = " " * (end - index)
        index = end
    return "".join(out)


def normalise_word_apostrophes(query: str) -> tuple[str, int]:
    """Escape only unescaped inner-word apostrophes in single-quoted strings."""
    output, index, quote, repairs = [], 0, None, 0
    while index < len(query):
        char = query[index]
        if quote is None and (query.startswith("//", index) or query.startswith("/*", index)):
            end = query.find("\n", index) if query.startswith("//", index) else query.find("*/", index + 2)
            end = len(query) if end < 0 else end + (2 if query.startswith("/*", index) else 0)
            output.append(query[index:end])
            index = end
            continue
        if quote is not None and char == "\\" and index + 1 < len(query):
            output.append(query[index:index + 2])
            index += 2
            continue
        if char in "\"'":
            if quote is None:
                quote = char
            elif char == quote:
                if (quote == "'" and index > 0 and index + 1 < len(query)
                        and query[index - 1].isalpha() and query[index + 1].isalpha()):
                    output.append("\\")
                    repairs += 1
                else:
                    quote = None
        output.append(char)
        index += 1
    return "".join(output), repairs


def case_clause_count(query: str) -> int:
    return len(re.findall(r"\bCASE\b", mask_cypher(query), re.IGNORECASE))


def split_projection(text: str) -> list[str]:
    masked, depth, start, result = mask_cypher(text), 0, 0, []
    for index, char in enumerate(masked):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
        if depth < 0:
            raise ValueError("unbalanced projection delimiters")
    if depth != 0:
        raise ValueError("unbalanced projection delimiters")
    return [*result, text[start:].strip()]


def top_level_clauses(query: str) -> list[tuple[str, str]]:
    """Find main-query clauses, ignoring scoring subqueries and expressions."""
    masked, stack, positions = mask_cypher(query), [], []
    keywords = re.compile(r"\b(OPTIONAL\s+MATCH|ORDER\s+BY|MATCH|WITH|WHERE|RETURN|LIMIT|SKIP|UNWIND)\b", re.I)
    index = 0
    while index < len(masked):
        char = masked[index]
        if not stack:
            match = keywords.match(masked, index)
            if match:
                positions.append((" ".join(match.group(0).upper().split()), index, match.end()))
                index = match.end()
                continue
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[char]:
                raise ValueError("unbalanced Cypher delimiters")
        index += 1
    if stack:
        raise ValueError("unbalanced Cypher delimiters")
    return [(name, query[end:positions[i + 1][1] if i + 1 < len(positions) else len(query)].strip())
            for i, (name, _, end) in enumerate(positions)]


def validate_full_graph_scope(query: str) -> None:
    """Allow only the full Passage match and post-ranking optional Elements.

    The only main-query WHERE is score > 0. Area/type matches may occur inside
    EXISTS scoring expressions, but cannot remove candidates from the main
    stream. A main AreaOfLaw MATCH or extra predicate is therefore rejected.
    """
    clauses = top_level_clauses(query)
    matches = [(name, text) for name, text in clauses if name in {"MATCH", "OPTIONAL MATCH"}]
    required = [
        ("MATCH", r"\(\s*p\s*:\s*Passage\s*\)\s*-\s*\[:\s*CONTAINS_DOCTRINE\s*\]\s*->\s*\(\s*d\s*:\s*Doctrine\s*\)"),
        ("OPTIONAL MATCH", r"\(\s*best\s*\)\s*-\s*\[:\s*HAS_ELEMENT\s*\]\s*->\s*\(\s*e\s*:\s*Element\s*\)"),
    ]
    if len(matches) != len(required) or any(
            name != expected_name or not re.fullmatch(pattern, text, re.I)
            for (name, text), (expected_name, pattern) in zip(matches, required)):
        raise ValueError("full-graph scope requires only Passage/Doctrine MATCH and optional best-doctrine Elements")
    predicates = [text for name, text in clauses if name == "WHERE"]
    if len(predicates) != 1 or not re.fullmatch(r"score\s*>\s*0", predicates[0], re.I):
        raise ValueError("the only main-query filter permitted is WHERE score > 0")
    if any(name in {"SKIP", "UNWIND"} for name, _ in clauses):
        raise ValueError("SKIP/UNWIND cannot alter full-graph candidate scope")
    if sum(name == "LIMIT" for name, _ in clauses) != 1:
        raise ValueError("only the final LIMIT 20 is permitted")


def validate_best_doctrine_ordering(query: str) -> None:
    """Require one deterministic highest-scoring doctrine per passage.

    Selecting head(collect(d)) is only meaningful after descending score and
    stable doctrine-ID ordering. Grouping this collection by d (or any other
    extra key) would leave several rows per passage, so the entire non-scoring
    suffix is validated rather than merely searching for aggregation tokens.
    """
    clauses = top_level_clauses(query)
    filter_position = next(index for index, (name, _) in enumerate(clauses) if name == "WHERE")
    prefix = clauses[:filter_position]
    if [name for name, _ in prefix] != ["MATCH", "WITH", "WITH"] or not re.match(r"^MATCH\b", query, re.I):
        raise ValueError("main query must start with MATCH, field normalisation, then score projection")
    normalisation = split_projection(prefix[1][1])
    expected_normalisation = [r"p", r"d"] + [
        rf"toLower\s*\(\s*coalesce\s*\(\s*{node}\.{property_name}\s*,\s*(?:''|\"\")\s*\)\s*\)\s+AS\s+{alias}"
        for node, property_name, alias in (("d", "name", "dn"), ("d", "definition", "dd"), ("p", "summary", "ps"))
    ]
    if len(normalisation) != 5 or any(not re.fullmatch(pattern, field, re.I)
            for field, pattern in zip(normalisation, expected_normalisation)):
        raise ValueError("field-normalisation WITH must preserve p,d and only define dn,dd,ps")
    scoring = split_projection(prefix[2][1])
    if (len(scoring) != 3 or [field.lower() for field in scoring[:2]] != ["p", "d"]
            or not re.fullmatch(r".+\s+AS\s+score", scoring[2], re.I | re.S)):
        raise ValueError("score projection must preserve p,d and only add the CASE score")
    suffix = clauses[filter_position:]
    required_names = ["WHERE", "WITH", "ORDER BY", "WITH", "OPTIONAL MATCH", "RETURN", "ORDER BY", "LIMIT"]
    if [name for name, _ in suffix] != required_names:
        raise ValueError("best-doctrine ordering/aggregation must precede optional Elements and final RETURN")
    if not re.fullmatch(r"p\s*,\s*d\s*,\s*score", suffix[1][1], re.I):
        raise ValueError("pre-selection projection must be exactly WITH p, d, score")
    if not re.fullmatch(r"score\s+DESC\s*,\s*d\.uid\s+ASC", suffix[2][1], re.I):
        raise ValueError("best doctrine requires ORDER BY score DESC, d.uid ASC")
    aggregation = (r"p\s*,\s*max\s*\(\s*score\s*\)\s+AS\s+score\s*,\s*"
                   r"head\s*\(\s*collect\s*\(\s*d\s*\)\s*\)\s+AS\s+best")
    if not re.fullmatch(aggregation, suffix[3][1], re.I):
        raise ValueError("best-doctrine aggregation must group only by p: WITH p, max(score) AS score, head(collect(d)) AS best")
    if any(name == "OPTIONAL MATCH" for name, _ in clauses[:filter_position]):
        raise ValueError("optional Elements must be matched only after selecting the best doctrine")


def validate_cypher(query: str) -> str:
    """Conservatively validate the dedicated read-only MBE retrieval contract."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("empty or non-string Cypher")
    body = query.strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    masked = mask_cypher(body)
    if ";" in masked or "`" in masked or "$" in masked:
        raise ValueError("multiple statements, backtick identifiers and parameters are prohibited")
    blocked = re.search(r"\b(CREATE|INSERT|MERGE|DELETE|DETACH|NODETACH|SET|REMOVE|DROP|FOREACH|LOAD|"
                        r"GRANT|DENY|REVOKE|ALTER|RENAME|START|STOP|CALL|UNION|USE|"
                        r"SHOW|TERMINATE|FINISH|FILTER|LET|YIELD|CYPHER|EXPLAIN|PROFILE)\b", masked, re.I)
    if blocked:
        raise ValueError(f"write/admin/procedure clause rejected: {blocked.group(0)}")
    if "__STATE__" in body or re.search(r"\b(?:p|d)\s*\.\s*(?:state|source)\b", masked, re.I):
        raise ValueError("state/source restrictions are prohibited")
    for pattern in (r"\(\s*p\s*:\s*Passage\s*\)", r"\(\s*d\s*:\s*Doctrine\s*\)",
                    r"\[:\s*CONTAINS_DOCTRINE\s*\]", r"\[:\s*HAS_ELEMENT\s*\]",
                    r"\bhead\s*\(\s*collect\s*\(\s*d\s*\)\s*\)\s+AS\s+best\b"):
        if not re.search(pattern, masked, re.I):
            raise ValueError(f"missing MBE query structure: {pattern}")
    if not re.search(r"ORDER\s+BY\s+score\s+DESC\s*,\s*(?:p\.)?row_index\s+ASC\s+LIMIT\s+20\s*$", masked, re.I):
        raise ValueError("query must end with deterministic ORDER BY score DESC, row_index ASC LIMIT 20")
    returns = list(re.finditer(r"\bRETURN\b", masked, re.I))
    if len(returns) != 1:
        raise ValueError("exactly one final RETURN is required")
    end = re.search(r"\bORDER\s+BY\b", masked[returns[0].end():], re.I)
    clause = body[returns[0].end():returns[0].end() + end.start()]
    fields = split_projection(clause)
    if len(fields) != 5:
        raise ValueError("RETURN must have exactly five fields")
    checks = (r"p\.row_index\s+AS\s+row_index", r"p\.summary\s+AS\s+source_summary",
              r"best\.definition\s+AS\s+doctrine_definition", None, r"score(?:\s+AS\s+score)?")
    for field, pattern in zip(fields, checks):
        if pattern and not re.fullmatch(pattern, field, re.I):
            raise ValueError("RETURN scalar fields must be row_index, source_summary, doctrine_definition and score")
    element_map = r"\{\s*element_number\s*:\s*e\.number\s*,\s*description\s*:\s*e\.description\s*\}"
    if not re.fullmatch(r"collect\s*\(\s*DISTINCT\s*" + element_map + r"\s*\)\s+AS\s+elements", fields[3], re.I):
        raise ValueError("elements must use collect(DISTINCT {element_number: e.number, description: e.description}), never raw nodes")
    before_return = masked[:returns[0].start()]
    if re.search(r"\bp\s*\.\s*row_index\b", before_return, re.I):
        raise ValueError("passage-ID restrictions are prohibited")
    validate_full_graph_scope(body)
    validate_best_doctrine_ordering(body)
    count = case_clause_count(body)
    if count > MAX_CASE_CLAUSES:
        raise ValueError(f"query has {count} CASE clauses; maximum is {MAX_CASE_CLAUSES}")
    if count == 0 or not re.search(r"\bWHERE\s+score\s*>\s*0\b", masked, re.I):
        raise ValueError("positive CASE-scored retrieval is required")
    return body


class GenerationError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def completion_payload(model: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    payload = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1.0,
               "max_tokens": max_tokens, "response_format": RESPONSE_FORMAT}
    if not model.lower().startswith("mistralai/"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def api_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Retain only the vLLM error envelope, never headers or echoed requests.
        detail = ""
        try:
            envelope = json.loads(exc.read(65536))
            error = envelope.get("error", {}) if isinstance(envelope, dict) else {}
            if isinstance(error, dict):
                error_type, message = error.get("type"), error.get("message")
                values = [value for value in (error_type, message) if isinstance(value, str)]
                detail = ": ".join(values)
                detail = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", detail)
                detail = re.sub(r"\b(?:sk-|hf_)[A-Za-z0-9_-]+", "[REDACTED]", detail)
                detail = re.sub(r"(?i)(?:api[_ -]?key|password|authorization)\s*[=:]\s*[^\s,;]+",
                                "credential=[REDACTED]", detail)
                detail = detail[:2000]
        except (ValueError, TypeError, OSError):
            pass
        finally:
            exc.close()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"model API returned HTTP {exc.code}{suffix}") from exc


def decode_response(response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    raw = (choice.get("message") or {}).get("content") or ""
    diagnostic = {"raw_output": raw, "finish_reason": choice.get("finish_reason"),
                  "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens"),
                  "completion_tokens": (response.get("usage") or {}).get("completion_tokens")}

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key in structured response")
            value[key] = item
        return value

    try:
        if choice.get("finish_reason") != "stop":
            raise ValueError("completion did not finish normally")
        parsed = json.loads(raw, object_pairs_hook=unique_keys)
        if not isinstance(parsed, dict) or set(parsed) != {"cypher_query"} or not isinstance(parsed["cypher_query"], str):
            raise ValueError("response must contain only the string key cypher_query")
        diagnostic["raw_cypher"] = parsed["cypher_query"]
        normalised, repairs = normalise_word_apostrophes(parsed["cypher_query"])
        canonical = validate_cypher(normalised)
    except (ValueError, TypeError) as exc:
        diagnostic["validation_error"] = str(exc)
        raise GenerationError(str(exc), diagnostic) from exc
    return {**diagnostic, "cypher_query": canonical, "apostrophe_repairs": repairs}


def generate_question(row: dict[str, Any], *, api_base: str, model: str,
                      max_tokens: int = 2000, timeout: float = 900.0) -> dict[str, Any]:
    base, fingerprint = build_messages(row)
    messages, attempts, started = base, [], time.monotonic()
    for number in (1, 2):
        try:
            response = api_json(f"{api_base.rstrip('/')}/chat/completions",
                                completion_payload(model, messages, max_tokens), timeout)
            decoded = decode_response(response)
        except Exception as exc:
            details = dict(getattr(exc, "details", {}))
            details.setdefault("validation_error", f"{type(exc).__name__}: {exc}")
            attempts.append({"attempt": number, "status": "failed", **details})
            match = re.fullmatch(r"query has (\d+) CASE clauses; maximum is 10", details["validation_error"])
            if number == 1 and match:
                instruction = CORRECTION_TEMPLATE.format(count=match.group(1))
                messages = [*base, {"role": "assistant", "content": details["raw_output"]},
                            {"role": "user", "content": instruction}]
                continue
            raise GenerationError(str(exc), {"generation_attempts": attempts}) from exc
        attempts.append({"attempt": number, "status": "passed", **decoded})
        break
    raw_query = decoded["raw_cypher"]
    record = {**{key: row[key] for key in IDENTITY_FIELDS},
              "input_warnings": row.get("input_warnings", []),
              "model": model, "prompt_version": PROMPT_VERSION,
              "prompt_fingerprint_sha256": fingerprint,
              "cypher_query": decoded["cypher_query"], "raw_cypher": raw_query,
              "raw_output": decoded["raw_output"], "structured_ok": True,
              "finish_reason": decoded["finish_reason"],
              "case_clause_count": case_clause_count(decoded["cypher_query"]),
              "apostrophe_repairs": decoded["apostrophe_repairs"],
              "normalisations": {"rule": "single_quoted_inner_word_apostrophe_v1",
                  "count": decoded["apostrophe_repairs"],
                  "raw_sha256": hashlib.sha256(raw_query.encode()).hexdigest(),
                  "canonical_sha256": hashlib.sha256(decoded["cypher_query"].encode()).hexdigest()},
              "generation_attempt_count": len(attempts), "generation_attempts": attempts,
              "completion_tokens_total": sum(int(a.get("completion_tokens") or 0) for a in attempts),
              "seconds": round(time.monotonic() - started, 3), "completed_at": utc_now()}
    validate_record_provenance(record, row)
    return record


def validate_record_provenance(record: dict[str, Any], expected_row: dict[str, Any] | None = None) -> None:
    """Reconstruct accepted output and every bounded attempt from recorded bytes."""
    if expected_row and any(record.get(key) != expected_row[key] for key in IDENTITY_FIELDS):
        raise ValueError("checkpoint input identity mismatch")
    if expected_row and record.get("input_warnings") != expected_row.get("input_warnings", []):
        raise ValueError("checkpoint input warnings mismatch")
    if type(record.get("row_index")) is not int or not isinstance(record.get("idx"), str):
        raise ValueError("invalid QA identity types")
    if record.get("input_sha256") != json_sha256({k: record[k] for k in ("question", "choices")}):
        raise ValueError("input fingerprint mismatch")
    if record.get("prompt_version") != PROMPT_VERSION or record.get("prompt_fingerprint_sha256") != build_messages(record)[1]:
        raise ValueError("prompt fingerprint mismatch")
    if record.get("structured_ok") is not True or record.get("finish_reason") != "stop":
        raise ValueError("accepted record is not a normal structured completion")
    attempts = record.get("generation_attempts")
    if not isinstance(attempts, list) or len(attempts) not in (1, 2) or record.get("generation_attempt_count") != len(attempts):
        raise ValueError("invalid bounded attempt ledger")
    for position, attempt in enumerate(attempts, start=1):
        if attempt.get("attempt") != position:
            raise ValueError("non-sequential attempt ledger")
        is_last = position == len(attempts)
        response = {"choices": [{"finish_reason": attempt.get("finish_reason"),
                                 "message": {"content": attempt.get("raw_output")}}]}
        try:
            decoded = decode_response(response)
        except GenerationError as exc:
            if is_last or attempt.get("status") != "failed" or not re.fullmatch(
                    r"query has \d+ CASE clauses; maximum is 10", str(exc)):
                raise ValueError("unreconstructable rejected attempt") from exc
            if attempt.get("validation_error") != str(exc) or attempt.get("raw_cypher") != exc.details.get("raw_cypher"):
                raise ValueError("rejected-attempt diagnostics mismatch")
        else:
            if not is_last or attempt.get("status") != "passed":
                raise ValueError("unexpected successful earlier attempt")
            if any(attempt.get(key) != decoded[key] for key in (
                    "raw_output", "raw_cypher", "cypher_query", "apostrophe_repairs")):
                raise ValueError("accepted-attempt diagnostics mismatch")
    if any(record.get(key) != decoded[key] for key in ("cypher_query", "raw_cypher", "raw_output", "apostrophe_repairs")):
        raise ValueError("accepted output differs from raw response")
    if record.get("case_clause_count") != case_clause_count(record["cypher_query"]):
        raise ValueError("CASE count mismatch")
    expected = {"rule": "single_quoted_inner_word_apostrophe_v1", "count": decoded["apostrophe_repairs"],
                "raw_sha256": hashlib.sha256(record["raw_cypher"].encode()).hexdigest(),
                "canonical_sha256": hashlib.sha256(record["cypher_query"].encode()).hexdigest()}
    if record.get("normalisations") != expected:
        raise ValueError("normalisation provenance mismatch")


def load_checkpoint(path: Path, rows: list[dict[str, Any]], model: str) -> dict[int, dict[str, Any]]:
    expected, completed, ids = {row["row_index"]: row for row in rows}, {}, set()
    if not path.exists():
        return completed
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("row_index")
        if type(key) is not int or key not in expected or key in completed or record.get("idx") in ids:
            raise ValueError(f"duplicate or unknown checkpoint identity on line {number}")
        if record.get("model") != model:
            raise ValueError("checkpoint model mismatch")
        validate_record_provenance(record, expected[key])
        completed[key] = record
        ids.add(record["idx"])
    return completed


def run_contract(args: argparse.Namespace, rows: list[dict[str, Any]], full_run: bool) -> dict[str, Any]:
    return {"runner_version": RUNNER_VERSION, "prompt_version": PROMPT_VERSION,
            "prompt_fingerprint_sha256": build_messages(rows[0])[1], "model": args.model,
            "settings": {"workers": args.workers, "max_tokens": args.max_tokens,
                "request_timeout": args.request_timeout, "temperature": 0.0, "top_p": 1.0,
                "api_base": args.api_base.rstrip("/"), "max_case_clauses": MAX_CASE_CLAUSES,
                "max_generation_attempts": 2,
                "chat_template_kwargs": None if args.model.lower().startswith("mistralai/") else {"enable_thinking": False}},
            "source_sha256": sha256_file(args.questions), "inputs_sha256": json_sha256(rows),
            "row_indices": [row["row_index"] for row in rows], "rows": len(rows), "full_run": full_run}


def ensure_run_contract(path: Path, contract: dict[str, Any], resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError("run manifest already exists; choose a new tag or --resume")
        old = json.loads(path.read_text())
        if any(old.get(key) != value for key, value in contract.items()):
            raise ValueError("resume model/prompt/input/settings mismatch")
        return old
    manifest = {**contract, "created_at": utc_now()}
    atomic_write_json(path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "Dataset" / "barexam_qa.csv")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--row-indices", help="Comma-separated positional QA indices for a pilot")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.workers <= 4 or not 1 <= args.max_tokens <= 2000 or args.request_timeout <= 0:
        raise ValueError("require 1–4 workers, 1–2000 tokens and a positive timeout")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.out_tag):
        raise ValueError("out-tag must be a safe filename component")
    rows = load_questions(args.questions)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected the full {EXPECTED_ROWS}-row MBE dataset, found {len(rows)}")
    full_run = args.row_indices is None
    if not full_run:
        selected = [int(value.strip()) for value in args.row_indices.split(",") if value.strip()]
        if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= len(rows) for i in selected):
            raise ValueError("pilot indices must be distinct known positional rows")
        rows = [row for row in rows if row["row_index"] in set(selected)]
    prefix = OUT_DIR / f"mbe_api__{PROMPT_VERSION}__{args.out_tag}"
    paths = {key: Path(f"{prefix}__{key}.{ext}") for key, ext in (
        ("queries", "json"), ("checkpoint", "jsonl"), ("failures", "jsonl"),
        ("manifest", "json"), ("progress", "json"), ("done", "json"))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # An OS advisory lock cannot become stale after a crash, unlike a PID marker.
    run_lock = Path(f"{prefix}__run.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        run_lock.close()
        raise RuntimeError("another process already owns this MBE output tag") from exc
    try:
        return run_selected(args, rows, full_run, paths)
    finally:
        run_lock.close()


def run_selected(args: argparse.Namespace, rows: list[dict[str, Any]], full_run: bool,
                 paths: dict[str, Path]) -> int:
    """Run a validated selection while its output-tag lock is held."""
    contract = run_contract(args, rows, full_run)
    if not paths["manifest"].exists() and any(path.exists() for path in paths.values()):
        raise ValueError("orphan run artefacts exist without a manifest")
    if not args.resume and any(path.exists() for path in paths.values()):
        raise FileExistsError("output tag already has artefacts; choose a new tag or --resume")
    # Validate everything before a network request or checkpoint mutation.
    manifest = ensure_run_contract(paths["manifest"], contract, args.resume)
    completed = load_checkpoint(paths["checkpoint"], rows, args.model)
    if paths["done"].exists():
        done = json.loads(paths["done"].read_text())
        if len(completed) != len(rows) or done.get("status") != "completed":
            raise ValueError("done manifest contradicts checkpoint")
        if any(sha256_file(paths[key]) != value for key, value in done["sha256"].items()):
            raise ValueError("completed artefact checksum mismatch")
        print(f"[done] existing verified generation rows={len(completed)}", flush=True)
        return 0
    models = api_json(f"{args.api_base.rstrip('/')}/models", None, 30)
    if args.model not in {item["id"] for item in models.get("data", [])}:
        raise ValueError("requested model is not served by the API")
    pending = [row for row in rows if row["row_index"] not in completed]
    started, initial, failures = time.monotonic(), len(completed), 0
    started_at = utc_now()

    def progress(status: str, last_row: int | None = None) -> None:
        elapsed = max(time.monotonic() - started, 0.001)
        rate = (len(completed) - initial) / elapsed
        atomic_write_json(paths["progress"], {**contract, "status": status, "pid": os.getpid(),
            "started_at": started_at, "updated_at": utc_now(), "rows_total": len(rows),
            "rows_completed": len(completed), "rows_failed_this_run": failures,
            "rows_remaining": len(rows) - len(completed), "resumed_rows": initial,
            "rows_per_hour": round(rate * 3600, 3), "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round((len(rows) - len(completed)) / rate, 3) if rate else None,
            "last_row": last_row})

    progress("running")
    print(f"[start] rows={len(rows)} pending={len(pending)} workers={args.workers} model={args.model}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_question, row, api_base=args.api_base, model=args.model,
                   max_tokens=args.max_tokens, timeout=args.request_timeout): row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                record = future.result()
                append_jsonl(paths["checkpoint"], record)
                completed[row["row_index"]] = record
                outcome = f"ok attempts={record['generation_attempt_count']} tokens={record['completion_tokens_total']}"
            except Exception as exc:
                failures += 1
                append_jsonl(paths["failures"], {**{k: row[k] for k in IDENTITY_FIELDS},
                    "model": args.model, "failed_at": utc_now(), "error": str(exc),
                    "response_diagnostics": getattr(exc, "details", {})})
                outcome = f"FAILED {type(exc).__name__}: {exc}"
            progress("running", row["row_index"])
            print(f"[row {row['row_index']}] {outcome} | {len(completed)}/{len(rows)} failures={failures}", flush=True)
    if len(completed) != len(rows):
        progress("completed_with_failures")
        return 2
    final = [completed[row["row_index"]] for row in rows]
    for record, row in zip(final, rows):
        validate_record_provenance(record, row)
    atomic_write_json(paths["queries"], final)
    if not paths["failures"].exists():
        paths["failures"].touch()
    progress("completed")
    integrity = {"row_count": len(final), "unique_idx_count": len({r["idx"] for r in final}),
        "all_structured_ok": True, "max_case_clause_count": max(r["case_clause_count"] for r in final),
        "placeholder_count": sum(r["cypher_query"].count("__STATE__") for r in final),
        "unresolved_failure_count": 0}
    outputs = {key: str(path) for key, path in paths.items() if key != "done"}
    atomic_write_json(paths["done"], {**contract, "status": "completed", "finished_at": utc_now(),
        "integrity": integrity, "outputs": outputs,
        "sha256": {key: sha256_file(paths[key]) for key in outputs}})
    print(f"[done] rows={len(final)} queries={paths['queries']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
