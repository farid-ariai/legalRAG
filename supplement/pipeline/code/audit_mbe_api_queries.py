#!/usr/bin/env python3
"""Audit MBE API query artefacts and their deterministic, read-only gates.

This audit measures integrity and executability, not retrieval accuracy.  Empty
results are retained and reported; they are not silently removed or repaired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "Dataset" / "barexam_qa.csv"
DEFAULT_CORPUS = ROOT / "Dataset" / "barexam_passage.csv"
EXPECTED_ROWS = 1071
EXPECTED_CORPUS_ROWS = 772001
SAMPLE_SEED = 20260830
MANDATORY_SAMPLE_ROWS = (391, 939)
IDENTITY_FIELDS = ("row_index", "idx", "question", "choices", "input_sha256")
MODEL_FIELDS = ("model", "prompt_version", "prompt_fingerprint_sha256")
GRAPH_COUNT_FIELDS = ("passages", "doctrines", "doctrine_stubs", "elements", "contains_doctrine", "has_element")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_plain_json(value: Any, path: str = "$") -> None:
    """Reject graph objects, non-finite numbers and non-string object keys."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            require_plain_json(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"non-string JSON key at {path}")
            require_plain_json(item, f"{path}.{key}")
        return
    raise ValueError(f"non-JSON value at {path}: {type(value).__name__}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    require_plain_json(value)
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def load_strict_json(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return decode_strict_json(payload, path.name), hashlib.sha256(payload).hexdigest()


def decode_strict_json(payload: bytes | str, name: str = "JSON") -> Any:

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {name}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {name}")
            result[key] = value
        return result

    value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=unique_object)
    require_plain_json(value)
    return value


def load_strict_jsonl(path: Path) -> tuple[list[Any], str]:
    payload = path.read_bytes()
    records = [decode_strict_json(line, f"{path.name}:{position}")
               for position, line in enumerate(payload.splitlines(), start=1) if line.strip()]
    return records, hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    require_plain_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in IDENTITY_FIELDS}


def uniform_model_identity(rows: list[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        raise ValueError("query artefact is empty")
    result = {}
    for field in MODEL_FIELDS:
        values = {row.get(field) for row in rows}
        if len(values) != 1 or not isinstance(next(iter(values)), str) or not next(iter(values)):
            raise ValueError(f"missing or mixed {field}")
        result[field] = next(iter(values))
    if not re.fullmatch(r"[0-9a-f]{64}", result["prompt_fingerprint_sha256"]):
        raise ValueError("invalid prompt fingerprint")
    return result


def load_corpus_ids(path: Path, expected_rows: int = EXPECTED_CORPUS_ROWS) -> set[str]:
    """Read every corpus ID; do not restrict retrieval to the mbe_* subset."""
    csv.field_size_limit(sys.maxsize)
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "idx" not in (reader.fieldnames or []):
            raise ValueError("corpus has no idx column")
        for position, row in enumerate(reader):
            idx = row["idx"]
            if not idx or idx in ids:
                raise ValueError(f"empty or duplicate corpus idx at row {position}")
            ids.add(idx)
    if len(ids) != expected_rows:
        raise ValueError(f"expected {expected_rows} corpus passages, found {len(ids)}")
    return ids


def load_graph_binding(
    path: Path, *, corpus_sha256: str, corpus_rows: int,
    uri: str | None = None, database: str | None = None,
) -> dict[str, Any]:
    """Bind the full-coverage graph audit, not merely its reused Bolt port."""
    graph_audit, audit_hash = load_strict_json(path)
    if not isinstance(graph_audit, dict) or graph_audit.get("status") != "passed":
        raise ValueError("graph coverage audit has not passed")
    if graph_audit.get("corpus_sha256") != corpus_sha256:
        raise ValueError("graph audit refers to a different corpus")
    for field in ("extraction_sha256", "passage_id_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(graph_audit.get(field, ""))):
            raise ValueError(f"graph audit has invalid {field}")
    counts = graph_audit.get("graph_counts", {})
    if set(counts) != set(GRAPH_COUNT_FIELDS) or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("graph audit has invalid graph counts")
    source_counts = graph_audit.get("source_counts", {})
    if any(counts[field] != source_counts.get(field) for field in ("passages", "doctrines", "elements")):
        raise ValueError("graph audit source/graph counts disagree")
    if counts["passages"] != corpus_rows or counts["contains_doctrine"] != counts["doctrines"] or counts["has_element"] != counts["elements"]:
        raise ValueError("graph audit does not cover the full source")
    if not isinstance(graph_audit.get("source_name"), str) or not graph_audit["source_name"]:
        raise ValueError("graph audit has no source name")
    if uri is not None and graph_audit.get("neo4j_uri") != uri:
        raise ValueError("graph audit and executor Neo4j endpoints differ")
    if database is not None and graph_audit.get("neo4j_database") != database:
        raise ValueError("graph audit and executor database names differ")
    return {"audit_sha256": audit_hash, **{field: graph_audit[field] for field in (
        "source_name", "extraction_sha256", "corpus_sha256", "passage_id_sha256",
        "graph_counts", "neo4j_uri", "neo4j_database")}}


def sample_stratum(row: dict[str, Any]) -> tuple[int, int]:
    """Use input length and choice count only, never labels or model outputs."""
    choices = row["choices"]
    choice_count = (
        len(choices) if isinstance(choices, list)
        else len(re.findall(r"(?m)^\s*[A-D][.):]\s", str(choices)))
    )
    length = len(str(row["question"])) + len(str(choices))
    length_bin = sum(length > boundary for boundary in (1000, 2000, 4000))
    return length_bin, choice_count


def deterministic_sample(
    rows: list[dict[str, Any]], size: int = 64, seed: int = SAMPLE_SEED,
) -> list[dict[str, Any]]:
    """Balance safe-input strata, retaining known long-input edge cases."""
    if size < 1:
        raise ValueError("sample size must be positive")
    by_position = {row["row_index"]: row for row in rows}
    if len(by_position) != len(rows):
        raise ValueError("duplicate sample source row_index")
    target = min(size, len(rows))
    mandatory = [index for index in MANDATORY_SAMPLE_ROWS if index in by_position]
    if len(mandatory) > target:
        raise ValueError("sample is too small for mandatory rows")
    selected = set(mandatory)
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["row_index"] not in selected:
            buckets[sample_stratum(row)].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: sha256_text(f"{seed}:{row['row_index']}:{row['input_sha256']}"))
    while len(selected) < target:
        advanced = False
        for stratum in sorted(buckets):
            if buckets[stratum] and len(selected) < target:
                selected.add(buckets[stratum].pop(0)["row_index"])
                advanced = True
        if not advanced:
            raise ValueError("unable to fill sample")
    return [by_position[index] for index in sorted(selected)]


def audit_rows(
    rows: list[dict[str, Any]], questions: list[dict[str, Any]], *,
    row_indices: list[int] | None = None, expected_model: str | None = None,
) -> dict[str, Any]:
    """Check exact source mapping and replay the runner's provenance validator."""
    from mbe_api_cypher_generation import build_messages, validate_cypher, validate_record_provenance

    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("queries must be a JSON list of objects")
    require_plain_json(rows)
    by_position = {row["row_index"]: row for row in questions}
    if len(by_position) != len(questions):
        raise ValueError("duplicate row_index in question source")
    expected_indices = sorted(by_position) if row_indices is None else sorted(row_indices)
    if len(set(expected_indices)) != len(expected_indices):
        raise ValueError("duplicate explicitly selected row_index")
    if not set(expected_indices) <= set(by_position):
        raise ValueError("selected row_index is absent from the question source")
    actual_indices = [row.get("row_index") for row in rows]
    if actual_indices != expected_indices:
        raise ValueError("query coverage/order differs from the explicit source selection")
    if len({row.get("idx") for row in rows}) != len(rows):
        raise ValueError("duplicate query idx")
    model_identity = uniform_model_identity(rows)
    if expected_model is not None and model_identity["model"] != expected_model:
        raise ValueError("query model differs from the required model")
    errors = []
    corrected = []
    normalised = []
    for row in rows:
        position = row["row_index"]
        expected = by_position[position]
        try:
            if source_identity(row) != source_identity(expected):
                raise ValueError("question, choices or source identity mismatch")
            if row.get("input_warnings", []) != expected.get("input_warnings", []):
                raise ValueError("source input warnings mismatch")
            if row.get("structured_ok") is not True:
                raise ValueError("structured_ok is not true")
            query = row.get("cypher_query")
            if not isinstance(query, str) or validate_cypher(query) != query:
                raise ValueError("query is not in canonical validated form")
            if "__STATE__" in query:
                raise ValueError("Housing state placeholder in MBE query")
            _, prompt_fingerprint = build_messages(expected)
            if row["prompt_fingerprint_sha256"] != prompt_fingerprint:
                raise ValueError("prompt fingerprint differs from current prompt")
            validate_record_provenance(row, expected_row=expected)
            if len(row.get("generation_attempts", [])) > 1:
                corrected.append(position)
            if row.get("raw_cypher") != query:
                normalised.append(position)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"row_index": position, "idx": row.get("idx"), "error": str(exc)})
    if errors:
        raise ValueError(f"query/provenance audit failed: {errors[:20]}")
    return {
        **model_identity, "row_count": len(rows), "unique_idx_count": len(rows),
        "question_source_row_count": len(questions), "row_indices": expected_indices,
        "question_coverage": "full" if expected_indices == sorted(by_position) else "explicit_subset",
        "all_structured_ok": True, "final_generation_failures": 0,
        "max_case_clause_count": max(row["case_clause_count"] for row in rows),
        "placeholder_count": 0, "corrected_row_indices": corrected,
        "normalised_row_indices": normalised,
        "input_identity_sha256": canonical_sha256([source_identity(row) for row in rows]),
    }


def verify_done(
    done: dict[str, Any], queries_path: Path, queries_hash: str,
    rows_audit: dict[str, Any], questions_hash: str, *,
    rows: list[dict[str, Any]], questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check every advertised output checksum, including unresolved failures."""
    if done.get("status") != "completed":
        raise ValueError("generation done manifest is not completed")
    outputs, hashes = done.get("outputs"), done.get("sha256")
    if not isinstance(outputs, dict) or not isinstance(hashes, dict):
        raise ValueError("done manifest has no output/checksum maps")
    required = {"queries", "checkpoint", "failures", "manifest", "progress"}
    if not required <= set(outputs) or set(outputs) != set(hashes):
        raise ValueError("done output/checksum keys are incomplete or inconsistent")
    checked = {}
    loaded = {}
    for key, advertised in outputs.items():
        if not isinstance(advertised, str) or not advertised:
            raise ValueError(f"invalid done path for {key}")
        # Copied artefacts retain their remote absolute paths in provenance.
        # Resolve only their basename beside the explicitly selected queries.
        local_path = queries_path.parent / Path(advertised).name
        if key == "queries" and local_path.resolve() != queries_path.resolve():
            raise ValueError("done manifest names a different query artefact")
        if key == "queries":
            actual_hash = queries_hash
        else:
            value, actual_hash = (
                load_strict_jsonl(local_path) if key in {"checkpoint", "failures"}
                else load_strict_json(local_path)
            )
            loaded[key] = value
        if actual_hash != hashes[key]:
            raise ValueError(f"done checksum mismatch for {key}")
        checked[key] = actual_hash
    checkpoint = loaded["checkpoint"]
    checkpoint_by_idx = {row.get("row_index"): row for row in checkpoint}
    if len(checkpoint_by_idx) != len(checkpoint) or checkpoint_by_idx != {row["row_index"]: row for row in rows}:
        raise ValueError("checkpoint and final queries differ or have duplicate/missing rows")
    selected = set(rows_audit["row_indices"])
    source_by_idx = {row["row_index"]: row for row in questions}
    failures = loaded["failures"]
    for failure in failures:
        position = failure.get("row_index")
        if position not in selected or source_identity(failure) != source_identity(source_by_idx[position]):
            raise ValueError("failure ledger has unknown or mismatched source identity")
        if failure.get("model") != rows_audit["model"] or not failure.get("error"):
            raise ValueError("failure ledger has missing diagnostics or a different model")
    manifest = loaded["manifest"]
    for field in MODEL_FIELDS:
        if manifest.get(field) != rows_audit[field] or done.get(field) != rows_audit[field]:
            raise ValueError(f"run manifest {field} mismatch")
    if manifest.get("source_sha256") != questions_hash:
        raise ValueError("run manifest question source checksum mismatch")
    if manifest.get("row_indices") != rows_audit["row_indices"]:
        raise ValueError("run manifest selected rows mismatch")
    if manifest.get("rows") != rows_audit["row_count"]:
        raise ValueError("run manifest row count mismatch")
    if bool(manifest.get("full_run")) != (rows_audit["question_coverage"] == "full"):
        raise ValueError("run manifest full/subset claim mismatch")
    if manifest.get("inputs_sha256") != canonical_sha256([source_by_idx[position] for position in rows_audit["row_indices"]]):
        raise ValueError("manifest safe-input checksum mismatch")
    for field in ("runner_version", "settings", "source_sha256", "inputs_sha256", "row_indices", "rows", "full_run"):
        if done.get(field) != manifest.get(field):
            raise ValueError(f"done/run-manifest contract mismatch for {field}")
    settings = manifest.get("settings", {})
    if not 1 <= settings.get("workers", 0) <= 4 or not 1 <= settings.get("max_tokens", 0) <= 2000:
        raise ValueError("run settings exceed worker/token limits")
    for key, expected in {"temperature": 0.0, "top_p": 1.0, "max_case_clauses": 10, "max_generation_attempts": 2}.items():
        if settings.get(key) != expected:
            raise ValueError(f"unexpected generation setting {key}")
    progress = loaded["progress"]
    if progress.get("status") != "completed" or progress.get("rows_completed") != len(rows) or progress.get("rows_remaining") != 0:
        raise ValueError("progress does not confirm complete generation")
    integrity = done.get("integrity", {})
    for field in ("row_count", "unique_idx_count", "all_structured_ok", "max_case_clause_count", "placeholder_count"):
        if integrity.get(field) != rows_audit[field]:
            raise ValueError(f"done integrity claim mismatch for {field}")
    if integrity.get("unresolved_failure_count") != 0:
        raise ValueError("done manifest reports unresolved failures")
    return {"checksums_verified": checked, "manifest": manifest,
            "preserved_failure_record_count": len(failures),
            "resolved_failure_row_indices": sorted({row["row_index"] for row in failures})}


def audit_execution_results(
    result: dict[str, Any], expected_queries: list[dict[str, Any]], *,
    queries_sha256: str, mode: str, corpus_ids: set[str] | None = None,
    graph_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind execution records to immutable inputs and reject every gate error."""
    from mbe_execute_cypher import validate_result_rows, rank_passages

    require_plain_json(result)
    if result.get("status") != "completed" or result.get("mode") != mode:
        raise ValueError(f"{mode} gate is incomplete or contains errors")
    identity = result.get("run_identity")
    if not isinstance(identity, dict) or result.get("run_identity_sha256") != canonical_sha256(identity):
        raise ValueError(f"{mode} gate run identity checksum mismatch")
    identity_expected = {"queries_sha256": queries_sha256, "mode": mode,
                         "rows": len(expected_queries), "row_indices": [row["row_index"] for row in expected_queries],
                         **uniform_model_identity(expected_queries)}
    if any(identity.get(field) != value for field, value in identity_expected.items()):
        raise ValueError(f"{mode} gate execution configuration/source mismatch")
    if graph_binding is None or identity.get("graph") != graph_binding:
        raise ValueError(f"{mode} gate graph audit/extraction binding is absent or mismatched")
    if identity.get("corpus_sha256") != graph_binding["corpus_sha256"]:
        raise ValueError(f"{mode} gate corpus/graph fingerprint mismatch")
    for field in ("live_graph_check_before", "live_graph_check_after"):
        check = result.get(field, {})
        if check.get("status") != "passed" or check.get("counts") != graph_binding["graph_counts"]:
            raise ValueError(f"{mode} gate has no matching {field}")
    if result.get("completed_count") != len(expected_queries) or result.get("failed_count") != 0:
        raise ValueError(f"{mode} gate completion/error counts mismatch")
    source = result.get("source", {})
    if source.get("sha256") != queries_sha256:
        raise ValueError(f"{mode} gate input checksum mismatch")
    if source.get("rows") != len(expected_queries):
        raise ValueError(f"{mode} gate input row count mismatch")
    for field, value in uniform_model_identity(expected_queries).items():
        if source.get(field) != value:
            raise ValueError(f"{mode} gate {field} mismatch")
    records = result.get("results")
    if not isinstance(records, list) or len(records) != len(expected_queries):
        raise ValueError(f"{mode} gate result coverage mismatch")
    errors = []
    empty = 0
    depths = []
    for query, record in zip(expected_queries, records):
        try:
            for field in ("row_index", "idx", "input_sha256", *MODEL_FIELDS):
                if record.get(field) != query[field]:
                    raise ValueError(f"result identity mismatch for {field}")
            if record.get("query_sha256") != sha256_text(query["cypher_query"]):
                raise ValueError("result query hash mismatch")
            if record.get("cypher_query") != query["cypher_query"]:
                raise ValueError("result query text mismatch")
            if record.get("mode") != mode or record.get("error") is not None or record.get("status") != "passed":
                raise ValueError(f"failed execution: {record.get('error')}")
            if record.get("query_type") != "r":
                raise ValueError("Neo4j did not classify the query as read-only")
            if mode == "explain" and (not isinstance(record.get("plan"), dict) or not record["plan"]):
                raise ValueError("EXPLAIN result has no query plan")
            if mode == "execute":
                hits = record.get("result_rows")
                validate_result_rows(hits, corpus_ids=corpus_ids)
                ranked, kept = rank_passages(hits)
                if ranked != record.get("retrieved_passage_idx") or kept != record.get("top_hits"):
                    raise ValueError("result ranking does not match returned rows")
                if record.get("n_result_rows") != len(hits) or record.get("n_passages") != len(ranked):
                    raise ValueError("result count mismatch")
                empty += not ranked
                depths.append(len(ranked))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"row_index": query["row_index"], "error": str(exc)})
    if errors:
        raise ValueError(f"{mode} gate binding failed: {errors[:20]}")
    return {
        "status": "passed", "mode": mode, "row_count": len(records),
        "query_error_count": 0, "serialisation_error_count": 0,
        "empty_result_count": empty,
        "mean_ranked_depth": sum(depths) / len(depths) if depths else None,
        "queries_sha256": queries_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-out", type=Path, required=True)
    parser.add_argument("--row-indices", help="Explicit comma-separated pilot rows; omit only for a full run.")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-question-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--expected-corpus-rows", type=int, default=EXPECTED_CORPUS_ROWS)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--explain-results", type=Path)
    parser.add_argument("--execute-results", type=Path)
    parser.add_argument("--graph-audit", type=Path, help="Required when auditing either execution gate.")
    return parser.parse_args()


def main() -> None:
    from mbe_api_cypher_generation import load_questions

    args = parse_args()
    questions = load_questions(args.questions)
    if len(questions) != args.expected_question_rows:
        raise ValueError(f"expected {args.expected_question_rows} question rows, found {len(questions)}")
    rows, query_hash = load_strict_json(args.queries)
    done, done_hash = load_strict_json(args.done)
    selection = [int(value) for value in args.row_indices.split(",")] if args.row_indices else None
    audit = audit_rows(rows, questions, row_indices=selection, expected_model=args.expected_model)
    source_hash = sha256_file(args.questions)
    audit.update(verify_done(done, args.queries, query_hash, audit, source_hash, rows=rows, questions=questions))
    corpus_ids = load_corpus_ids(args.corpus, args.expected_corpus_rows)
    corpus_hash = sha256_file(args.corpus)
    if (args.explain_results or args.execute_results) and not args.graph_audit:
        raise ValueError("execution gates require --graph-audit to identify the extraction store")
    graph_binding = load_graph_binding(args.graph_audit, corpus_sha256=corpus_hash, corpus_rows=len(corpus_ids)) if args.graph_audit else None
    sample = deterministic_sample(rows, args.sample_size, args.seed)
    if args.sample_out.exists():
        existing, _ = load_strict_json(args.sample_out)
        if existing != sample:
            raise ValueError("existing sample differs; preserve it and use a new sample path")
    else:
        atomic_write_json(args.sample_out, sample)
    sample_hash = sha256_file(args.sample_out)
    gates = {}
    for mode, path, expected, digest in (
        ("explain", args.explain_results, rows, query_hash),
        ("execute", args.execute_results, sample, sample_hash),
    ):
        if path:
            result, result_hash = load_strict_json(path)
            gates[mode] = audit_execution_results(
                result, expected, queries_sha256=digest, mode=mode, corpus_ids=corpus_ids, graph_binding=graph_binding,
            )
            gates[mode]["results_sha256"] = result_hash
    audit.update({
        "status": "passed", "audited_at": utc_now(), "scope": "integrity_and_executability_not_retrieval_accuracy",
        "corpus_row_count": len(corpus_ids), "sample_count": len(sample), "sample_seed": args.seed,
        "sample_row_indices": [row["row_index"] for row in sample],
        "sample_strata": {str(key): value for key, value in Counter(sample_stratum(row) for row in sample).items()},
        "sha256": {"questions": source_hash, "queries": query_hash, "done": done_hash,
                   "sample": sample_hash, "corpus": corpus_hash},
        "gates": gates, "graph_binding": graph_binding,
    })
    atomic_write_json(args.out, audit)
    print(f"MBE audit passed: {len(rows)} queries; {len(corpus_ids)} corpus passages; sample {len(sample)}; gates {','.join(gates) or 'pending'}")


if __name__ == "__main__":
    main()
