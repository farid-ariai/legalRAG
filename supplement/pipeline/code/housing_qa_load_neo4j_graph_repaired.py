#!/usr/bin/env python3
"""Load the frozen Housing extraction into a separate occurrence-preserving graph.

This loader deliberately has no wipe option. It accepts only an empty, isolated
Neo4j store and the 49 checksum-bound server extraction files recorded by the
8 September 2026 loader audit. A failed file can be replayed explicitly with
--resume because every node and relationship has a stable identity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import ijson
from neo4j import GraphDatabase


EXPECTED_STATUTES = 1_837_403
EXPECTED_NAMED_RULES = 3_167_715
EXPECTED_EXCEPTIONS = 630_823
EXPECTED_SOURCE_FILES = 49

CONSTRAINTS = (
    "CREATE CONSTRAINT repair_statute_idx IF NOT EXISTS FOR (s:Statute) REQUIRE s.idx IS UNIQUE",
    "CREATE CONSTRAINT repair_provision_uid IF NOT EXISTS FOR (p:Provision) REQUIRE p.uid IS UNIQUE",
    "CREATE CONSTRAINT repair_condition_uid IF NOT EXISTS FOR (c:Condition) REQUIRE c.uid IS UNIQUE",
    "CREATE CONSTRAINT repair_area_name IF NOT EXISTS FOR (a:AreaOfLaw) REQUIRE a.name IS UNIQUE",
    "CREATE INDEX repair_provision_name IF NOT EXISTS FOR (p:Provision) ON (p.name)",
    "CREATE INDEX repair_provision_type IF NOT EXISTS FOR (p:Provision) ON (p.type)",
    "CREATE INDEX repair_provision_statute_name IF NOT EXISTS FOR (p:Provision) ON (p.statute_idx, p.name)",
    "CREATE INDEX repair_citation_key IF NOT EXISTS FOR (x:Citation) ON (x.code_name, x.section)",
)

LOAD_QUERY = """
UNWIND $rows AS row
MERGE (s:Statute {idx: row.idx})
  SET s.citation = row.citation,
      s.state = row.state,
      s.path = row.path,
      s.summary = row.summary,
      s.source_file_sha256 = row.source_file_sha256
WITH s, row
UNWIND row.provisions AS prov
MERGE (p:Provision {uid: prov.uid})
  SET p.statute_idx = row.idx,
      p.ordinal = prov.ordinal,
      p.name = prov.name,
      p.type = prov.type,
      p.rule = prov.rule,
      p.amounts = prov.amounts,
      p.actors = prov.actors,
      p.exceptions = prov.exceptions,
      p.is_stub = false
MERGE (s)-[:CONTAINS_PROVISION]->(p)
FOREACH (area IN CASE WHEN prov.area IS NULL THEN [] ELSE [prov.area] END |
  MERGE (a:AreaOfLaw {name: area})
  MERGE (p)-[:BELONGS_TO]->(a)
)
FOREACH (cond IN prov.conditions |
  MERGE (c:Condition {uid: cond.uid})
    SET c.statute_idx = row.idx,
        c.provision = prov.name,
        c.provision_uid = prov.uid,
        c.number = cond.number,
        c.description = cond.description
  MERGE (p)-[:HAS_CONDITION]->(c)
)
FOREACH (cit IN prov.citations |
  MERGE (x:Citation {code_name: cit.code_name, section: cit.section})
  MERGE (p)-[:SUPPORTED_BY]->(x)
)
FOREACH (rel IN prov.related |
  MERGE (r:Provision {uid: rel.uid})
    ON CREATE SET r.statute_idx = row.idx,
                  r.name = rel.name,
                  r.is_stub = rel.is_stub
  MERGE (p)-[:RELATED_TO]->(r)
)
"""

COUNT_QUERY = """
CALL () { MATCH (s:Statute) RETURN count(s) AS statutes }
CALL () { MATCH (:Statute)-[r:CONTAINS_PROVISION]->(:Provision) RETURN count(r) AS owned_provisions }
CALL () { MATCH (p:Provision) WHERE p.is_stub = false RETURN count(p) AS source_provisions }
CALL () { MATCH (p:Provision) WHERE p.is_stub = true RETURN count(p) AS related_stubs }
RETURN statutes, owned_provisions, source_provisions, related_stubs
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalise_name(value: str) -> str:
    return " ".join(value.casefold().split())


def shape_row(record: dict, source_sha256: str) -> tuple[dict, Counter]:
    """Keep every named rule as an individual statute-local occurrence."""
    raw_idx = record.get("statute_idx")
    idx = str(raw_idx)
    if not idx.isascii() or not idx.isdecimal() or idx != str(int(idx)):
        raise ValueError(f"non-canonical statute_idx: {raw_idx!r}")
    raw_rules = record.get("rules") or []
    if not isinstance(raw_rules, list):
        raise ValueError(f"statute {idx} has a non-list rules field")

    first_uid_by_name: dict[str, str] = {}
    for ordinal, raw in enumerate(raw_rules, start=1):
        name = (raw.get("topic_name") or "").strip()
        if not name:
            raise ValueError(f"statute {idx} rule {ordinal} lacks a name")
        first_uid_by_name.setdefault(normalise_name(name), f"{idx}:rule:{ordinal}")

    provisions = []
    counts = Counter(statutes=1)
    for ordinal, raw in enumerate(raw_rules, start=1):
        uid = f"{idx}:rule:{ordinal}"
        conditions = [
            {
                "uid": f"{uid}:condition:{position}",
                "number": condition.get("number"),
                "description": (condition.get("description") or "").strip(),
            }
            for position, condition in enumerate(raw.get("conditions") or [], start=1)
            if (condition.get("description") or "").strip()
        ]
        citations = [
            {
                "code_name": (citation.get("code_name") or "").strip(),
                "section": (citation.get("section") or "").strip(),
            }
            for citation in (raw.get("cross_references") or [])
            if (citation.get("code_name") or "").strip()
            or (citation.get("section") or "").strip()
        ]
        related = []
        for term in raw.get("related_topics") or []:
            name = (term or "").strip()
            if not name:
                continue
            key = normalise_name(name)
            real_uid = first_uid_by_name.get(key)
            related.append({
                "uid": real_uid or f"{idx}:related:{hashlib.sha256(key.encode()).hexdigest()}",
                "name": name,
                "is_stub": real_uid is None,
            })
        exceptions = [
            str(value).strip() for value in (raw.get("exceptions") or [])
            if str(value).strip()
        ]
        provisions.append({
            "uid": uid,
            "ordinal": ordinal,
            "name": (raw.get("topic_name") or "").strip(),
            "type": raw.get("topic_type"),
            "area": raw.get("area_of_law"),
            "rule": raw.get("concise_rule"),
            "amounts": raw.get("deadlines_or_amounts") or [],
            "actors": raw.get("actors_affected") or [],
            "exceptions": exceptions,
            "conditions": conditions,
            "citations": citations,
            "related": related,
        })
        counts["named_rules"] += 1
        counts["exceptions"] += len(exceptions)
        counts["conditions"] += len(conditions)

    return {
        "idx": idx,
        "citation": record.get("citation") or "",
        "state": record.get("state") or "",
        "path": record.get("path") or "",
        "summary": record.get("source_text_summary") or "",
        "source_file_sha256": source_sha256,
        "provisions": provisions,
    }, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validated_sources(audit_path: Path) -> tuple[list[dict], dict, str]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sources = audit.get("sources")
    if audit.get("status") != "PASS" or not isinstance(sources, list):
        raise ValueError("source audit is not a PASS with a source list")
    if len(sources) != EXPECTED_SOURCE_FILES:
        raise ValueError("source audit does not list the 49 frozen files")
    counts = audit.get("counts") or {}
    expected = {
        "statutes": EXPECTED_STATUTES,
        "named_rule_entries": EXPECTED_NAMED_RULES,
        "exception_strings_not_loaded": EXPECTED_EXCEPTIONS,
    }
    if any(counts.get(key) != value for key, value in expected.items()):
        raise ValueError("source audit counts differ from the frozen full corpus")
    for source in sources:
        path = Path(source["path"])
        if not path.is_file():
            raise ValueError(f"missing frozen extraction file: {path}")
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"frozen extraction hash differs: {path}")
    return sources, expected, sha256_file(audit_path)


def main() -> None:
    args = parse_args()
    parsed_uri = urlparse(args.neo4j_uri)
    if parsed_uri.scheme not in {"bolt", "neo4j"} or parsed_uri.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("repair loader requires an explicit localhost Neo4j URI")
    if parsed_uri.port in {None, 7687, 7688}:
        raise ValueError("repair loader refuses the historical Housing and MBE ports")
    if args.batch_size < 1 or args.batch_size > 1000:
        raise ValueError("batch size must be from 1 to 1000")
    sources, expected, audit_sha256 = validated_sources(args.source_audit)
    output_dir = args.output_dir.resolve()
    if args.resume:
        status_path = output_dir / "loader_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") not in {"running", "failed"}:
            raise ValueError("resume requires a running or failed loader status")
        if status.get("source_audit_sha256") != audit_sha256 or status.get("neo4j_uri") != args.neo4j_uri:
            raise ValueError("resume source or endpoint differs")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        status_path = output_dir / "loader_status.json"
        status = {
            "status": "running",
            "source_audit_sha256": audit_sha256,
            "neo4j_uri": args.neo4j_uri,
            "completed_files": [],
            "counts": {},
        }
        atomic_json(status_path, status)

    password = args.password_file.read_text(encoding="utf-8").strip()
    if not password:
        raise ValueError("password file is empty")
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password))
    totals = Counter(status.get("counts", {}))
    completed = set(status.get("completed_files", []))
    try:
        driver.verify_connectivity()
        with driver.session(database=args.neo4j_database) as session:
            if not args.resume:
                existing = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
                if existing != 0:
                    raise ValueError("repair target is not an empty isolated store")
                for statement in CONSTRAINTS:
                    session.run(statement).consume()
                session.run("CALL db.awaitIndexes(600)").consume()

            for source in sources:
                path = Path(source["path"])
                if str(path) in completed:
                    continue
                file_counts = Counter()
                batch = []
                with path.open("rb") as handle:
                    for record in ijson.items(handle, "item"):
                        row, counts = shape_row(record, source["sha256"])
                        file_counts.update(counts)
                        batch.append(row)
                        if len(batch) >= args.batch_size:
                            session.run(LOAD_QUERY, rows=batch).consume()
                            batch.clear()
                    if batch:
                        session.run(LOAD_QUERY, rows=batch).consume()
                if file_counts["statutes"] != source["statutes"]:
                    raise ValueError(f"source statute count differs for {path.name}")
                totals.update(file_counts)
                completed.add(str(path))
                status["completed_files"] = sorted(completed)
                status["counts"] = dict(totals)
                atomic_json(status_path, status)
                print(json.dumps({"completed_file": path.name, "counts": dict(file_counts)}), flush=True)

            if totals["statutes"] != expected["statutes"] or totals["named_rules"] != expected["named_rule_entries"] or totals["exceptions"] != expected["exception_strings_not_loaded"]:
                raise ValueError("full extraction counts differ from frozen source audit")
            graph_counts = dict(session.run(COUNT_QUERY).single())
            if graph_counts["statutes"] != EXPECTED_STATUTES or graph_counts["owned_provisions"] != EXPECTED_NAMED_RULES or graph_counts["source_provisions"] != EXPECTED_NAMED_RULES:
                raise ValueError(f"repaired graph counts differ: {graph_counts}")
            status["status"] = "completed"
            status["graph_counts"] = graph_counts
            atomic_json(status_path, status)
            print(json.dumps({"status": "completed", "graph_counts": graph_counts}), flush=True)
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(status_path, status)
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    main()
