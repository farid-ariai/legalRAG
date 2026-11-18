#!/usr/bin/env python3
"""Stream an MBE structured-extraction JSON file into Neo4j.

The loader keeps each model extraction in its own Neo4j store.  Doctrine and
Element identities are passage-local so repeated doctrine names cannot join
unrelated passages or overwrite one another's definitions.  Citations and
areas of law are shared when their complete values match.

The extraction files are close to 1 GB, so records are parsed and written in
batches rather than loading the whole JSON array into memory.

Example::

    NEO4J_PASSWORD="..." python Codes/mbe_load_neo4j_graph.py \
        --extraction 'Structured_data/MBE_extraction_results(Qwen2.5_7B).json' \
        --neo4j-uri bolt://127.0.0.1:7688
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson
from neo4j import GraphDatabase


DEFAULT_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7688")
DEFAULT_USER = os.environ.get(
    "NEO4J_USER", os.environ.get("NEO4J_USERNAME", "neo4j")
)
DEFAULT_DATABASE = os.environ.get(
    "NEO4J_DATABASE", os.environ.get("NEO4J_DB", "neo4j")
)
DEFAULT_PASSWORD_FILE = Path.home() / ".neo4j_pass"

AREAS_OF_LAW = {
    "Evidence",
    "Criminal Law",
    "Civil Procedure",
    "Constitutional Law",
    "Torts",
    "Real Property",
    "Criminal Procedure",
    "Contracts",
    "Secured Transactions",
    "Family Law",
    "Trusts and Estates",
    "Conflict of Laws",
}

DOCTRINE_TYPES = {
    "Evidentiary Rule",
    "Common Law Doctrine",
    "Procedural Rule",
    "Constitutional Principle",
    "Legal Test",
    "Affirmative Defense",
    "Statutory Rule",
    "Cause of Action",
}

CONSTRAINTS = [
    "CREATE CONSTRAINT mbe_passage_row_index IF NOT EXISTS "
    "FOR (p:Passage) REQUIRE p.row_index IS UNIQUE",
    "CREATE CONSTRAINT mbe_doctrine_uid IF NOT EXISTS "
    "FOR (d:Doctrine) REQUIRE d.uid IS UNIQUE",
    "CREATE CONSTRAINT mbe_element_uid IF NOT EXISTS "
    "FOR (e:Element) REQUIRE e.uid IS UNIQUE",
    "CREATE CONSTRAINT mbe_citation_uid IF NOT EXISTS "
    "FOR (c:Citation) REQUIRE c.uid IS UNIQUE",
    "CREATE CONSTRAINT mbe_area_name IF NOT EXISTS "
    "FOR (a:AreaOfLaw) REQUIRE a.name IS UNIQUE",
    "CREATE INDEX mbe_doctrine_name IF NOT EXISTS FOR (d:Doctrine) ON (d.name)",
    "CREATE INDEX mbe_doctrine_type IF NOT EXISTS FOR (d:Doctrine) ON (d.type)",
    "CREATE INDEX mbe_doctrine_passage IF NOT EXISTS "
    "FOR (d:Doctrine) ON (d.passage_row_index)",
]

LOAD_QUERY = """
UNWIND $rows AS row
MERGE (p:Passage {row_index: row.row_index})
  SET p.summary = row.summary,
      p.jurisdiction = row.jurisdiction
WITH p, row
UNWIND row.doctrines AS doctrine
  MERGE (d:Doctrine {uid: doctrine.uid})
    SET d.passage_row_index = row.row_index,
        d.ordinal = doctrine.ordinal,
        d.name = doctrine.name,
        d.type = doctrine.type,
        d.definition = doctrine.definition,
        d.exceptions = doctrine.exceptions,
        d.is_stub = false
  MERGE (p)-[:CONTAINS_DOCTRINE]->(d)
  FOREACH (_ IN CASE WHEN doctrine.area = "" THEN [] ELSE [1] END |
    MERGE (a:AreaOfLaw {name: doctrine.area})
    MERGE (d)-[:BELONGS_TO]->(a)
  )
  FOREACH (element IN doctrine.elements |
    MERGE (e:Element {uid: element.uid})
      SET e.doctrine_uid = doctrine.uid,
          e.number = element.number,
          e.description = element.description
    MERGE (d)-[:HAS_ELEMENT]->(e)
  )
  FOREACH (citation IN doctrine.citations |
    MERGE (c:Citation {uid: citation.uid})
      SET c.case_name = citation.case_name,
          c.volume = citation.volume,
          c.reporter = citation.reporter,
          c.page = citation.page,
          c.pinpoint = citation.pinpoint,
          c.court = citation.court,
          c.year = citation.year,
          c.parallel_citations = citation.parallel_citations
    MERGE (d)-[:SUPPORTED_BY]->(c)
  )
  FOREACH (related IN doctrine.related |
    MERGE (r:Doctrine {uid: related.uid})
      ON CREATE SET r.passage_row_index = row.row_index,
                    r.name = related.name,
                    r.is_stub = true
    MERGE (d)-[:RELATED_TO]->(r)
  )
"""

COUNT_QUERY = """
CALL () { MATCH (p:Passage) RETURN count(p) AS passages }
CALL () { MATCH (d:Doctrine) RETURN count(d) AS doctrines }
CALL () { MATCH (d:Doctrine {is_stub: true}) RETURN count(d) AS doctrine_stubs }
CALL () { MATCH (e:Element) RETURN count(e) AS elements }
CALL () { MATCH (c:Citation) RETURN count(c) AS citations }
CALL () { MATCH (a:AreaOfLaw) RETURN count(a) AS areas }
CALL () { MATCH ()-[r:CONTAINS_DOCTRINE]->() RETURN count(r) AS contains_doctrine }
CALL () { MATCH ()-[r:HAS_ELEMENT]->() RETURN count(r) AS has_element }
CALL () { MATCH ()-[r:SUPPORTED_BY]->() RETURN count(r) AS supported_by }
CALL () { MATCH ()-[r:BELONGS_TO]->() RETURN count(r) AS belongs_to }
CALL () { MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS related_to }
RETURN passages, doctrines, doctrine_stubs, elements, citations, areas,
       contains_doctrine, has_element, supported_by, belongs_to, related_to
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--neo4j-uri", default=DEFAULT_URI)
    parser.add_argument("--neo4j-user", default=DEFAULT_USER)
    parser.add_argument("--neo4j-database", default=DEFAULT_DATABASE)
    parser.add_argument("--password-file", type=Path, default=DEFAULT_PASSWORD_FILE)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stream and validate the extraction without connecting to Neo4j.",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete every node in the selected database before loading.",
    )
    return parser.parse_args()


def resolve_password(password_file: Path) -> str:
    password = os.environ.get("NEO4J_PASSWORD") or os.environ.get("NEO4J_PASS")
    if password:
        return password
    if password_file.exists():
        return password_file.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "no password: set NEO4J_PASSWORD or NEO4J_PASS, or provide "
        f"--password-file (currently {password_file})"
    )


def clean_text(value: Any) -> str:
    """Return a stripped string suitable for a Neo4j scalar property."""
    if value is None:
        return ""
    return str(value).strip()


def normalise_name(value: Any) -> str:
    """Normalise a doctrine name for matching related-doctrine references."""
    return " ".join(clean_text(value).casefold().split())


def stable_uid(prefix: str, values: list[Any]) -> str:
    """Build a compact deterministic identifier from JSON-compatible values."""
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def shape_citation(citation: dict[str, Any]) -> dict[str, Any]:
    parallel = [
        clean_text(value)
        for value in (citation.get("parallel_citations") or [])
        if clean_text(value)
    ]
    fields = {
        "case_name": clean_text(citation.get("case_name")),
        "volume": clean_text(citation.get("volume")),
        "reporter": clean_text(citation.get("reporter")),
        "page": clean_text(citation.get("page")),
        "pinpoint": clean_text(citation.get("pinpoint")),
        "court": clean_text(citation.get("court")),
        "year": clean_text(citation.get("year")),
        "parallel_citations": parallel,
    }
    fields["uid"] = stable_uid(
        "citation",
        [
            fields["case_name"],
            fields["volume"],
            fields["reporter"],
            fields["page"],
            fields["pinpoint"],
            fields["court"],
            fields["year"],
            parallel,
        ],
    )
    return fields


def shape_row(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one extraction record to the parameter shape used by LOAD_QUERY."""
    row_index = clean_text(record.get("row_index"))
    if not row_index:
        raise ValueError("record has no row_index")

    raw_doctrines = record.get("doctrines") or []
    if not isinstance(raw_doctrines, list):
        raise ValueError(f"record {row_index} has a non-list doctrines field")

    doctrines: list[dict[str, Any]] = []
    first_uid_by_name: dict[str, str] = {}
    for ordinal, raw_doctrine in enumerate(raw_doctrines, start=1):
        if not isinstance(raw_doctrine, dict):
            raise ValueError(f"record {row_index} doctrine {ordinal} is not an object")
        uid = f"{row_index}:doctrine:{ordinal}"
        name = clean_text(raw_doctrine.get("doctrine_name"))
        name_key = normalise_name(name)
        if name_key and name_key not in first_uid_by_name:
            first_uid_by_name[name_key] = uid

        elements = []
        for element_ordinal, raw_element in enumerate(
            raw_doctrine.get("elements") or [], start=1
        ):
            if not isinstance(raw_element, dict):
                continue
            elements.append(
                {
                    "uid": f"{uid}:element:{element_ordinal}",
                    "number": raw_element.get("element_number"),
                    "description": clean_text(raw_element.get("description")),
                }
            )

        citations = [
            shape_citation(raw_citation)
            for raw_citation in (raw_doctrine.get("supporting_citations") or [])
            if isinstance(raw_citation, dict)
        ]
        exceptions = [
            clean_text(value)
            for value in (raw_doctrine.get("exceptions") or [])
            if clean_text(value)
        ]
        doctrines.append(
            {
                "uid": uid,
                "ordinal": ordinal,
                "name": name,
                "type": clean_text(raw_doctrine.get("doctrine_type")),
                "area": clean_text(raw_doctrine.get("area_of_law")),
                "definition": clean_text(raw_doctrine.get("definition")),
                "exceptions": exceptions,
                "elements": elements,
                "citations": citations,
                "raw_related": raw_doctrine.get("related_doctrines") or [],
            }
        )

    for doctrine in doctrines:
        related = []
        seen_uids = set()
        for raw_name in doctrine.pop("raw_related"):
            name = clean_text(raw_name)
            name_key = normalise_name(name)
            if not name_key:
                continue
            target_uid = first_uid_by_name.get(name_key)
            if target_uid is None:
                target_uid = stable_uid("related", [row_index, name_key])
            if target_uid == doctrine["uid"] or target_uid in seen_uids:
                continue
            seen_uids.add(target_uid)
            related.append({"uid": target_uid, "name": name})
        doctrine["related"] = related

    return {
        "row_index": row_index,
        "summary": clean_text(record.get("source_text_summary")),
        "jurisdiction": clean_text(record.get("jurisdiction")),
        "doctrines": doctrines,
    }


def iter_rows(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield shaped records from a top-level JSON array without retaining it."""
    if not path.is_file():
        raise SystemExit(f"extraction JSON not found: {path}")
    with path.open("rb") as handle:
        for record_number, record in enumerate(ijson.items(handle, "item"), start=1):
            if limit is not None and record_number > limit:
                break
            if not isinstance(record, dict):
                raise ValueError(f"record {record_number} is not an object")
            yield shape_row(record)


def update_stats(stats: Counter[str], row: dict[str, Any]) -> None:
    stats["passages"] += 1
    stats["doctrines"] += len(row["doctrines"])
    for doctrine in row["doctrines"]:
        stats["elements"] += len(doctrine["elements"])
        stats["citations"] += len(doctrine["citations"])
        stats["related"] += len(doctrine["related"])
        if doctrine["area"]:
            stats["areas_assigned"] += 1
        if doctrine["area"] and doctrine["area"] not in AREAS_OF_LAW:
            stats[f"unexpected area: {doctrine['area']}"] += 1
        if doctrine["type"] and doctrine["type"] not in DOCTRINE_TYPES:
            stats[f"unexpected type: {doctrine['type']}"] += 1


def print_stats(prefix: str, stats: Counter[str]) -> None:
    core = ("passages", "doctrines", "elements", "citations", "related")
    print(prefix, " | ".join(f"{key}={stats[key]:,}" for key in core))
    for key in sorted(value for value in stats if value.startswith("unexpected ")):
        print(f"[warning] {key} ({stats[key]:,} doctrines)")


def load_batch(transaction: Any, rows: list[dict[str, Any]]) -> None:
    transaction.run(LOAD_QUERY, rows=rows).consume()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    stats: Counter[str] = Counter()
    if args.dry_run:
        for row in iter_rows(args.extraction, args.limit):
            update_stats(stats, row)
        print_stats("[dry-run]", stats)
        return

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, resolve_password(args.password_file)),
    )
    driver.verify_connectivity()
    try:
        with driver.session(database=args.neo4j_database) as session:
            if args.wipe:
                print("[wipe] deleting every node in the selected database")
                session.run(
                    "MATCH (n) CALL (n) { DETACH DELETE n } "
                    "IN TRANSACTIONS OF 10000 ROWS"
                ).consume()
            for statement in CONSTRAINTS:
                session.run(statement).consume()
            session.run("CALL db.awaitIndexes(600)").consume()

            batch: list[dict[str, Any]] = []
            for row in iter_rows(args.extraction, args.limit):
                update_stats(stats, row)
                batch.append(row)
                if len(batch) >= args.batch_size:
                    session.execute_write(load_batch, batch)
                    batch = []
                    if stats["passages"] % 10_000 == 0:
                        print_stats("[load]", stats)
            if batch:
                session.execute_write(load_batch, batch)

            print_stats("[loaded]", stats)
            counts = dict(session.run(COUNT_QUERY).single())
            print(f"[graph] {counts}")
            if counts["passages"] != stats["passages"]:
                raise RuntimeError(
                    "passage count mismatch: "
                    f"input={stats['passages']}, graph={counts['passages']}"
                )
            if counts["contains_doctrine"] != stats["doctrines"]:
                raise RuntimeError(
                    "doctrine relationship count mismatch: "
                    f"input={stats['doctrines']}, "
                    f"graph={counts['contains_doctrine']}"
                )
            if counts["has_element"] != stats["elements"]:
                raise RuntimeError(
                    "element relationship count mismatch: "
                    f"input={stats['elements']}, graph={counts['has_element']}"
                )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
