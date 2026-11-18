"""Standalone executable Cypher safety contract for Housing QA retrieval."""

from __future__ import annotations

import re


MAX_CASE_CLAUSES = 10
UNESCAPED_WORD_APOSTROPHE = re.compile(r"(?<!\\)(?<=[A-Za-z])'(?=[A-Za-z])")
RAW_CONDITION_NODE_COLLECTION = re.compile(
    r"collect\s*\(\s*(?:distinct\s+)?c\s*\)\s+AS\s+conditions",
    re.IGNORECASE,
)


def strip_string_literals(query: str) -> str:
    return re.sub(r'''(["'])(?:\\.|(?!\1).)*\1''', '""', query, flags=re.DOTALL)


def mask_cypher_literals_and_comments(query: str) -> str:
    """Mask quoted strings and comments while preserving character offsets."""
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


def unsafe_returned_graph_values(query: str) -> list[str]:
    """Find bare graph entities or node-collection aliases in the final RETURN."""
    masked = mask_cypher_literals_and_comments(query)
    return_matches = list(re.finditer(r"\bRETURN\b", masked, re.IGNORECASE))
    if not return_matches:
        return []
    tail = masked[return_matches[-1].end():]
    terminator = re.search(r"\b(?:ORDER\s+BY|LIMIT|SKIP|UNION)\b", tail, re.IGNORECASE)
    clause = tail[:terminator.start()] if terminator else tail
    graph_variables = set(re.findall(r"\(\s*([A-Za-z_]\w*)\s*:", masked))
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
        safe_clause = re.sub(rf"\b{escaped}\s*\.\s*[A-Za-z_]\w*", "", safe_clause)
    return [
        variable
        for variable in sorted(graph_variables | graph_collection_aliases)
        if re.search(rf"(?<![\w.]){re.escape(variable)}(?![\w.])", safe_clause)
    ]


def validate_read_only_cypher(query: str, *, expected_limit: int = 20) -> str:
    """Validate the frozen executable safety and shape contract."""
    if type(expected_limit) is not int or expected_limit <= 0:
        raise ValueError("expected LIMIT must be a positive integer")
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
    unsafe = unsafe_returned_graph_values(body)
    if unsafe:
        raise ValueError("final RETURN contains raw graph values: " + ", ".join(unsafe))
    case_count = len(re.findall(r"(?i)\bCASE\b", scrubbed))
    if case_count > MAX_CASE_CLAUSES:
        raise ValueError(f"query has {case_count} CASE clauses; maximum is {MAX_CASE_CLAUSES}")
    required_patterns = (
        r"\bMATCH\b",
        r"\bRETURN\b",
        rf"\bLIMIT\s+{expected_limit}\b",
    )
    for required in required_patterns:
        if not re.search(required, scrubbed, flags=re.IGNORECASE):
            raise ValueError(f"query is missing required pattern {required}")
    return body
