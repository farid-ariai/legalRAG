#!/usr/bin/env python3
"""Structured entity extraction for Housing Statute QA passages (benchmark 3).

Mirrors the vLLM + JSON-schema workflow of ``Codes/Structured_extraction.py``
(MBE) and ``Codes/legal_rag_bench_structured_extraction.py`` (Legal RAG Bench),
with the schema realigned to U.S. **state housing and eviction statutes** from
``reglab/housing_qa``.

Entity design deliberately parallels the MBE knowledge graph so the three
benchmarks stay structurally comparable:

    MBE                     Housing QA
    ------------------      ---------------------------
    Doctrine                StatutoryProvision (topic + rule statement)
    Element                 RuleCondition (requirement / precondition)
    Citation                StatuteCitation
    AreaOfLaw               area_of_law
    Passage.summary         source_text_summary

Why this benchmark is retrieval-first
-------------------------------------
Housing QA answers are binary yes/no and class-imbalanced. The raw source had a
64.8% always-"No" baseline; the filtered clean cohort has a 67.32% baseline, so
downstream QA accuracy is a blunt instrument for the
paper's claim. On Legal RAG Bench, gold recall varying 14%->28% moved QA accuracy
only 71->73, i.e. retrieval quality largely stopped propagating to answers. This
extraction therefore exists to support **IR metrics** (Recall/NDCG/MAP over gold
statutes) as the primary measurement, with QA accuracy secondary and always
reported against the majority baseline.

Scale note: the default three-state subset is 59,306 statutes, roughly 23x the
MBE corpus, so this is the dominant cost of benchmark 3. The script writes
incrementally and resumes, because a job this long will be interrupted.

Input is the raw ``statutes.tsv`` from the dataset repo (``datasets`` >= 3.0
refuses to run the repo's loading script):

    curl -L -o statutes.tsv.zip \\
      https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data/statutes.tsv.zip

Example
-------
    python Codes/housing_qa_structured_extraction.py \\
        --statutes statutes.tsv --states Wisconsin Delaware Hawaii
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Structured_data"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # same extractor role as MBE Stage 1
# 12288 rather than 8192: Qwen2.5-7B supports 32K, and the extra room helps
# twice over -- the bulk pass gets a ~9.6K prompt budget instead of 5.6K (so
# fewer long statutes are trimmed), and the retry pass has space for a doubled
# output cap. Costs ~30% throughput via reduced KV-cache concurrency, which is
# a good trade against silently losing the longest statutes.
DEFAULT_MAX_MODEL_LEN = 12288
DEFAULT_MAX_NUM_SEQS = 16
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_MAX_TOKENS = 2560
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SAVE_EVERY = 200
DEFAULT_STATES = ("Wisconsin", "Delaware", "Hawaii")

# The corpus is each state's ENTIRE code, not its housing chapters: only ~1.8%
# of Wisconsin's 17,739 sections are landlord-tenant related. The knowledge graph
# must cover the same pool that dense and BM25 retrieve over, otherwise KG
# retrieval gets an oracle-like filter that can only return housing statutes --
# vastly more likely to be gold -- and the comparison collapses. So the schema is
# deliberately domain-GENERAL, exactly as the MBE schema is general across bar
# subjects, with subject captured in `area_of_law` rather than baked into the
# provision types.
PROVISION_TYPES = [
    "Definition",
    "Scope or Application",
    "Duty or Obligation",
    "Right or Entitlement",
    "Prohibition",
    "Procedure or Process",
    "Time Limit or Deadline",
    "Monetary Limit or Fee",
    "Remedy",
    "Penalty or Sanction",
    "Exemption or Exception",
    "Administrative or Agency Power",
    "Enforcement Mechanism",
    "Cross-Reference or Incorporation",
]

ProvisionType = Literal[
    "Definition",
    "Scope or Application",
    "Duty or Obligation",
    "Right or Entitlement",
    "Prohibition",
    "Procedure or Process",
    "Time Limit or Deadline",
    "Monetary Limit or Fee",
    "Remedy",
    "Penalty or Sanction",
    "Exemption or Exception",
    "Administrative or Agency Power",
    "Enforcement Mechanism",
    "Cross-Reference or Incorporation",
]

# Broad subject areas of a state code. "Landlord-Tenant and Housing" is one area
# among many; forcing every section into a housing category is what produced
# nonsense on the first smoke test (sovereignty statutes classified as housing).
AREAS_OF_LAW = [
    "Landlord-Tenant and Housing",
    "Property and Real Estate",
    "Contracts and Commercial",
    "Civil Procedure",
    "Criminal Law and Procedure",
    "Family and Domestic Relations",
    "Government and Administration",
    "Taxation and Finance",
    "Health and Safety",
    "Employment and Labor",
    "Environment and Natural Resources",
    "Transportation and Vehicles",
    "Business and Corporations",
    "Civil Rights and Discrimination",
    "Courts and Judiciary",
    "Elections",
    "Education",
    "Other",
]

AreaOfLaw = Literal[
    "Landlord-Tenant and Housing",
    "Property and Real Estate",
    "Contracts and Commercial",
    "Civil Procedure",
    "Criminal Law and Procedure",
    "Family and Domestic Relations",
    "Government and Administration",
    "Taxation and Finance",
    "Health and Safety",
    "Employment and Labor",
    "Environment and Natural Resources",
    "Transportation and Vehicles",
    "Business and Corporations",
    "Civil Rights and Discrimination",
    "Courts and Judiciary",
    "Elections",
    "Education",
    "Other",
]

SYSTEM_MESSAGE = """You are a legal information extraction specialist working with U.S. state statutes. Your task is to extract structured legal information from individual sections of a state code.

The corpus spans an entire state code, so a passage may concern any subject: landlord-tenant law, taxation, criminal procedure, elections, natural resources, and so on. Classify each passage on its own terms. Do NOT assume a passage concerns housing or landlord-tenant law, and never force a housing interpretation onto a section that is about something else.

These passages are statutory text, not case summaries. They typically state:
- duties, rights, prohibitions, and entitlements;
- procedural requirements and administrative processes;
- definitions, scope provisions, and exemptions;
- time limits, monetary limits, penalties, and remedies.

When analysing a passage, you must:

1. Identify each distinct legal provision: rule, requirement, right, duty, procedure, or definition.
2. Classify each with the provided provision types, and assign the subject area from the provided list.
3. State the core legal proposition concisely, preserving numbers, deadlines, and dollar limits exactly as written.
4. Extract the conditions, preconditions, elements, or triggering circumstances that govern when the provision applies.
5. Extract the actors affected, e.g. landlord, tenant, court, sheriff, clerk, agency, employer, taxpayer, driver.
6. Extract statutory cross-references in U.S. form, including code name and section number.
7. Note exceptions, exemptions, and expressly related provisions.

Important guidelines:
- Deadlines, notice periods, dollar amounts, day counts, and percentages are the most legally operative details in statutes. Never round, generalise, or omit them; record them verbatim.
- Many sections are purely definitional or purely administrative. That is normal; classify them accurately rather than inventing substantive rules.
- If a passage has no extractable legal provision, return an empty list rather than fabricating one.
- If information is absent, use null or an empty list rather than inferring.
- Do not import knowledge of other states' law; extract only what this passage states.

Output must strictly conform to the provided JSON schema."""


class StatuteCitation(BaseModel):
    """Statutory cross-reference appearing in the passage."""

    code_name: Optional[str] = Field(
        None, description="Code or act name, e.g. 'Wis. Stat.' or 'Del. Code Ann. tit. 25'."
    )
    section: Optional[str] = Field(
        None, description="Section or subsection number, e.g. '704.28' or '5514(a)'."
    )
    description: Optional[str] = Field(
        None, description="What the cross-referenced provision covers, if stated."
    )


class RuleCondition(BaseModel):
    """A condition, precondition, element, or triggering circumstance."""

    number: Optional[int] = Field(
        None, description="Ordinal if the passage enumerates the conditions."
    )
    description: str = Field(..., description="The condition as stated in the passage.")


class StatutoryProvision(BaseModel):
    """A discrete legal provision extracted from one statute section."""

    topic_name: str = Field(
        ..., description="Short name of the provision, right, duty, procedure, or definition."
    )
    topic_type: ProvisionType = Field(..., description="Provision classification.")
    area_of_law: AreaOfLaw = Field(
        ...,
        description=(
            "Subject area of the state code this provision belongs to. Assign on the "
            "passage's own terms; most sections are NOT landlord-tenant."
        ),
    )
    concise_rule: str = Field(
        ...,
        description=(
            "Concise statement of the rule. Preserve exact deadlines, notice "
            "periods, day counts, and dollar amounts."
        ),
    )
    # REQUIRED, not Optional: under constrained decoding an Optional list is
    # satisfied by emitting null, and this field was None in 385/385 provisions
    # in testing despite the amounts being present in the text. Deadlines and
    # dollar limits are the operative content of statutes, so losing them would
    # hollow out the graph silently.
    deadlines_or_amounts: list[str] = Field(
        ..., max_length=20,
        description=(
            "Verbatim deadlines, notice periods, day counts, percentages, or dollar "
            "limits stated in the passage, e.g. '14 days', 'two months rent'."
        ),
    )
    actors_affected: list[str] = Field(
        ..., max_length=15,
        description="Participants directly affected, e.g. landlord, tenant, court, sheriff.",
    )
    conditions: list[RuleCondition] = Field(
        ..., max_length=25,
        description="Conditions, elements, or preconditions stated in the passage."
    )
    cross_references: list[StatuteCitation] = Field(
        ..., max_length=25,
        description="Statutory cross-references supporting or qualifying this rule."
    )
    exceptions: list[str] = Field(
        ..., max_length=15,
        description="Express exceptions, exemptions, or qualifications; [] if none."
    )
    related_topics: list[str] = Field(
        ..., max_length=15,
        description="Other provisions expressly linked to this rule; [] if none."
    )


class HousingStatuteExtraction(BaseModel):
    """Root structured output for one Housing QA statute section."""

    passage_role: Optional[str] = Field(
        None,
        description=(
            "High-level role, e.g. 'definition section', 'procedural requirement', "
            "'substantive duty', 'penalty provision', or 'scope and exemptions'."
        ),
    )
    rules: list[StatutoryProvision] = Field(
        ..., max_length=20,
        description="All distinct legal provisions extracted from the passage."
    )
    jurisdiction: Optional[str] = Field(
        None, description="U.S. state or territory whose law this passage states."
    )
    source_text_summary: str = Field(
        ..., description="Brief summary of what the statute section covers."
    )


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--statutes", type=Path, required=True, help="statutes.tsv")
    p.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    p.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--batch-size", type=int, default=256, help="Rows per generate() call.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Process a RANDOM sample of N rows (seeded). Prefer this over --limit "
            "for smoke tests: --limit takes the opening sections of the code, which "
            "are sovereignty/administrative boilerplate and badly unrepresentative "
            "of the corpus."
        ),
    )
    p.add_argument("--sample-seed", type=int, default=20260418)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--no-retry",
        action="store_true",
        help="Skip the retry pass over rows whose JSON was truncated.",
    )
    return p.parse_args()


def output_path_for(args: argparse.Namespace) -> Path:
    if args.out:
        return args.out
    states = safe_name("_".join(sorted(s.lower() for s in args.states)))
    model = safe_name(args.model.split("/")[-1])
    return OUT_DIR / f"housing_qa_entities__model_{model}__states_{states}.json"


def load_statutes(args: argparse.Namespace) -> pd.DataFrame:
    wanted = {s.lower() for s in args.states}
    frame = pd.read_csv(
        args.statutes, sep="\t", dtype=str, keep_default_na=False, on_bad_lines="skip"
    )
    frame["state"] = frame["state"].str.lower()
    frame = frame[frame["state"].isin(wanted)].reset_index(drop=True)
    frame["idx"] = frame["idx"].astype(str)
    if args.sample:
        frame = frame.sample(
            n=min(args.sample, len(frame)), random_state=args.sample_seed
        ).reset_index(drop=True)
    elif args.limit:
        frame = frame.head(args.limit).copy()
    return frame


def build_runtime(args: argparse.Namespace):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
    )
    schema = HousingStatuteExtraction.model_json_schema()
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        # See the note in legal_rag_bench_experiment.build_judge_sampling_params:
        # the JSON grammar allows unlimited whitespace between tokens and models
        # can loop on it. The extraction failures reported errors at line 3,300+
        # of output, which is the signature of thousands of repeated "\n \t"
        # sequences rather than repeated content. maxItems bounded the content
        # runaway; this bounds the whitespace one.
        structured_outputs=StructuredOutputsParams(
            json=schema, disable_any_whitespace=True
        ),
    )
    return llm, sampling


def build_user_message(row: pd.Series, text: str | None = None) -> str:
    body = row["text"].strip() if text is None else text
    return (
        f"### Jurisdiction\n{row['state'].title()}\n\n"
        f"### Citation\n{row.get('citation', '') or 'unknown'}\n\n"
        f"### Statute Text\n{body}\n\n"
        "### Task\nExtract the structured legal information described in the system message."
    )


def render_prompt(tokenizer, user_message: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompt_for_row(tokenizer, row: pd.Series, *, budget_tokens: int) -> str:
    """Render a prompt, shrinking the statute text until it fits the context.

    Some sections of a state code are very long; without this, vLLM raises
    ``VLLMValidationError: ... prompt contains at least 8193 input tokens`` and
    kills the whole run rather than skipping the row. Only the statute body is
    trimmed -- the system message and the citation header carry the instructions
    and provenance, so they must survive intact.
    """
    text = row["text"].strip()
    prompt = render_prompt(tokenizer, build_user_message(row, text))
    if len(tokenizer.encode(prompt)) <= budget_tokens:
        return prompt

    # Binary search on the character cut that fits, rather than a fixed guess:
    # token density varies a lot between dense statutory prose and tables.
    lo, hi = 0, len(text)
    best = render_prompt(tokenizer, build_user_message(row, text[:200]))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = render_prompt(tokenizer, build_user_message(row, text[:mid]))
        if len(tokenizer.encode(candidate)) <= budget_tokens:
            best, lo = candidate, mid
        else:
            hi = mid - 1
    return best


def empty_extraction() -> dict:
    return {"passage_role": None, "rules": [], "jurisdiction": None, "source_text_summary": ""}


def parse_extraction(raw_text: str) -> tuple[dict, Optional[str]]:
    try:
        return HousingStatuteExtraction.model_validate(json.loads(raw_text)).model_dump(), None
    except Exception as exc:  # noqa: BLE001 - record and continue
        return empty_extraction(), f"{type(exc).__name__}: {exc}"


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text())
    except Exception:
        print(f"[warn] could not parse existing {path}; starting fresh")
        return {}
    return {str(r["statute_idx"]): r for r in records if "statute_idx" in r}


def write_records(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(list(records.values()), indent=1))
    tmp.replace(path)  # atomic: a crash mid-write must not corrupt hours of work


def main() -> None:
    args = parse_args()
    out_path = output_path_for(args)
    statutes = load_statutes(args)
    records = load_existing(out_path)

    todo = statutes[~statutes["idx"].isin(records.keys())].reset_index(drop=True)
    print(
        f"[data] {len(statutes)} statutes in {args.states}; "
        f"{len(records)} already extracted; {len(todo)} to do"
    )
    print(f"[out]  {out_path}")
    # Resume must not skip the retry pass. Failed rows ARE present in `records`
    # (carrying a parse_error), so `todo` is empty on a re-run even when
    # thousands of rows still need retrying -- returning here would make it
    # impossible to reprocess them after a sampling-params fix, which is exactly
    # the situation the disable_any_whitespace change created.
    pending_retry = [k for k, v in records.items() if v.get("parse_error")]
    if todo.empty and not pending_retry:
        print("[done] nothing left to extract")
        return
    if todo.empty:
        print(f"[resume] no new rows; {len(pending_retry)} failed rows to retry")

    llm, sampling = build_runtime(args)
    tokenizer = llm.get_tokenizer()
    # Leave room for the generated JSON plus a small safety margin for chat
    # template tokens the tokenizer may add.
    budget = args.max_model_len - args.max_tokens - 64
    print(f"[prompt] token budget per prompt: {budget}")

    failures = 0
    processed = 0
    for start in tqdm(range(0, len(todo), args.batch_size), desc="batches"):
        chunk = todo.iloc[start : start + args.batch_size]
        prompts = [
            build_prompt_for_row(tokenizer, r, budget_tokens=budget)
            for _, r in chunk.iterrows()
        ]
        outputs = llm.generate(prompts, sampling)

        for (_, row), out in zip(chunk.iterrows(), outputs):
            payload, error = parse_extraction(out.outputs[0].text)
            if error:
                failures += 1
            records[str(row["idx"])] = {
                "statute_idx": str(row["idx"]),
                "state": row["state"],
                "citation": row.get("citation", ""),
                "path": row.get("path", ""),
                "parse_error": error,
                **payload,
            }
            processed += 1

        if processed % args.save_every < args.batch_size:
            write_records(out_path, records)

    write_records(out_path, records)

    # Retry pass. Remaining failures are JSON truncated mid-string, i.e. the
    # model needed more output tokens than the cap allowed. That loss is NOT
    # random: it concentrates on the longest, most rule-dense sections, which is
    # exactly where gold passages live. Leaving them out would make those
    # statutes unreachable by KG retrieval while dense and BM25 still find them
    # -- a systematic handicap to the KG condition that would read as a result.
    # Retrying only the failures keeps the bulk run fast.
    retry_ids = [k for k, v in records.items() if v.get("parse_error")]
    if retry_ids and not args.no_retry:
        retry_rows = statutes[statutes["idx"].isin(retry_ids)]
        big_tokens = min(args.max_tokens * 2, args.max_model_len - 1024)
        print(f"\n[retry] {len(retry_rows)} failed rows at max_tokens={big_tokens}")
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        retry_sampling = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=big_tokens,
            structured_outputs=StructuredOutputsParams(
                json=HousingStatuteExtraction.model_json_schema()
            ),
        )
        retry_budget = args.max_model_len - big_tokens - 64
        recovered = 0
        for start in range(0, len(retry_rows), args.batch_size):
            chunk = retry_rows.iloc[start : start + args.batch_size]
            prompts = [
                build_prompt_for_row(tokenizer, r, budget_tokens=retry_budget)
                for _, r in chunk.iterrows()
            ]
            outputs = llm.generate(prompts, retry_sampling)
            for (_, row), out in zip(chunk.iterrows(), outputs):
                payload, error = parse_extraction(out.outputs[0].text)
                if not error:
                    recovered += 1
                records[str(row["idx"])].update({"parse_error": error, **payload})
        write_records(out_path, records)
        failures -= recovered
        print(f"[retry] recovered {recovered}/{len(retry_rows)}")

    print(f"[done] {len(records)} records written to {out_path}")
    print(f"[done] {failures} parse failures ({100 * failures / max(processed, 1):.2f}%)")


if __name__ == "__main__":
    main()
