import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import warnings

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

warnings.filterwarnings('ignore')

import torch

torch.cuda.empty_cache()

barexam_passage = pd.read_csv('barexam_passage.csv')


SYSTEM_MESSAGE = """You are a legal information extraction specialist with expertise in US Bar Exam content and legal doctrine analysis. Your task is to extract structured information from legal texts, particularly those related to the US Bar Exam.

When analyzing legal text, you must:

1. **Identify Legal Doctrines**: Extract the name of each legal doctrine, rule, claim, or test mentioned in the text. Common examples include "res ipsa loquitur," "strict liability," "proximate cause," etc.

2. **Classify the Doctrine**:
   - Determine the doctrine type (Common Law Doctrine, Statutory Rule, Constitutional Principle, Evidentiary Rule, Procedural Rule, Affirmative Defense, Cause of Action, Legal Test, or Presumption)
   - Identify the area of law (Torts, Contracts, Criminal Law, Criminal Procedure, Constitutional Law, Evidence, Real Property, Civil Procedure, Business Associations, Trusts and Estates, Family Law, Secured Transactions, or Conflict of Laws)

3. **Extract the Definition**: Capture the core definition or description of what the doctrine establishes or enables.

4. **Identify Elements/Criteria**: Extract all numbered elements, criteria, or requirements that must be established to apply the doctrine. Preserve the exact language and numbering from the source text.

5. **Parse Case Citations**: For each case citation, extract:
   - Case name (e.g., "Smith v. Jones")
   - Volume number
   - Reporter abbreviation (e.g., F.2d, N.Y.S.2d, So.2d, N.E.2d)
   - Page number and pinpoint citation if provided
   - Court (e.g., "2d Cir.", "Fla. 5th DCA", "N.Y.")
   - Year of decision
   - Any parallel citations

6. **Note Relationships**: Identify any related doctrines, exceptions, or cross-references mentioned in the text.

**Citation Format Examples**:
- Federal: "907 F.2d 299, 302 (2d Cir. 1990)" → volume: 907, reporter: F.2d, page: 299, pinpoint: 302, court: 2d Cir., year: 1990
- State: "7 N.Y.3d 203, 818 N.Y.S.2d 792" → volume: 7, reporter: N.Y.3d, page: 203, parallel: 818 N.Y.S.2d 792
- District Court of Appeal: "681 So.2d 859, 861 (Fla. 5th DCA 1996)"

**Important Guidelines**:
- Be precise with legal terminology; do not paraphrase element language
- Distinguish between primary citations and parenthetical citations (quoting other cases)
- If a citation references another case within parentheses, extract both as separate supporting citations
- Identify the jurisdiction based on the courts cited (Federal, specific state)
- If information is not explicitly stated, use null/None rather than inferring

Output must strictly conform to the provided JSON schema."""


class CaseCitation(BaseModel):
    """
    Represents a legal case citation.
    """
    case_name: str = Field(
        ..., description="The name of the case (e.g., 'St. Paul Fire & Marine Ins. Co. v. City of New York')."
    )
    volume: Optional[str] = Field(
        None, description="The volume number of the reporter."
    )
    reporter: Optional[str] = Field(
        None, description="The reporter abbreviation (e.g., 'F.2d', 'N.Y.S.2d', 'So.2d')."
    )
    page: Optional[str] = Field(
        None, description="The starting page number of the case."
    )
    pinpoint: Optional[str] = Field(
        None, description="The specific page being cited, if different from starting page."
    )
    court: Optional[str] = Field(
        None, description="The court that decided the case (e.g., '2d Cir.', 'Fla. 5th DCA')."
    )
    year: Optional[str] = Field(
        None, description="The year the case was decided."
    )
    parallel_citations: Optional[List[str]] = Field(
        None, description="Any parallel citations for the same case in different reporters."
    )


class LegalElement(BaseModel):
    """
    Represents a single element or criterion of a legal doctrine or claim.
    """
    element_number: int = Field(
        ..., description="The sequential number of this element (1, 2, 3, etc.)."
    )
    description: str = Field(
        ..., description="The text describing what must be proven or established for this element."
    )


legal_areas = [
    "Torts",
    "Contracts",
    "Criminal Law",
    "Criminal Procedure",
    "Constitutional Law",
    "Evidence",
    "Real Property",
    "Civil Procedure",
    "Business Associations",
    "Trusts and Estates",
    "Family Law",
    "Secured Transactions",
    "Conflict of Laws"
]

doctrine_types = [
    "Common Law Doctrine",
    "Statutory Rule",
    "Constitutional Principle",
    "Evidentiary Rule",
    "Procedural Rule",
    "Affirmative Defense",
    "Cause of Action",
    "Legal Test",
    "Presumption"
]


class LegalDoctrine(BaseModel):
    """
    Represents a legal doctrine, rule, or claim from the US Bar Exam.
    """
    doctrine_name: str = Field(
        ..., description="The name of the legal doctrine, rule, or type of claim (e.g., 'res ipsa loquitur', 'strict liability products liability')."
    )
    doctrine_type: str = Field(
        ...,
        description="The type/category of the legal doctrine.",
        enum=doctrine_types
    )
    area_of_law: str = Field(
        ...,
        description="The area of law this doctrine belongs to.",
        enum=legal_areas
    )
    definition: str = Field(
        ..., description="A brief definition or description of what the doctrine does or establishes."
    )
    elements: List[LegalElement] = Field(
        ..., description="The numbered elements or criteria that must be established to apply this doctrine."
    )
    supporting_citations: List[CaseCitation] = Field(
        ..., description="List of case citations that support or define this doctrine."
    )
    exceptions: Optional[List[str]] = Field(
        None, description="Any notable exceptions to the doctrine."
    )
    related_doctrines: Optional[List[str]] = Field(
        None, description="Names of related legal doctrines or concepts."
    )


class BarExamExtraction(BaseModel):
    """
    Root model for extracting structured legal information from US Bar Exam texts.
    """
    doctrines: List[LegalDoctrine] = Field(
        ..., description="List of legal doctrines extracted from the text."
    )
    jurisdiction: Optional[str] = Field(
        None, description="The primary jurisdiction referenced (e.g., 'Federal', 'New York', 'Florida')."
    )
    source_text_summary: str = Field(
        ..., description="A brief summary of what the source text discusses."
    )


# Initialize vLLM with structured outputs support
llm = LLM(
    model=MODEL_NAME,
    max_model_len=10000,
    gpu_memory_utilization=0.95,
    dtype="auto",
    enable_prefix_caching=True,
    disable_log_stats=True,
    max_num_seqs=40,
    seed=22,
    enforce_eager=True,
    tensor_parallel_size=1,
)

# Create JSON schema from Pydantic model
json_schema = BarExamExtraction.model_json_schema()

# Set up structured outputs parameters
structured_outputs_params = StructuredOutputsParams(json=json_schema)
sampling_params = SamplingParams(
    structured_outputs=structured_outputs_params,
    max_tokens=4096,
    temperature=0,
)


def extract_single(document: str) -> dict:
    """Extract legal information from a single document."""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_MESSAGE}<|eot_id|><|start_header_id|>user<|end_header_id|>

{document}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    return json.loads(outputs[0].outputs[0].text)