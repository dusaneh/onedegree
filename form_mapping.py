"""
Form-to-Canonical Question Mapping System.

Maps each form's questions to canonical questions with similarity scores (1-5).
Mappings are stored per-question in form_structure.questions[].mappings.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from gemini_client import get_and_parse_structured_response
from canonical_questions import (
    ModelTokenConfig,
    GEMINI_2_5_FLASH,
    GEMINI_2_5_PRO,
    CANONICAL_MAPPINGS_PATH,
    estimate_tokens,
    get_model_config,
    load_cached_result as load_canonical_version,
    list_canonical_versions,
)
from extract_metadata import QuestionMapping
from database import (
    load_form_records_db,
    save_form_record_db,
    save_all_form_records_db,
)

# Logger Setup
logger = logging.getLogger("form_mapping")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Constants
DATA_DIR = Path(__file__).parent / "data"
JSONL_PATH = DATA_DIR / "form_metadata.jsonl"


# ============================================================================
# Output Schemas for LLM
# ============================================================================

class QuestionMappingOutput(BaseModel):
    """LLM output for a single question mapping."""
    question_number: str = Field(description="The question number from the form.")
    canonical_question_id: Optional[str] = Field(
        default=None,
        description="The canonical question ID (e.g., 'cq_001') or null if no match."
    )
    similarity_score: int = Field(
        ge=1, le=5,
        description="Similarity score 1-5 (5=exact semantic match, 1=no match)."
    )


class FormMappingResult(BaseModel):
    """LLM output for entire form mapping."""
    opportunity_id: str = Field(description="The opportunity ID of the form being mapped.")
    question_mappings: List[QuestionMappingOutput] = Field(
        description="List of mappings for each question in the form."
    )


# ============================================================================
# Stats Classes
# ============================================================================


@dataclass
class FormMappingStats:
    """Stats for a single form's mapping."""
    opportunity_id: str
    total_questions: int = 0
    questions_mapped: int = 0  # Questions that got a mapping response
    questions_with_match: int = 0  # Score 3-5 (has canonical_question_id)
    questions_no_match: int = 0  # Score 1-2 (no canonical_question_id)
    score_distribution: Dict[int, int] = dataclass_field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    canonical_ids_matched: List[str] = dataclass_field(default_factory=list)
    unmapped_question_numbers: List[str] = dataclass_field(default_factory=list)

    @property
    def match_rate(self) -> float:
        """Percentage of questions with a match (score 3-5)."""
        if self.questions_mapped == 0:
            return 0.0
        return self.questions_with_match / self.questions_mapped * 100

    @property
    def avg_score(self) -> float:
        """Average similarity score."""
        if self.questions_mapped == 0:
            return 0.0
        total = sum(score * count for score, count in self.score_distribution.items())
        return total / self.questions_mapped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "total_questions": self.total_questions,
            "questions_mapped": self.questions_mapped,
            "questions_with_match": self.questions_with_match,
            "questions_no_match": self.questions_no_match,
            "match_rate": f"{self.match_rate:.1f}%",
            "avg_score": f"{self.avg_score:.2f}",
            "score_distribution": self.score_distribution,
            "canonical_ids_matched": len(set(self.canonical_ids_matched)),
            "unmapped_question_numbers": self.unmapped_question_numbers,
        }

    def __str__(self) -> str:
        return (
            f"Form {self.opportunity_id}: {self.questions_with_match}/{self.questions_mapped} matched "
            f"({self.match_rate:.1f}%), avg_score={self.avg_score:.2f}"
        )


@dataclass
class RunMappingStats:
    """Aggregated stats across multiple form mappings."""
    forms_processed: int = 0
    forms_successful: int = 0
    forms_failed: int = 0
    total_questions: int = 0
    total_questions_with_match: int = 0
    total_questions_no_match: int = 0
    score_distribution: Dict[int, int] = dataclass_field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    all_canonical_ids_matched: set = dataclass_field(default_factory=set)
    form_stats: List[FormMappingStats] = dataclass_field(default_factory=list)
    errors: List[str] = dataclass_field(default_factory=list)

    @property
    def overall_match_rate(self) -> float:
        """Overall percentage of questions with a match."""
        total = self.total_questions_with_match + self.total_questions_no_match
        if total == 0:
            return 0.0
        return self.total_questions_with_match / total * 100

    @property
    def overall_avg_score(self) -> float:
        """Overall average similarity score."""
        total_mapped = sum(self.score_distribution.values())
        if total_mapped == 0:
            return 0.0
        total = sum(score * count for score, count in self.score_distribution.items())
        return total / total_mapped

    def add_form_stats(self, form_stats: FormMappingStats) -> None:
        """Add a form's stats to the aggregate."""
        self.form_stats.append(form_stats)
        self.forms_processed += 1
        self.forms_successful += 1
        self.total_questions += form_stats.total_questions
        self.total_questions_with_match += form_stats.questions_with_match
        self.total_questions_no_match += form_stats.questions_no_match
        for score, count in form_stats.score_distribution.items():
            self.score_distribution[score] += count
        self.all_canonical_ids_matched.update(form_stats.canonical_ids_matched)

    def add_error(self, opportunity_id: str, error: str) -> None:
        """Record a failed form mapping."""
        self.forms_processed += 1
        self.forms_failed += 1
        self.errors.append(f"{opportunity_id}: {error}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forms_processed": self.forms_processed,
            "forms_successful": self.forms_successful,
            "forms_failed": self.forms_failed,
            "total_questions": self.total_questions,
            "total_questions_with_match": self.total_questions_with_match,
            "total_questions_no_match": self.total_questions_no_match,
            "overall_match_rate": f"{self.overall_match_rate:.1f}%",
            "overall_avg_score": f"{self.overall_avg_score:.2f}",
            "score_distribution": self.score_distribution,
            "unique_canonical_ids_matched": len(self.all_canonical_ids_matched),
            "errors": self.errors,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the run stats."""
        print(f"\n{'='*50}")
        print("MAPPING RUN SUMMARY")
        print(f"{'='*50}")
        print(f"Forms: {self.forms_successful}/{self.forms_processed} successful")
        if self.forms_failed > 0:
            print(f"  Failed: {self.forms_failed}")
        print()
        print(f"Questions: {self.total_questions_with_match + self.total_questions_no_match} mapped")
        print(f"  With match (score 3-5): {self.total_questions_with_match}")
        print(f"  No match (score 1-2):   {self.total_questions_no_match}")
        print(f"  Match rate: {self.overall_match_rate:.1f}%")
        print(f"  Avg score:  {self.overall_avg_score:.2f}")
        print()
        print("Score distribution:")
        for score in range(5, 0, -1):
            count = self.score_distribution[score]
            total = sum(self.score_distribution.values())
            pct = (count / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {score}: {count:4d} ({pct:5.1f}%) {bar}")
        print()
        print(f"Unique canonical questions matched: {len(self.all_canonical_ids_matched)}")
        if self.errors:
            print(f"\nErrors:")
            for err in self.errors:
                print(f"  - {err}")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_canonical_questions(
    version_id: Optional[str] = None,
    cache_path: Path = CANONICAL_MAPPINGS_PATH
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load canonical questions with IDs from the database.

    Args:
        version_id: Specific version to load, or None for latest
        cache_path: Deprecated - kept for backwards compatibility

    Returns:
        Tuple of (version_id, list of canonical question dicts)

    Raises:
        ValueError: If version not found or no versions exist
    """
    # Use the versioned loader from canonical_questions module (now database-backed)
    result = load_canonical_version(version_id)

    if result is None:
        if version_id:
            # List available versions for helpful error
            versions = list_canonical_versions()
            available = [v["version_id"] for v in versions]
            raise ValueError(
                f"Canonical version '{version_id}' not found.\n"
                f"Available versions: {available}"
            )
        else:
            raise ValueError("No canonical question versions found in database.")

    # Convert to list of dicts
    canonical_questions = [cq.model_dump() for cq in result.canonical_questions]

    # Verify all questions have IDs
    for i, cq in enumerate(canonical_questions):
        if not cq.get("canonical_question_id"):
            raise ValueError(
                f"Canonical question at index {i} missing 'canonical_question_id'.\n"
                "Re-run 'python src/canonical_questions.py --force' to regenerate with IDs."
            )

    logger.info(
        f"Loaded {len(canonical_questions)} canonical questions (version: {result.version_id})"
    )
    return result.version_id, canonical_questions


def load_form_records(jsonl_path: Path = JSONL_PATH) -> List[Dict[str, Any]]:
    """
    Load all form metadata records from database.

    Args:
        jsonl_path: Deprecated - kept for backwards compatibility

    Returns:
        List of form metadata dictionaries
    """
    return load_form_records_db()


def is_already_mapped(
    form_record: Dict[str, Any],
    canonical_version_id: str
) -> bool:
    """
    Check if a form is already mapped to the given canonical version.

    A form is considered mapped if ALL its questions have a mapping entry
    for the specified canonical version.

    Args:
        form_record: Form metadata record
        canonical_version_id: Version ID to check for

    Returns:
        True if already mapped to this version, False otherwise
    """
    questions = form_record.get("form_structure", {}).get("questions", [])
    if not questions:
        return False

    for question in questions:
        mappings = question.get("mappings", [])
        has_version = any(
            m.get("canonical_version_id") == canonical_version_id
            for m in mappings
        )
        if not has_version:
            return False

    return True


# ============================================================================
# Form Questions with Canonical Overrides
# ============================================================================

@dataclass
class EnrichedQuestion:
    """A form question enriched with canonical information."""
    question_number: Optional[str]
    question_text: str
    question_type: str
    is_canonical: bool
    canonical_question_id: Optional[str] = None
    similarity_score: Optional[int] = None
    original_question_text: Optional[str] = None
    original_question_type: Optional[str] = None
    # Additional original fields
    is_required: bool = False
    char_limit: Optional[int] = None
    is_agreement: bool = False
    sub_question_of: Optional[str] = None
    mapped_pdf_fields: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_number": self.question_number,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "is_canonical": self.is_canonical,
            "canonical_question_id": self.canonical_question_id,
            "similarity_score": self.similarity_score,
            "original_question_text": self.original_question_text,
            "original_question_type": self.original_question_type,
            "is_required": self.is_required,
            "char_limit": self.char_limit,
            "is_agreement": self.is_agreement,
            "sub_question_of": self.sub_question_of,
            "mapped_pdf_fields": self.mapped_pdf_fields,
        }


@dataclass
class EnrichedForm:
    """A form with questions enriched with canonical information."""
    opportunity_id: str
    form_description: str
    questions: List[EnrichedQuestion]
    canonical_version_id: str
    min_similarity_threshold: int
    total_questions: int = 0
    canonical_questions_count: int = 0
    non_canonical_questions_count: int = 0

    def __post_init__(self):
        self.total_questions = len(self.questions)
        self.canonical_questions_count = sum(1 for q in self.questions if q.is_canonical)
        self.non_canonical_questions_count = self.total_questions - self.canonical_questions_count

    @property
    def canonical_coverage(self) -> float:
        """Percentage of questions that are canonical."""
        if self.total_questions == 0:
            return 0.0
        return self.canonical_questions_count / self.total_questions * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "form_description": self.form_description,
            "canonical_version_id": self.canonical_version_id,
            "min_similarity_threshold": self.min_similarity_threshold,
            "total_questions": self.total_questions,
            "canonical_questions_count": self.canonical_questions_count,
            "non_canonical_questions_count": self.non_canonical_questions_count,
            "canonical_coverage": f"{self.canonical_coverage:.1f}%",
            "questions": [q.to_dict() for q in self.questions],
        }


def get_enriched_forms(
    form_ids: Optional[List[str]] = None,
    canonical_version_id: Optional[str] = None,
    min_similarity: int = 3
) -> List[EnrichedForm]:
    """
    Retrieve forms with questions enriched by canonical definitions.

    Questions that map to a canonical question (at or above the similarity threshold)
    will have their text and type overridden with canonical values.

    Args:
        form_ids: List of opportunity IDs to retrieve, or None for all forms
        canonical_version_id: Canonical version to use, or None for latest
        min_similarity: Minimum similarity score (1-5) to consider a question canonical.
                       Questions with score >= min_similarity will be overridden.
                       Default is 3 (moderate match or better).

    Returns:
        List of EnrichedForm objects with canonical overrides applied
    """
    # Load canonical questions
    version_id, canonical_questions = load_canonical_questions(canonical_version_id)

    # Build lookup by canonical_question_id
    canonical_lookup = {
        cq["canonical_question_id"]: cq
        for cq in canonical_questions
        if cq.get("canonical_question_id")
    }

    # Load form records
    records = load_form_records()

    # Filter to specific forms if requested
    if form_ids:
        records = [r for r in records if r.get("opportunity_id") in form_ids]

    enriched_forms = []

    for record in records:
        opp_id = record.get("opportunity_id", "unknown")
        form_structure = record.get("form_structure", {})
        form_description = form_structure.get("form_description", "")
        questions = form_structure.get("questions", [])

        enriched_questions = []

        for q in questions:
            q_num = q.get("question_number")
            original_text = q.get("question_text") or q.get("question_text_short", "")
            original_type = q.get("question_type", "")

            # Check for mapping to this canonical version
            mappings = q.get("mappings", [])
            version_mapping = None
            for m in mappings:
                if m.get("canonical_version_id") == version_id:
                    version_mapping = m
                    break

            # Determine if canonical based on threshold
            is_canonical = False
            canonical_id = None
            similarity_score = None
            final_text = original_text
            final_type = original_type

            if version_mapping:
                similarity_score = version_mapping.get("similarity_score", 0)
                canonical_id = version_mapping.get("canonical_question_id")

                if similarity_score >= min_similarity and canonical_id:
                    is_canonical = True
                    # Override with canonical definition
                    canonical_def = canonical_lookup.get(canonical_id)
                    if canonical_def:
                        final_text = canonical_def.get("question_text", original_text)
                        final_type = canonical_def.get("question_type", original_type)

            enriched_questions.append(EnrichedQuestion(
                question_number=q_num,
                question_text=final_text,
                question_type=final_type,
                is_canonical=is_canonical,
                canonical_question_id=canonical_id if is_canonical else None,
                similarity_score=similarity_score,
                original_question_text=original_text if is_canonical else None,
                original_question_type=original_type if is_canonical else None,
                is_required=q.get("is_required", False),
                char_limit=q.get("char_limit"),
                is_agreement=q.get("is_agreement", False),
                sub_question_of=q.get("sub_question_of"),
                mapped_pdf_fields=q.get("mapped_pdf_fields"),
            ))

        enriched_forms.append(EnrichedForm(
            opportunity_id=opp_id,
            form_description=form_description,
            questions=enriched_questions,
            canonical_version_id=version_id,
            min_similarity_threshold=min_similarity,
        ))

    return enriched_forms


# ============================================================================
# Prompt Building
# ============================================================================

def _build_mapping_prompt(
    form_record: Dict[str, Any],
    canonical_questions: List[Dict[str, Any]]
) -> str:
    """
    Build the LLM prompt for mapping form questions to canonical questions.

    Args:
        form_record: Form metadata record
        canonical_questions: List of canonical question dicts

    Returns:
        Prompt string for the LLM
    """
    opp_id = form_record.get("opportunity_id", "unknown")

    # Format canonical questions
    cq_lines = []
    for cq in canonical_questions:
        cq_id = cq.get("canonical_question_id", "")
        cq_text = cq.get("question_text", "")
        cq_type = cq.get("question_type", "")
        cq_lines.append(f"- {cq_id}: {cq_text} (type: {cq_type})")

    canonical_section = "\n".join(cq_lines)

    # Format form questions
    questions = form_record.get("form_structure", {}).get("questions", [])
    fq_lines = []
    for idx, q in enumerate(questions):
        # Use question_number if available, otherwise generate index-based ID
        q_num = q.get("question_number") or f"_idx_{idx}"
        q_text = q.get("question_text", "") or q.get("question_text_short", "")
        # Truncate long texts
        if len(q_text) > 200:
            q_text = q_text[:200] + "..."
        fq_lines.append(f"- {q_num}: {q_text}")

    form_section = "\n".join(fq_lines)

    return f"""Map form questions to canonical questions with similarity scores.

## Canonical Questions
{canonical_section}

## Form Questions (opportunity_id: {opp_id})
{form_section}

## Task
For EACH form question, provide a mapping with:
- question_number: The form question number (exactly as shown above)
- canonical_question_id: The best matching canonical question ID (e.g., "cq_001") OR null if no good match
- similarity_score: 1-5 scale

## Similarity Score Guide
| Score | Meaning | Example |
|-------|---------|---------|
| 5 | Exact semantic match | "Last Name" -> "Last Name" |
| 4 | Strong match, slightly different wording | "Date of Birth" -> "Birth Date" |
| 3 | Moderate match, same general concept | "Current Address" -> "Street Address" |
| 2 | Weak/uncertain match | "Mailing Address" -> "Street Address" |
| 1 | No match, form-specific question | Unique question with no canonical equivalent |

## Rules
1. Be GENEROUS with matching - prefer false positives over false negatives
2. For scores 1-2: set canonical_question_id to null (no confident match)
3. For scores 3-5: must provide a valid canonical_question_id
4. Map EVERY form question - no question should be skipped
5. A canonical question can be mapped to multiple form questions if appropriate

Return the result as a FormMappingResult with opportunity_id="{opp_id}" and a question_mappings list."""


# ============================================================================
# Mapping Functions
# ============================================================================

def estimate_mapping_tokens(
    form_record: Dict[str, Any],
    canonical_questions: List[Dict[str, Any]],
    config: ModelTokenConfig
) -> Tuple[int, int]:
    """
    Estimate input and output tokens for a mapping operation.

    Args:
        form_record: Form metadata record
        canonical_questions: List of canonical questions
        config: Model token configuration

    Returns:
        Tuple of (estimated_input_tokens, estimated_output_tokens)
    """
    prompt = _build_mapping_prompt(form_record, canonical_questions)
    input_tokens = estimate_tokens(prompt, config.chars_per_token)

    # Output: ~20 tokens per question for the mapping
    num_questions = len(form_record.get("form_structure", {}).get("questions", []))
    output_tokens = num_questions * 20 + 50  # base overhead

    return input_tokens, output_tokens


def map_single_form(
    form_record: Dict[str, Any],
    canonical_version_id: str,
    canonical_questions: List[Dict[str, Any]],
    config: ModelTokenConfig
) -> Tuple[List[QuestionMappingOutput], FormMappingStats]:
    """
    Map one form's questions to canonical questions using the LLM.

    Args:
        form_record: Form metadata record
        canonical_version_id: Version ID of canonical questions
        canonical_questions: List of canonical question dicts
        config: Model token configuration

    Returns:
        Tuple of (List of QuestionMappingOutput, FormMappingStats)

    Raises:
        RuntimeError: If LLM call fails or returns invalid data
    """
    opp_id = form_record.get("opportunity_id", "unknown")
    logger.info(f"Mapping form {opp_id}...")

    # Get total questions in form for stats
    form_questions = form_record.get("form_structure", {}).get("questions", [])
    # Use question_number if available, otherwise use index-based ID (same as prompt)
    form_question_numbers = {
        q.get("question_number") or f"_idx_{idx}"
        for idx, q in enumerate(form_questions)
    }

    prompt = _build_mapping_prompt(form_record, canonical_questions)

    raw_response, parsed_output = get_and_parse_structured_response(
        prompt=prompt,
        output_schema=FormMappingResult,
        model=config.model_name
    )

    # Handle various output types
    if isinstance(parsed_output, str):
        if "error" in parsed_output.lower():
            raise RuntimeError(f"LLM mapping failed for {opp_id}: {parsed_output}")
        try:
            parsed_data = json.loads(parsed_output)
            result = FormMappingResult(**parsed_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse mapping response for {opp_id}: {e}")
    elif isinstance(parsed_output, FormMappingResult):
        result = parsed_output
    elif isinstance(parsed_output, dict):
        result = FormMappingResult(**parsed_output)
    else:
        raise RuntimeError(f"Unexpected output type for {opp_id}: {type(parsed_output)}")

    # Build stats from mappings
    stats = FormMappingStats(
        opportunity_id=opp_id,
        total_questions=len(form_questions)
    )

    mapped_question_numbers = set()
    for mapping in result.question_mappings:
        stats.questions_mapped += 1
        mapped_question_numbers.add(mapping.question_number)

        score = mapping.similarity_score
        if 1 <= score <= 5:
            stats.score_distribution[score] += 1

        if mapping.canonical_question_id and score >= 3:
            stats.questions_with_match += 1
            stats.canonical_ids_matched.append(mapping.canonical_question_id)
        else:
            stats.questions_no_match += 1

    # Track questions that weren't in the LLM response
    stats.unmapped_question_numbers = list(form_question_numbers - mapped_question_numbers)

    logger.info(f"Mapped {len(result.question_mappings)} questions for form {opp_id} - {stats}")
    return result.question_mappings, stats


def update_form_with_mappings(
    form_record: Dict[str, Any],
    mappings: List[QuestionMappingOutput],
    canonical_version_id: str
) -> Dict[str, Any]:
    """
    Add mappings to each question in form_structure.questions.

    Args:
        form_record: Form metadata record (will be modified in place)
        mappings: List of QuestionMappingOutput from LLM
        canonical_version_id: Version ID of the canonical questions

    Returns:
        Updated form record
    """
    # Build lookup by question_number
    mapping_lookup = {m.question_number: m for m in mappings}

    questions = form_record.get("form_structure", {}).get("questions", [])

    for idx, question in enumerate(questions):
        # Use question_number if available, otherwise use index-based ID (same as prompt)
        q_num = question.get("question_number") or f"_idx_{idx}"

        # Initialize mappings list if not present
        if "mappings" not in question:
            question["mappings"] = []

        # Remove any existing mapping for this version (in case of re-mapping)
        question["mappings"] = [
            m for m in question["mappings"]
            if m.get("canonical_version_id") != canonical_version_id
        ]

        # Add new mapping
        mapping_output = mapping_lookup.get(q_num)
        if mapping_output:
            new_mapping = QuestionMapping(
                canonical_version_id=canonical_version_id,
                canonical_question_id=mapping_output.canonical_question_id,
                similarity_score=mapping_output.similarity_score
            )
            question["mappings"].append(new_mapping.model_dump())
        else:
            # Question not in LLM output - add with score 1 (no match)
            logger.warning(f"Question {q_num} not found in LLM mapping output")
            new_mapping = QuestionMapping(
                canonical_version_id=canonical_version_id,
                canonical_question_id=None,
                similarity_score=1
            )
            question["mappings"].append(new_mapping.model_dump())

    return form_record


def save_updated_forms(
    records: List[Dict[str, Any]],
    jsonl_path: Path = JSONL_PATH
) -> None:
    """
    Save updated form records to database.

    Args:
        records: List of form records to write
        jsonl_path: Deprecated - kept for backwards compatibility
    """
    count = save_all_form_records_db(records)
    logger.info(f"Saved {count} records to database")


# ============================================================================
# Stats Functions
# ============================================================================

def print_mapping_stats(records: List[Dict[str, Any]], canonical_version_id: str) -> None:
    """Print statistics about form mappings."""
    total_forms = len(records)
    mapped_forms = 0
    total_questions = 0
    mapped_questions = 0
    score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for record in records:
        questions = record.get("form_structure", {}).get("questions", [])
        total_questions += len(questions)

        form_has_mapping = False
        for q in questions:
            mappings = q.get("mappings", [])
            for m in mappings:
                if m.get("canonical_version_id") == canonical_version_id:
                    form_has_mapping = True
                    mapped_questions += 1
                    score = m.get("similarity_score", 1)
                    if 1 <= score <= 5:
                        score_distribution[score] += 1
                    break

        if form_has_mapping:
            mapped_forms += 1

    print(f"\n=== Mapping Statistics (version: {canonical_version_id}) ===")
    print(f"Forms mapped: {mapped_forms} / {total_forms}")
    print(f"Questions mapped: {mapped_questions} / {total_questions}")
    print()
    print("Score distribution:")
    for score in range(5, 0, -1):
        count = score_distribution[score]
        pct = (count / mapped_questions * 100) if mapped_questions > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {score}: {count:4d} ({pct:5.1f}%) {bar}")

    # Count canonical question coverage
    matched_canonical = set()
    for record in records:
        for q in record.get("form_structure", {}).get("questions", []):
            for m in q.get("mappings", []):
                if m.get("canonical_version_id") == canonical_version_id:
                    cq_id = m.get("canonical_question_id")
                    if cq_id:
                        matched_canonical.add(cq_id)

    print()
    print(f"Canonical questions matched: {len(matched_canonical)}")


# ============================================================================
# Main Entry Points
# ============================================================================

def map_all_forms(
    force: bool = False,
    form_ids: Optional[List[str]] = None,
    dry_run: bool = False,
    config: ModelTokenConfig = GEMINI_2_5_FLASH,
    canonical_version: Optional[str] = None
) -> Optional[RunMappingStats]:
    """
    Main entry point: map all forms to canonical questions.

    Args:
        force: If True, re-map even if already mapped to current version
        form_ids: If provided, only map these specific forms
        dry_run: If True, show what would be mapped without calling LLM
        config: Model token configuration
        canonical_version: Specific canonical version ID to use, or None for latest

    Returns:
        RunMappingStats with aggregated results, or None for dry-run/errors
    """
    # Load canonical questions (specific version or latest)
    canonical_version_id, canonical_questions = load_canonical_questions(canonical_version)

    # Load form records
    records = load_form_records()
    if not records:
        print("No form records found.")
        return None

    # Filter to specific forms if requested
    if form_ids:
        records = [r for r in records if r.get("opportunity_id") in form_ids]
        if not records:
            print(f"No forms found matching IDs: {form_ids}")
            return None
        logger.info(f"Filtered to {len(records)} forms")

    # Determine which forms need mapping
    forms_to_map = []
    for record in records:
        opp_id = record.get("opportunity_id", "unknown")
        if force or not is_already_mapped(record, canonical_version_id):
            forms_to_map.append(record)
        else:
            logger.debug(f"Skipping {opp_id} - already mapped to version {canonical_version_id}")

    if not forms_to_map:
        print(f"All {len(records)} forms already mapped to version {canonical_version_id}")
        print("Use --force to re-map.")
        return None

    logger.info(f"{len(forms_to_map)} forms need mapping")

    # Dry run - just show estimates
    if dry_run:
        print(f"\n=== Dry Run ===")
        print(f"Forms to map: {len(forms_to_map)}")
        print(f"Canonical version: {canonical_version_id}")
        print()

        total_input = 0
        total_output = 0
        for record in forms_to_map:
            opp_id = record.get("opportunity_id", "unknown")
            num_q = len(record.get("form_structure", {}).get("questions", []))
            input_est, output_est = estimate_mapping_tokens(
                record, canonical_questions, config
            )
            total_input += input_est
            total_output += output_est
            print(f"  {opp_id}: {num_q} questions, ~{input_est:,} input / ~{output_est:,} output tokens")

        print()
        print(f"Total estimated tokens: ~{total_input:,} input / ~{total_output:,} output")
        return None

    # Process each form with stats tracking
    import time
    run_stats = RunMappingStats()

    # Create a lookup for quick updates
    record_lookup = {r.get("opportunity_id"): r for r in records}

    total_forms = len(forms_to_map)
    logger.info(f"[Mapping] Starting to map {total_forms} forms to version {canonical_version_id[:8]}...")
    mapping_start = time.time()

    for idx, form_record in enumerate(forms_to_map, 1):
        opp_id = form_record.get("opportunity_id", "unknown")
        num_questions = len(form_record.get("form_structure", {}).get("questions", []))
        try:
            form_start = time.time()
            logger.info(f"[Mapping] ({idx}/{total_forms}) Mapping form {opp_id} ({num_questions} questions)...")

            # Get mappings and stats from LLM
            mappings, form_stats = map_single_form(
                form_record,
                canonical_version_id,
                canonical_questions,
                config
            )

            # Update the record in place
            update_form_with_mappings(form_record, mappings, canonical_version_id)

            # Aggregate stats
            run_stats.add_form_stats(form_stats)

            elapsed = time.time() - form_start
            logger.info(f"[Mapping] ({idx}/{total_forms}) Form {opp_id} mapped in {elapsed:.1f}s - {form_stats.questions_with_match}/{form_stats.total_questions} matched")

        except Exception as e:
            logger.error(f"[Mapping] ({idx}/{total_forms}) Failed to map form {opp_id}: {e}")
            run_stats.add_error(opp_id, str(e))

    total_elapsed = time.time() - mapping_start
    logger.info(f"[Mapping] All forms mapped in {total_elapsed:.1f}s")

    # Save all records (including unchanged ones to preserve order)
    if run_stats.forms_successful > 0:
        save_updated_forms(records)

    # Print run summary
    run_stats.print_summary()
    print(f"\nCanonical version: {canonical_version_id}")

    return run_stats


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Map form questions to canonical questions"
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        help="Specific form opportunity IDs to map (space-separated)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-map even if already mapped to current version"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be mapped without calling LLM"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show mapping statistics and exit"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specific canonical version ID to use (default: latest)"
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List available canonical question versions and exit"
    )

    args = parser.parse_args()

    config = get_model_config(args.model)

    if args.list_versions:
        versions = list_canonical_versions()
        if not versions:
            print("No canonical question versions found.")
        else:
            print(f"\n=== Canonical Question Versions ({len(versions)} total) ===")
            for v in versions:
                latest_mark = " [LATEST]" if v["is_latest"] else ""
                print(f"  {v['version_id']}: {v['num_questions']} questions, "
                      f"created {v['created_at'][:19]}{latest_mark}")
        return

    if args.stats:
        # Just show stats
        try:
            canonical_version_id, _ = load_canonical_questions(args.version)
            records = load_form_records()
            print_mapping_stats(records, canonical_version_id)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except ValueError as e:
            print(f"Error: {e}")
            return
        return

    # Run mapping
    try:
        map_all_forms(
            force=args.force,
            form_ids=args.forms,
            dry_run=args.dry_run,
            config=config,
            canonical_version=args.version
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        logger.exception(f"Mapping failed: {e}")


if __name__ == "__main__":
    main()
