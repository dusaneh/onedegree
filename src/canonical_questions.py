"""
Canonical Question Identification System.

Identifies semantically equivalent questions across multiple forms using Gemini Pro
for semantic analysis, with graceful token limit handling.
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from gemini_client import get_and_parse_structured_response
from database import (
    load_form_records_db,
    list_canonical_versions_db,
    get_canonical_version_db,
    save_canonical_version_db,
    set_latest_canonical_version_db,
)

# Logger Setup
logger = logging.getLogger("canonical_questions")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
JSONL_PATH = DATA_DIR / "form_metadata.jsonl"
CANONICAL_MAPPINGS_PATH = DATA_DIR / "canonical_mappings.json"


# ============================================================================
# Custom Exceptions
# ============================================================================

class TokenLimitError(Exception):
    """Raised BEFORE API call if limits would be exceeded."""

    def __init__(
        self,
        message: str,
        estimated_tokens: int = 0,
        limit: int = 0,
        recommended_batch_size: Optional[int] = None
    ):
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        self.recommended_batch_size = recommended_batch_size


class GeminiAPIError(Exception):
    """Raised when API call fails."""

    def __init__(self, message: str, raw_response: Any = None):
        super().__init__(message)
        self.raw_response = raw_response


# ============================================================================
# Input Schemas (Simplified for LLM)
# ============================================================================

class AnswerType(str, Enum):
    TEXT = "text"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    DATE = "date"
    NUMERIC = "numeric"
    YES_NO = "yes_no"
    AGREEMENT = "agreement"


class SimplifiedQuestion(BaseModel):
    """Simplified question for LLM processing."""
    form_id: str = Field(description="Opportunity ID")
    question_id: str = Field(description="Unique ID: {form_id}_{question_number}")
    text: str = Field(description="Question text (truncated to 300 chars)")
    answer_type: AnswerType
    options: Optional[List[str]] = Field(default=None, description="From PDF field options")


class SimplifiedForm(BaseModel):
    """Simplified form for LLM processing."""
    form_id: str
    description: str = Field(description="Form description (truncated to 200 chars)")
    questions: List[SimplifiedQuestion]


# ============================================================================
# Output Schemas
# ============================================================================

class CanonicalQuestionDetails(BaseModel):
    """Minimal fields for canonical questions (token-efficient).

    Matches QuestionDetails schema to minimize output tokens.
    Only questions appearing on 2+ forms are included.
    """
    canonical_question_id: Optional[str] = Field(default=None, description="Unique ID assigned post-LLM: 'cq_001', 'cq_002', etc.")
    question_text: str = Field(description="Representative text for this canonical question")
    question_type: str = Field(description="'open ended', 'multiple choice (choose one)', 'date', 'numeric', 'agreement/authorization'")
    char_limit: Optional[int] = None
    is_agreement: bool = False
    sub_question_of: Optional[str] = None
    unique_structure: bool = False


class CanonicalMappingResult(BaseModel):
    """Complete result of canonical question identification."""
    version_id: str
    created_at: str
    model_used: str
    input_token_count: int
    output_token_count: int
    canonical_questions: List[CanonicalQuestionDetails]  # ONLY questions on 2+ forms
    processing_notes: Optional[str] = None


# ============================================================================
# Token Validation Schemas
# ============================================================================

class TokenEstimate(BaseModel):
    """Token estimation result."""
    estimated_input_tokens: int
    estimated_output_tokens: int
    exceeds_input_limit: bool
    exceeds_output_limit: bool
    forms_processed: int = 0
    total_forms: int = 0


class ModelTokenConfig(BaseModel):
    """Token limits and estimation params per model."""
    model_name: str
    max_total_tokens: int      # Total context window
    max_input_tokens: int      # Max input allowed
    max_output_tokens: int     # Max output allowed
    chars_per_token: float = 4.0

    # Output estimation formula params
    tokens_per_question_out: int = 50      # tpqo
    base_questions_out: int = 3            # bqo
    additional_questions_per_form: float = 1.5  # aqpfa


# Model Presets
GEMINI_2_5_FLASH = ModelTokenConfig(
    model_name="gemini-2.5-flash",
    max_total_tokens=1_000_000,
    max_input_tokens=700_000,     # Conservative limit
    max_output_tokens=60_000,     # Conservative limit
    tokens_per_question_out=50,
    base_questions_out=3,
    additional_questions_per_form=1.5
)

GEMINI_2_5_PRO = ModelTokenConfig(
    model_name="gemini-2.5-pro",
    max_total_tokens=1_000_000,
    max_input_tokens=700_000,     # Same limits
    max_output_tokens=60_000,
    tokens_per_question_out=50,
    base_questions_out=3,
    additional_questions_per_form=1.5
)

GEMINI_3_FLASH = ModelTokenConfig(
    model_name="gemini-3-flash-preview",
    max_total_tokens=1_000_000,
    max_input_tokens=700_000,     # Conservative limit
    max_output_tokens=65_535,     # Maximum output for Gemini 3
    tokens_per_question_out=50,
    base_questions_out=3,
    additional_questions_per_form=1.5
)


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_answer_type(question_type: str, has_options: bool) -> AnswerType:
    """
    Map raw question types to AnswerType enum.

    Args:
        question_type: Raw question type from form metadata
        has_options: Whether the question has predefined options

    Returns:
        Normalized AnswerType
    """
    q_type_lower = question_type.lower()

    if "agreement" in q_type_lower or "authorization" in q_type_lower:
        return AnswerType.AGREEMENT

    if "date" in q_type_lower:
        return AnswerType.DATE

    if "numeric" in q_type_lower or "number" in q_type_lower:
        return AnswerType.NUMERIC

    if "yes" in q_type_lower and "no" in q_type_lower:
        return AnswerType.YES_NO

    if "multi" in q_type_lower and "choice" in q_type_lower:
        if "all" in q_type_lower or "choose all" in q_type_lower:
            return AnswerType.MULTI_SELECT
        return AnswerType.SINGLE_SELECT

    if "multiple choice" in q_type_lower or "choose one" in q_type_lower:
        return AnswerType.SINGLE_SELECT

    if has_options:
        return AnswerType.SINGLE_SELECT

    return AnswerType.TEXT


def extract_question_options(
    question: Dict[str, Any],
    pdf_fields: Optional[List[Dict[str, Any]]]
) -> Optional[List[str]]:
    """
    Get options from mapped PDF fields.

    Args:
        question: Question dict from form metadata
        pdf_fields: List of PDF field info dicts

    Returns:
        List of options if available, None otherwise
    """
    if not pdf_fields:
        return None

    mapped_field_names = question.get("mapped_pdf_fields") or []
    if not mapped_field_names:
        return None

    # Build lookup of PDF fields
    field_lookup = {f["field_name"]: f for f in pdf_fields}

    # Collect options from mapped fields
    all_options = []
    for field_name in mapped_field_names:
        field_info = field_lookup.get(field_name)
        if field_info and field_info.get("field_options"):
            all_options.extend(field_info["field_options"])

    return all_options if all_options else None


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count from text.

    Args:
        text: Text to estimate
        chars_per_token: Characters per token (default 4.0 conservative)

    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)


def compute_version_id(forms: List[SimplifiedForm]) -> str:
    """
    Compute a version ID from simplified forms content.

    Args:
        forms: List of SimplifiedForm objects

    Returns:
        First 12 characters of SHA-256 hash
    """
    content = json.dumps(
        [f.model_dump() for f in forms],
        sort_keys=True,
        ensure_ascii=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_form_metadata(jsonl_path: Path = JSONL_PATH) -> List[Dict[str, Any]]:
    """
    Load all form metadata records from database.

    Args:
        jsonl_path: Deprecated - kept for backwards compatibility

    Returns:
        List of form metadata dictionaries
    """
    records = load_form_records_db()
    logger.info(f"Loaded {len(records)} form metadata records from database")
    return records


def simplify_form(record: Dict[str, Any]) -> SimplifiedForm:
    """
    Convert a form metadata record to a SimplifiedForm for LLM processing.

    Args:
        record: Form metadata record dict

    Returns:
        SimplifiedForm object
    """
    form_id = str(record["opportunity_id"])
    form_structure = record["form_structure"]
    description = form_structure.get("form_description", "")[:200]

    pdf_fields = form_structure.get("pdf_form_fields") or []

    questions = []
    for q in form_structure.get("questions", []):
        q_num = q.get("question_number") or str(len(questions) + 1)
        question_id = f"{form_id}_{q_num}"

        # Get full question text, truncated
        text = q.get("question_text") or q.get("question_text_short", "")
        text = text[:300]

        # Normalize answer type
        q_type = q.get("question_type", "open ended")
        options = extract_question_options(q, pdf_fields)
        answer_type = normalize_answer_type(q_type, options is not None)

        # Override for agreement fields
        if q.get("is_agreement"):
            answer_type = AnswerType.AGREEMENT

        questions.append(SimplifiedQuestion(
            form_id=form_id,
            question_id=question_id,
            text=text,
            answer_type=answer_type,
            options=options[:10] if options else None  # Limit options
        ))

    return SimplifiedForm(
        form_id=form_id,
        description=description,
        questions=questions
    )


def load_and_simplify_all_forms(
    jsonl_path: Path = JSONL_PATH,
    form_ids: Optional[List[str]] = None
) -> List[SimplifiedForm]:
    """
    Load JSONL and convert to SimplifiedForm list.

    Args:
        jsonl_path: Path to the form metadata JSONL file
        form_ids: If provided, only include these opportunity IDs

    Returns:
        List of SimplifiedForm objects
    """
    records = load_form_metadata(jsonl_path)

    # Filter to specific forms if requested
    if form_ids:
        records = [r for r in records if r.get("opportunity_id") in form_ids]
        logger.info(f"Filtered to {len(records)} forms (requested: {form_ids})")

    forms = [simplify_form(r) for r in records]

    total_questions = sum(len(f.questions) for f in forms)
    logger.info(f"Simplified {len(forms)} forms with {total_questions} total questions")

    return forms


# ============================================================================
# Token Validation Functions
# ============================================================================

def calculate_tokens_progressive(
    forms: List[SimplifiedForm],
    config: ModelTokenConfig
) -> Tuple[List[SimplifiedForm], TokenEstimate]:
    """
    Calculate tokens progressively. STOP and return subset if limit exceeded.

    Args:
        forms: List of SimplifiedForm objects
        config: Model token configuration

    Returns:
        Tuple of (forms_to_process, token_estimate)
        - If limit exceeded, returns only forms that fit under limit
    """
    cumulative_input = 0
    forms_to_process = []

    for form in forms:
        form_json = json.dumps(form.model_dump(), ensure_ascii=False)
        form_tokens = len(form_json) / config.chars_per_token

        # Check if adding this form would exceed limit
        if cumulative_input + form_tokens > config.max_input_tokens:
            logger.warning(f"Token limit reached at {len(forms_to_process)} forms. Stopping.")
            break

        cumulative_input += form_tokens
        forms_to_process.append(form)

    num_forms = len(forms_to_process)
    # Output formula: (bqo + num_forms * aqpfa) * tpqo
    estimated_canonical = config.base_questions_out + (num_forms * config.additional_questions_per_form)
    estimated_output = int(estimated_canonical * config.tokens_per_question_out)

    return forms_to_process, TokenEstimate(
        estimated_input_tokens=int(cumulative_input),
        estimated_output_tokens=estimated_output,
        exceeds_input_limit=False,  # Already truncated
        exceeds_output_limit=estimated_output > config.max_output_tokens,
        forms_processed=len(forms_to_process),
        total_forms=len(forms)
    )


# ============================================================================
# Canonical Question Identification
# ============================================================================

def _build_canonical_prompt(forms: List[SimplifiedForm]) -> str:
    """Build the prompt for canonical question identification."""

    forms_data = json.dumps(
        [f.model_dump() for f in forms],
        indent=2,
        ensure_ascii=False
    )

    return f"""Analyze the following form questions and identify semantically equivalent "canonical" questions.

## Input Forms Data
```json
{forms_data}
```

## Task
Identify questions that appear SEMANTICALLY on **2 or more forms**.

IMPORTANT:
- Only output questions that appear on AT LEAST 2 different forms
- Do NOT include questions that only appear on a single form
- Minimal output format - only 6 fields per question

For each canonical question, provide ONLY:
- question_text: Representative text (concise, max 50 chars)
- question_type: One of 'open ended', 'multiple choice (choose one)', 'date', 'numeric', 'agreement/authorization'
- char_limit: Integer or null
- is_agreement: Boolean (true for consent/authorization questions)
- sub_question_of: String or null (if this is a sub-question)
- unique_structure: Boolean (true if question has unusual format)

## Guidelines
- Focus on SEMANTIC equivalence, not exact text matching
- "What is your name?" and "Applicant Name" are semantically equivalent
- Questions about the same field for different entities are DIFFERENT (applicant_name vs emergency_contact_name)
- Only include questions that appear on 2+ forms
- Keep question_text CONCISE (max 50 chars)

## Output Requirements
- Be concise - minimize text length in all string fields
- Skip version_id, created_at, model_used, input_token_count, output_token_count (will be filled automatically)
- Return canonical_questions as a list of objects with the 6 fields above

Return the result according to the CanonicalMappingResult schema."""


def identify_canonical_questions(
    forms: List[SimplifiedForm],
    config: ModelTokenConfig = GEMINI_2_5_FLASH
) -> CanonicalMappingResult:
    """
    Main entry point - identify canonical questions using Gemini.

    Args:
        forms: List of SimplifiedForm objects
        config: Model token configuration

    Returns:
        CanonicalMappingResult with identified canonical questions
    """
    import time

    total_questions = sum(len(f.questions) for f in forms)
    logger.info(f"[Canonical] Starting identification for {len(forms)} forms, {total_questions} questions")

    # Calculate tokens progressively, truncating if needed
    step_start = time.time()
    forms_to_process, token_estimate = calculate_tokens_progressive(forms, config)
    logger.info(f"[Canonical] Token calculation took {time.time() - step_start:.1f}s")

    if len(forms_to_process) < len(forms):
        logger.warning(
            f"[Canonical] Processing {len(forms_to_process)}/{len(forms)} forms due to token limit"
        )

    logger.info(
        f"[Canonical] Token estimate - Input: {token_estimate.estimated_input_tokens:,}, "
        f"Output: {token_estimate.estimated_output_tokens:,}"
    )

    # Build prompt
    step_start = time.time()
    prompt = _build_canonical_prompt(forms_to_process)
    logger.info(f"[Canonical] Prompt built in {time.time() - step_start:.1f}s (length: {len(prompt):,} chars)")

    # Call Gemini with maximum output token limit for large JSON responses
    step_start = time.time()
    logger.info(f"[Canonical] Calling {config.model_name}... (this may take 1-5 minutes)")
    logger.info(f"[Canonical] Estimated output tokens: {token_estimate.estimated_output_tokens:,}")
    raw_response, parsed_output = get_and_parse_structured_response(
        prompt=prompt,
        output_schema=CanonicalMappingResult,
        model=config.model_name,
        max_output_tokens=65535  # Maximum limit to prevent truncation
    )
    logger.info(f"[Canonical] LLM response received in {time.time() - step_start:.1f}s")

    # Handle errors
    if isinstance(parsed_output, str):
        if "error" in parsed_output.lower():
            raise GeminiAPIError(f"API call failed: {parsed_output}", raw_response)
        # Try manual parsing
        try:
            parsed_data = json.loads(parsed_output)
            result = CanonicalMappingResult(**parsed_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise GeminiAPIError(f"Failed to parse response: {e}", raw_response)
    elif isinstance(parsed_output, CanonicalMappingResult):
        result = parsed_output
    elif isinstance(parsed_output, dict):
        result = CanonicalMappingResult(**parsed_output)
    else:
        raise GeminiAPIError(f"Unexpected output type: {type(parsed_output)}", raw_response)

    # Fill in metadata if not provided by LLM
    version_id = compute_version_id(forms_to_process)
    result.version_id = version_id
    result.created_at = datetime.now().isoformat()
    result.model_used = config.model_name

    # Get token counts from raw response if available
    if raw_response and hasattr(raw_response, "usage_metadata"):
        usage = raw_response.usage_metadata
        result.input_token_count = getattr(usage, "prompt_token_count", token_estimate.estimated_input_tokens)
        result.output_token_count = getattr(usage, "candidates_token_count", token_estimate.estimated_output_tokens)
    else:
        result.input_token_count = token_estimate.estimated_input_tokens
        result.output_token_count = token_estimate.estimated_output_tokens

    total_questions = sum(len(f.questions) for f in forms_to_process)
    result.processing_notes = f"{len(forms_to_process)}/{len(forms)} forms, {total_questions} questions"

    # Assign canonical_question_id to each question post-LLM
    for idx, cq in enumerate(result.canonical_questions):
        cq.canonical_question_id = f"cq_{idx + 1:03d}"

    logger.info(
        f"Identified {len(result.canonical_questions)} canonical questions (2+ forms)"
    )

    return result


# ============================================================================
# Caching Functions (Database-backed)
# ============================================================================

def list_canonical_versions(
    cache_path: Path = CANONICAL_MAPPINGS_PATH
) -> List[Dict[str, Any]]:
    """
    List all available canonical question versions.

    Args:
        cache_path: Deprecated - kept for backwards compatibility

    Returns:
        List of dicts with version info (version_id, created_at, model_used, num_questions)
    """
    db_versions = list_canonical_versions_db()
    versions = []

    for v in db_versions:
        versions.append({
            "version_id": v.get("version_id"),
            "created_at": v.get("created_at"),
            "model_used": v.get("model_used"),
            "num_questions": len(v.get("canonical_questions", [])),
            "is_latest": v.get("is_latest", False)
        })

    return versions


def load_cached_result(
    version_id: Optional[str] = None,
    cache_path: Path = CANONICAL_MAPPINGS_PATH
) -> Optional[CanonicalMappingResult]:
    """
    Load cached canonical mappings by version ID from database.

    Args:
        version_id: Version ID to load, or None for latest
        cache_path: Deprecated - kept for backwards compatibility

    Returns:
        CanonicalMappingResult if found, None otherwise
    """
    db_versions = list_canonical_versions_db()

    if not db_versions:
        logger.info("No cached canonical mappings found in database")
        return None

    # If no version specified, find the latest
    if version_id is None:
        for v in db_versions:
            if v.get("is_latest"):
                version_id = v.get("version_id")
                break
        # Fallback to most recent if no latest flag
        if not version_id and db_versions:
            version_id = db_versions[0].get("version_id")

    # Get the specific version
    version_data = get_canonical_version_db(version_id)
    if version_data:
        logger.info(f"Loaded canonical version from database: {version_id}")
        return CanonicalMappingResult(**version_data)

    logger.info(f"Version {version_id} not found in database")
    return None


def save_cached_result(
    result: CanonicalMappingResult,
    cache_path: Path = CANONICAL_MAPPINGS_PATH,
    set_as_latest: bool = True
) -> None:
    """
    Save canonical mappings to database.

    Args:
        result: CanonicalMappingResult to save
        cache_path: Deprecated - kept for backwards compatibility
        set_as_latest: If True, mark this version as the latest
    """
    result_dict = result.model_dump()
    save_canonical_version_db(result_dict)

    if set_as_latest:
        set_latest_canonical_version_db(result.version_id)
        logger.info(f"Set {result.version_id} as latest canonical version")

    logger.info(f"Saved canonical version to database: {result.version_id}")


def get_model_config(model_name: str) -> ModelTokenConfig:
    """
    Get model configuration by name.

    Args:
        model_name: Model name string

    Returns:
        ModelTokenConfig for the specified model
    """
    if "gemini-3" in model_name.lower():
        return ModelTokenConfig(**{**GEMINI_3_FLASH.model_dump(), "model_name": model_name})
    if "pro" in model_name.lower():
        return ModelTokenConfig(**{**GEMINI_2_5_PRO.model_dump(), "model_name": model_name})
    # Default to 2.5 flash config
    return ModelTokenConfig(**{**GEMINI_2_5_FLASH.model_dump(), "model_name": model_name})


def get_canonical_questions(
    config: ModelTokenConfig = GEMINI_2_5_FLASH,
    force_refresh: bool = False,
    version_id: Optional[str] = None,
    form_ids: Optional[List[str]] = None
) -> Tuple[CanonicalMappingResult, List[SimplifiedForm]]:
    """
    Main entry point - get canonical questions with caching.

    Args:
        config: Model token configuration
        force_refresh: If True, create new version even if current forms version exists
        version_id: If provided, load this specific version (ignores force_refresh)
        form_ids: If provided, only use these forms for canonical question identification

    Returns:
        Tuple of (CanonicalMappingResult, List[SimplifiedForm] for stats)
    """
    # Load and simplify forms (optionally filtered)
    forms = load_and_simplify_all_forms(form_ids=form_ids)
    if not forms:
        raise ValueError("No forms found in form metadata")

    # If specific version requested, just load it
    if version_id is not None:
        cached = load_cached_result(version_id)
        if cached:
            return cached, forms
        raise ValueError(f"Canonical version {version_id} not found")

    # Compute version ID for cache check
    current_version_id = compute_version_id(forms)
    logger.info(f"Forms version ID: {current_version_id}")

    # Check cache unless force refresh
    if not force_refresh:
        cached = load_cached_result(current_version_id)
        if cached:
            return cached, forms

    # Process forms (progressive truncation if needed)
    result = identify_canonical_questions(forms, config)

    # Save to cache (appends to versions list)
    save_cached_result(result)

    return result, forms


def print_stats(result: CanonicalMappingResult, forms: List[SimplifiedForm]) -> None:
    """Print processing statistics."""
    total_questions = sum(len(f.questions) for f in forms)
    avg_questions = total_questions / len(forms) if forms else 0

    # Parse forms processed from processing_notes
    forms_processed = len(forms)
    if result.processing_notes and "/" in result.processing_notes:
        try:
            parts = result.processing_notes.split("/")
            forms_processed = int(parts[0])
        except (ValueError, IndexError):
            pass

    print(f"\n=== Canonical Question Stats ===")
    print(f"Forms processed: {forms_processed} / {len(forms)}")
    print(f"Total questions ingested: {total_questions}")
    print(f"Avg questions per form: {avg_questions:.1f}")
    print()
    print(f"Input tokens used: {result.input_token_count:,}")
    print(f"Output tokens used: {result.output_token_count:,}")
    print()
    print(f"Canonical questions output: {len(result.canonical_questions)}")
    print(f"(Only questions appearing on 2+ forms)")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify canonical questions across forms"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force create new version even if current forms version exists"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate token limits, don't call API"
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List all available canonical question versions"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Load a specific version by ID"
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        help="Only use these form opportunity IDs (space-separated)"
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
                      f"created {v['created_at'][:19]}, model={v['model_used']}{latest_mark}")
    elif args.validate_only:
        forms = load_and_simplify_all_forms(form_ids=args.forms)
        forms_to_process, estimate = calculate_tokens_progressive(forms, config)
        print(f"\n=== Token Estimate ===")
        print(f"Forms that fit: {len(forms_to_process)} / {len(forms)}")
        print(f"Input tokens: {estimate.estimated_input_tokens:,}")
        print(f"Output tokens (estimated): {estimate.estimated_output_tokens:,}")
        print(f"Exceeds output limit: {estimate.exceeds_output_limit}")
        if len(forms_to_process) == len(forms):
            print("\nAll forms fit within token limits")
        else:
            print(f"\nWARNING: {len(forms) - len(forms_to_process)} forms will be truncated")
    else:
        try:
            result, forms = get_canonical_questions(
                config=config,
                force_refresh=args.force,
                version_id=args.version,
                form_ids=args.forms
            )

            print(f"\n=== Canonical Question Identification Complete ===")
            print(f"Version ID: {result.version_id}")
            print(f"Model: {result.model_used}")
            print(f"\nResults saved to: {CANONICAL_MAPPINGS_PATH}")

            # Print stats
            print_stats(result, forms)

            # Print sample canonical questions
            if result.canonical_questions:
                print(f"\n=== Sample Canonical Questions ===")
                for cq in result.canonical_questions[:5]:
                    cq_id = cq.canonical_question_id or "?"
                    agreement_mark = " [AGREEMENT]" if cq.is_agreement else ""
                    print(f"  - {cq_id}: {cq.question_text} ({cq.question_type}){agreement_mark}")
                if len(result.canonical_questions) > 5:
                    print(f"  ... and {len(result.canonical_questions) - 5} more")

        except GeminiAPIError as e:
            print(f"\nError: API call failed")
            print(f"  {e}")
        except ValueError as e:
            print(f"\nError: {e}")
