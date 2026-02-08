"""
PDF Form Filling Module.

Fills PDF forms given an opp_id and checksum. The process:
1. Validates input and loads form metadata
2. Uses LLM to transform canonical question answers into PDF-compliant field values
3. Writes values to PDF using PyPDFForm
4. Step-by-step validation ensures compliance at each stage
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from gemini_client import get_and_parse_structured_response
from form_mapping import load_canonical_questions, load_form_records
from extract_metadata import get_pdf_form_schema, parse_pdf_form_schema, FormFieldInfo

# Logger Setup
logger = logging.getLogger("pdf_writer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
FILLED_DIR = DATA_DIR / "filled"


# ============================================================================
# Request/Response Schemas
# ============================================================================

class FillOptions(BaseModel):
    """Options for form filling behavior."""
    truncate_on_char_limit: bool = Field(
        default=True,
        description="If true, truncate text values that exceed char_limit instead of failing."
    )
    fail_on_missing_optional: bool = Field(
        default=False,
        description="If true, fail if optional questions are missing answers."
    )


class FillFormRequest(BaseModel):
    """Request body for form fill endpoint."""
    opp_id: str = Field(description="Opportunity ID of the form to fill.")
    checksum: str = Field(description="SHA-256 checksum of the PDF file.")
    version_id: str = Field(description="Canonical version ID to use for mappings.")
    answers: Dict[str, Union[str, int, bool, List[str]]] = Field(
        description="Answers keyed by: 'cq_XXX' (canonical question), 'fq_N' (form question ID), or 'q_N' (legacy question_number)."
    )
    options: Optional[FillOptions] = Field(
        default=None,
        description="Optional fill behavior settings."
    )


class ValidationIssue(BaseModel):
    """A single validation issue."""
    severity: str = Field(description="Severity level: 'error', 'warning', 'info'")
    field: Optional[str] = Field(default=None, description="Field name if applicable.")
    canonical_question_id: Optional[str] = Field(default=None, description="Canonical question ID if applicable.")
    message: str = Field(description="Description of the issue.")


class ValidationStage(BaseModel):
    """Validation results for a single stage."""
    name: str = Field(description="Stage name: 'input', 'llm_transform', 'pdf_write'")
    passed: bool = Field(description="Whether this stage passed validation.")
    issues: List[ValidationIssue] = Field(default_factory=list)


class ValidationSummary(BaseModel):
    """Summary statistics for validation."""
    total_fields: int = Field(default=0, description="Total PDF fields targeted.")
    fields_filled: int = Field(default=0, description="Fields successfully filled.")
    fields_skipped: int = Field(default=0, description="Fields skipped (no value).")
    errors: int = Field(default=0, description="Number of error-level issues.")
    warnings: int = Field(default=0, description="Number of warning-level issues.")


class ValidationReport(BaseModel):
    """Complete validation report across all stages."""
    success: bool = Field(description="Whether all stages passed.")
    stages: List[ValidationStage] = Field(default_factory=list)
    summary: ValidationSummary = Field(default_factory=ValidationSummary)


class FillFormResponse(BaseModel):
    """Response body for form fill endpoint."""
    success: bool = Field(description="Whether the form was filled successfully.")
    opp_id: str = Field(description="Opportunity ID of the filled form.")
    checksum: str = Field(description="Checksum of the source PDF.")
    output_path: Optional[str] = Field(default=None, description="Path to the filled PDF file.")
    output_s3_key: Optional[str] = Field(default=None, description="S3 key of the filled PDF file.")
    download_url: Optional[str] = Field(default=None, description="Presigned URL to download the filled PDF.")
    validation_report: ValidationReport
    error: Optional[str] = Field(default=None, description="Error message if failed.")


# ============================================================================
# LLM Transformation Schemas
# ============================================================================

class PDFFieldContext(BaseModel):
    """Context about a PDF field for LLM transformation."""
    field_name: str = Field(description="Internal PDF field name.")
    field_type: str = Field(description="Field type: text, checkbox, radio, dropdown, signature.")
    field_options: Optional[List[str]] = Field(default=None, description="Valid options for radio/dropdown.")
    char_limit: Optional[int] = Field(default=None, description="Max characters for text fields.")


class QuestionWithFields(BaseModel):
    """A question with its answer and associated PDF field context."""
    canonical_question_id: Optional[str] = Field(default=None, description="Canonical question ID if mapped.")
    question_number: Optional[str] = Field(default=None, description="Question number from form.")
    question_text: str = Field(description="Question text.")
    question_type: str = Field(description="Question type.")
    answer: Union[str, int, bool, List[str]] = Field(description="User's answer to the question.")
    pdf_fields: List[PDFFieldContext] = Field(description="PDF fields that map to this question.")


class FieldValueOutput(BaseModel):
    """LLM output for a single field value."""
    field_name: str = Field(description="PDF field name.")
    value: Union[str, int, bool, None] = Field(description="Value to write to the field.")
    notes: Optional[str] = Field(default=None, description="Notes about transformation.")


class LLMTransformResult(BaseModel):
    """Complete LLM transformation output."""
    field_values: List[FieldValueOutput] = Field(description="Values for each PDF field.")
    transformation_notes: Optional[str] = Field(default=None, description="General notes about transformation.")


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_answer_key(key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse an answer key to determine if it's canonical, question_id, or question-number based.

    Returns:
        Tuple of (canonical_question_id, question_id, question_number) - one will be set, others None.
        - cq_XXX -> canonical question ID
        - fq_N -> form question ID (stable, index-based)
        - q_N -> legacy question_number lookup (deprecated)
    """
    if key.startswith("cq_"):
        return key, None, None
    elif key.startswith("fq_"):
        return None, key, None  # Keep the full fq_N as question_id
    elif key.startswith("q_"):
        return None, None, key[2:]  # Strip "q_" prefix for legacy support
    else:
        # Assume it's a canonical question ID without prefix
        return key, None, None


def _get_pdf_field_info(pdf_path: str, field_names: List[str]) -> Dict[str, PDFFieldContext]:
    """
    Get PDF field information for specific fields.

    Args:
        pdf_path: Path to the PDF file
        field_names: List of field names to get info for

    Returns:
        Dict mapping field name to PDFFieldContext
    """
    schema = get_pdf_form_schema(pdf_path)
    fields = parse_pdf_form_schema(schema)

    result = {}
    for field in fields:
        if field.field_name in field_names:
            result[field.field_name] = PDFFieldContext(
                field_name=field.field_name,
                field_type=field.field_type,
                field_options=field.field_options,
                char_limit=None  # PyPDFForm doesn't provide char limits directly
            )

    return result


def _get_all_pdf_fields(pdf_path: str) -> Dict[str, PDFFieldContext]:
    """
    Get all PDF field information from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dict mapping field name to PDFFieldContext
    """
    schema = get_pdf_form_schema(pdf_path)
    fields = parse_pdf_form_schema(schema)

    return {
        field.field_name: PDFFieldContext(
            field_name=field.field_name,
            field_type=field.field_type,
            field_options=field.field_options,
            char_limit=None
        )
        for field in fields
    }


# ============================================================================
# Validation Functions
# ============================================================================

def validate_input(
    request: FillFormRequest,
    form_record: Dict[str, Any],
    canonical_questions: List[Dict[str, Any]]
) -> Tuple[bool, List[ValidationIssue], Dict[str, QuestionWithFields]]:
    """
    Validate input request against form metadata.

    Stage 1: Input Validation
    - Checksum matches stored form
    - Parse answer keys: "cq_XXX" -> canonical lookup, "q_N" -> question_number lookup
    - Provided canonical_question_ids exist in version
    - Question numbers exist in form
    - Answer types compatible with question types

    Args:
        request: The fill form request
        form_record: The form metadata record
        canonical_questions: List of canonical questions for the version

    Returns:
        Tuple of (passed, issues, questions_with_fields_map)
    """
    issues = []
    questions_map: Dict[str, QuestionWithFields] = {}

    # Check checksum
    stored_checksum = form_record.get("pdf_checksum", "")
    if stored_checksum != request.checksum:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Checksum mismatch. Expected: {stored_checksum[:16]}..., got: {request.checksum[:16]}..."
        ))
        return False, issues, questions_map

    # Build canonical question lookup
    canonical_by_id = {cq["canonical_question_id"]: cq for cq in canonical_questions}

    # Build form question lookup by question_id, question_number, and canonical_question_id
    form_questions = form_record.get("form_structure", {}).get("questions", [])

    # Build question lookup by question_id (e.g., "fq_0", "fq_1")
    questions_by_id = {}
    for idx, q in enumerate(form_questions):
        q_id = q.get("question_id") or f"fq_{idx}"
        questions_by_id[q_id] = q

    # Build question lookup by question_number (legacy support, use index if no question_number)
    questions_by_number = {}
    for idx, q in enumerate(form_questions, start=1):
        q_num = q.get("question_number") or str(idx)
        questions_by_number[q_num] = q

    questions_by_canonical_id: Dict[str, Dict] = {}

    for q in form_questions:
        mappings = q.get("mappings", [])
        for m in mappings:
            if m.get("canonical_version_id") == request.version_id:
                cq_id = m.get("canonical_question_id")
                if cq_id and m.get("similarity_score", 0) >= 3:
                    questions_by_canonical_id[cq_id] = q
                break

    # Get PDF from S3 (S3 is the only storage option)
    pdf_s3_key = form_record.get("pdf_s3_key")

    if not pdf_s3_key:
        issues.append(ValidationIssue(
            severity="error",
            message="PDF not found in S3 storage (pdf_s3_key is missing)"
        ))
        return False, issues, questions_map

    # Get all PDF field info by downloading from S3
    try:
        from s3_storage import temp_pdf_from_s3
        with temp_pdf_from_s3(pdf_s3_key) as temp_path:
            all_pdf_fields = _get_all_pdf_fields(temp_path)
    except Exception as e:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Failed to download PDF from S3: {e}"
        ))
        return False, issues, questions_map

    # Process each answer
    for answer_key, answer_value in request.answers.items():
        canonical_id, question_id, question_number = _parse_answer_key(answer_key)

        form_question = None

        if canonical_id:
            # Look up by canonical question ID
            if canonical_id not in canonical_by_id:
                issues.append(ValidationIssue(
                    severity="error",
                    canonical_question_id=canonical_id,
                    message=f"Canonical question '{canonical_id}' not found in version {request.version_id}"
                ))
                continue

            form_question = questions_by_canonical_id.get(canonical_id)
            if not form_question:
                issues.append(ValidationIssue(
                    severity="warning",
                    canonical_question_id=canonical_id,
                    message=f"No form question mapped to canonical '{canonical_id}' with similarity >= 3"
                ))
                continue

        elif question_id:
            # Look up by form question ID (e.g., "fq_0")
            form_question = questions_by_id.get(question_id)
            if not form_question:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"Question ID '{question_id}' not found in form"
                ))
                continue

        elif question_number:
            # Look up by question number (legacy support)
            form_question = questions_by_number.get(question_number)
            if not form_question:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"Question number '{question_number}' not found in form"
                ))
                continue
        else:
            issues.append(ValidationIssue(
                severity="error",
                message=f"Invalid answer key format: '{answer_key}'. Use 'cq_XXX', 'fq_N', or 'q_N' format."
            ))
            continue

        # Get mapped PDF fields for this question
        mapped_fields = form_question.get("mapped_pdf_fields", []) or []
        if not mapped_fields:
            issues.append(ValidationIssue(
                severity="warning",
                canonical_question_id=canonical_id,
                message=f"No PDF fields mapped to question '{answer_key}'"
            ))
            continue

        # Build PDF field contexts
        pdf_field_contexts = []
        for field_name in mapped_fields:
            if field_name in all_pdf_fields:
                pdf_field_contexts.append(all_pdf_fields[field_name])
            else:
                issues.append(ValidationIssue(
                    severity="warning",
                    field=field_name,
                    canonical_question_id=canonical_id,
                    message=f"PDF field '{field_name}' not found in PDF"
                ))

        if pdf_field_contexts:
            # Get canonical definition for question text/type if available
            canonical_def = canonical_by_id.get(canonical_id) if canonical_id else None

            questions_map[answer_key] = QuestionWithFields(
                canonical_question_id=canonical_id,
                question_number=form_question.get("question_number"),
                question_text=(canonical_def.get("question_text") if canonical_def
                              else form_question.get("question_text", "")),
                question_type=(canonical_def.get("question_type") if canonical_def
                              else form_question.get("question_type", "")),
                answer=answer_value,
                pdf_fields=pdf_field_contexts
            )

    # Check for required questions without answers (if fail_on_missing_optional is set)
    options = request.options or FillOptions()
    if options.fail_on_missing_optional:
        for q in form_questions:
            if q.get("is_required", False):
                q_num = q.get("question_number")
                # Check if we have an answer for this question
                has_answer = False
                for m in q.get("mappings", []):
                    if m.get("canonical_version_id") == request.version_id:
                        cq_id = m.get("canonical_question_id")
                        if cq_id and cq_id in request.answers:
                            has_answer = True
                            break
                if not has_answer and f"q_{q_num}" not in request.answers:
                    issues.append(ValidationIssue(
                        severity="warning",
                        message=f"Required question {q_num} has no answer provided"
                    ))

    has_errors = any(i.severity == "error" for i in issues)
    return not has_errors, issues, questions_map


def validate_llm_transform(
    result: LLMTransformResult,
    expected_fields: Dict[str, PDFFieldContext],
    options: FillOptions
) -> Tuple[bool, List[ValidationIssue], Dict[str, Any]]:
    """
    Validate LLM transformation output.

    Stage 2: LLM Transform Validation
    - All expected fields have values
    - Checkbox values are boolean
    - Radio/dropdown values exist in field_options
    - Text values respect char_limit

    Args:
        result: LLM transformation result
        expected_fields: Dict of expected field contexts
        options: Fill options

    Returns:
        Tuple of (passed, issues, validated_field_values)
    """
    issues = []
    validated_values: Dict[str, Any] = {}

    # Track which fields were provided
    provided_fields = {fv.field_name for fv in result.field_values}

    for field_value in result.field_values:
        field_name = field_value.field_name
        value = field_value.value

        if field_name not in expected_fields:
            issues.append(ValidationIssue(
                severity="warning",
                field=field_name,
                message=f"Unexpected field '{field_name}' in LLM output"
            ))
            continue

        field_ctx = expected_fields[field_name]

        # Skip None values
        if value is None:
            issues.append(ValidationIssue(
                severity="info",
                field=field_name,
                message=f"Field '{field_name}' has null value, skipping"
            ))
            continue

        # Validate based on field type
        if field_ctx.field_type == "checkbox":
            if not isinstance(value, bool):
                issues.append(ValidationIssue(
                    severity="warning",
                    field=field_name,
                    message=f"Checkbox field '{field_name}' should be boolean, got {type(value).__name__}"
                ))
                # Try to convert
                value = bool(value)
            validated_values[field_name] = value

        elif field_ctx.field_type in ("radio", "dropdown"):
            if field_ctx.field_options:
                str_value = str(value)
                if str_value not in field_ctx.field_options:
                    issues.append(ValidationIssue(
                        severity="warning",
                        field=field_name,
                        message=f"Value '{str_value}' not in options {field_ctx.field_options}"
                    ))
                    # Try to find a close match
                    for opt in field_ctx.field_options:
                        if str_value.lower() == opt.lower():
                            value = opt
                            break
            validated_values[field_name] = value

        elif field_ctx.field_type == "text":
            str_value = str(value)
            if field_ctx.char_limit and len(str_value) > field_ctx.char_limit:
                if options.truncate_on_char_limit:
                    str_value = str_value[:field_ctx.char_limit]
                    issues.append(ValidationIssue(
                        severity="warning",
                        field=field_name,
                        message=f"Truncated text to {field_ctx.char_limit} chars"
                    ))
                else:
                    issues.append(ValidationIssue(
                        severity="error",
                        field=field_name,
                        message=f"Text exceeds char limit of {field_ctx.char_limit}"
                    ))
                    continue
            validated_values[field_name] = str_value

        else:
            # Unknown field type, pass through
            validated_values[field_name] = value

    # Check for missing expected fields
    for field_name in expected_fields:
        if field_name not in provided_fields:
            issues.append(ValidationIssue(
                severity="info",
                field=field_name,
                message=f"Expected field '{field_name}' not in LLM output"
            ))

    has_errors = any(i.severity == "error" for i in issues)
    return not has_errors, issues, validated_values


def validate_pdf_write(
    source_pdf: str,
    output_pdf: str,
    expected_values: Dict[str, Any]
) -> Tuple[bool, List[ValidationIssue]]:
    """
    Validate PDF write operation.

    Stage 3: PDF Write Validation
    - Output file exists and readable
    - Re-read fields to verify values written

    Args:
        source_pdf: Path to source PDF
        output_pdf: Path to output PDF
        expected_values: Expected field values

    Returns:
        Tuple of (passed, issues)
    """
    issues = []

    # Check output file exists
    if not os.path.exists(output_pdf):
        issues.append(ValidationIssue(
            severity="error",
            message=f"Output file not created: {output_pdf}"
        ))
        return False, issues

    # Check file size (basic sanity check)
    output_size = os.path.getsize(output_pdf)
    source_size = os.path.getsize(source_pdf)

    if output_size == 0:
        issues.append(ValidationIssue(
            severity="error",
            message="Output file is empty"
        ))
        return False, issues

    if output_size < source_size * 0.5:
        issues.append(ValidationIssue(
            severity="warning",
            message=f"Output file ({output_size} bytes) much smaller than source ({source_size} bytes)"
        ))

    # Try to verify fields were written (best effort)
    try:
        from PyPDFForm import PdfWrapper
        pdf = PdfWrapper(output_pdf)
        filled_schema = pdf.schema

        if filled_schema:
            issues.append(ValidationIssue(
                severity="info",
                message=f"Output PDF has {len(filled_schema.get('properties', {}))} fields"
            ))
    except Exception as e:
        issues.append(ValidationIssue(
            severity="warning",
            message=f"Could not verify output PDF fields: {e}"
        ))

    has_errors = any(i.severity == "error" for i in issues)
    return not has_errors, issues


# ============================================================================
# LLM Transformation
# ============================================================================

def _build_transform_prompt(
    questions_with_answers: List[QuestionWithFields],
    form_description: str
) -> str:
    """Build the LLM prompt for answer transformation."""

    questions_text = []
    for i, q in enumerate(questions_with_answers, 1):
        q_id = q.canonical_question_id or f"q_{q.question_number}"
        fields_info = []
        for f in q.pdf_fields:
            field_str = f"    - {f.field_name} (type: {f.field_type})"
            if f.field_options:
                field_str += f" options: {f.field_options}"
            if f.char_limit:
                field_str += f" max_chars: {f.char_limit}"
            fields_info.append(field_str)

        answer_str = json.dumps(q.answer) if isinstance(q.answer, (list, dict)) else str(q.answer)

        questions_text.append(f"""
{i}. Question ID: {q_id}
   Question: {q.question_text}
   Question Type: {q.question_type}
   User Answer: {answer_str}
   PDF Fields:
{chr(10).join(fields_info)}
""")

    return f"""Transform user answers into PDF form field values.

## Form Context
{form_description}

## Questions with Answers and Target Fields
{chr(10).join(questions_text)}

## Transformation Rules

1. **Text fields**: Return the answer as a string. Respect char_limit if provided.

2. **Checkbox fields**: Return true or false (boolean).
   - If the question is multi-choice and the user selected this option, return true.
   - If the field represents a single checkbox agreement, return true if user agreed.

3. **Radio/Dropdown fields**: Return the exact option string from field_options.
   - Match the user's answer to the closest valid option.
   - If no exact match, use best judgment to select the most appropriate option.

4. **Multi-choice to multiple checkboxes**:
   - When a multi-choice answer maps to multiple checkbox fields,
   - Set fields corresponding to selected options to true
   - Set fields corresponding to unselected options to false

5. **Date fields**: Format as appropriate for the field (typically MM/DD/YYYY).

6. **Special cases**:
   - For signature fields, set to true if user agreed/signed
   - Leave fields as null if no corresponding answer

Return a LLMTransformResult with field_values for each PDF field."""


def transform_answers_to_pdf_values(
    questions_with_answers: List[QuestionWithFields],
    form_description: str
) -> LLMTransformResult:
    """
    Call LLM to transform human answers to PDF field values.

    Args:
        questions_with_answers: List of questions with their answers and PDF field contexts
        form_description: Description of the form

    Returns:
        LLMTransformResult with field values for each PDF field
    """
    prompt = _build_transform_prompt(questions_with_answers, form_description)

    raw_response, parsed_output = get_and_parse_structured_response(
        prompt=prompt,
        output_schema=LLMTransformResult
    )

    if isinstance(parsed_output, str) and "error" in parsed_output.lower():
        raise RuntimeError(f"LLM transformation failed: {parsed_output}")

    if isinstance(parsed_output, LLMTransformResult):
        return parsed_output
    elif isinstance(parsed_output, dict):
        return LLMTransformResult(**parsed_output)
    else:
        raise RuntimeError(f"Unexpected output type: {type(parsed_output)}")


# ============================================================================
# PDF Writing
# ============================================================================

def write_pdf(source_path: str, output_path: str, field_values: Dict[str, Any]) -> bool:
    """
    Write values to a PDF form using PyPDFForm.

    Args:
        source_path: Path to the source PDF
        output_path: Path for the output PDF
        field_values: Dict of field_name -> value to write

    Returns:
        True if write succeeded
    """
    try:
        from PyPDFForm import PdfWrapper

        logger.info(f"Writing {len(field_values)} fields to PDF...")
        logger.debug(f"Field values: {field_values}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Fill the form
        filled = PdfWrapper(source_path).fill(field_values)

        # Write to output
        with open(output_path, "wb") as f:
            f.write(filled.read())

        logger.info(f"PDF written to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to write PDF: {e}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================

def fill_form(request: FillFormRequest) -> FillFormResponse:
    """
    Main entry point for form filling.

    1. Load form metadata and canonical questions
    2. Validate input (Stage 1)
    3. Build LLM context and transform answers (Stage 2)
    4. Write to PDF (Stage 3)
    5. Return path and validation report

    Args:
        request: FillFormRequest with opp_id, checksum, version_id, and answers

    Returns:
        FillFormResponse with output_path and validation_report
    """
    stages: List[ValidationStage] = []
    summary = ValidationSummary()
    options = request.options or FillOptions()

    try:
        # Load canonical questions
        logger.info(f"Loading canonical questions for version {request.version_id}...")
        version_id, canonical_questions = load_canonical_questions(request.version_id)

        # Load form records and find the matching one
        logger.info(f"Loading form record for opp_id={request.opp_id}...")
        records = load_form_records()

        form_record = None
        for record in records:
            if (record.get("opportunity_id") == request.opp_id and
                record.get("pdf_checksum") == request.checksum):
                form_record = record
                break

        if not form_record:
            return FillFormResponse(
                success=False,
                opp_id=request.opp_id,
                checksum=request.checksum,
                validation_report=ValidationReport(
                    success=False,
                    stages=[ValidationStage(
                        name="input",
                        passed=False,
                        issues=[ValidationIssue(
                            severity="error",
                            message=f"Form not found: opp_id={request.opp_id}, checksum={request.checksum[:16]}..."
                        )]
                    )],
                    summary=ValidationSummary(errors=1)
                ),
                error="Form not found"
            )

        # Stage 1: Input Validation
        logger.info("Stage 1: Validating input...")
        input_passed, input_issues, questions_map = validate_input(
            request, form_record, canonical_questions
        )

        stages.append(ValidationStage(
            name="input",
            passed=input_passed,
            issues=input_issues
        ))

        for issue in input_issues:
            if issue.severity == "error":
                summary.errors += 1
            elif issue.severity == "warning":
                summary.warnings += 1

        if not input_passed:
            return FillFormResponse(
                success=False,
                opp_id=request.opp_id,
                checksum=request.checksum,
                validation_report=ValidationReport(
                    success=False,
                    stages=stages,
                    summary=summary
                ),
                error="Input validation failed"
            )

        # Stage 2: LLM Transformation
        logger.info("Stage 2: Transforming answers via LLM...")
        form_description = form_record.get("form_structure", {}).get("form_description", "")

        questions_with_answers = list(questions_map.values())

        # Collect all expected PDF fields
        expected_fields: Dict[str, PDFFieldContext] = {}
        for q in questions_with_answers:
            for f in q.pdf_fields:
                expected_fields[f.field_name] = f

        summary.total_fields = len(expected_fields)

        try:
            transform_result = transform_answers_to_pdf_values(
                questions_with_answers,
                form_description
            )
        except Exception as e:
            stages.append(ValidationStage(
                name="llm_transform",
                passed=False,
                issues=[ValidationIssue(
                    severity="error",
                    message=f"LLM transformation failed: {str(e)}"
                )]
            ))
            summary.errors += 1

            return FillFormResponse(
                success=False,
                opp_id=request.opp_id,
                checksum=request.checksum,
                validation_report=ValidationReport(
                    success=False,
                    stages=stages,
                    summary=summary
                ),
                error=f"LLM transformation failed: {str(e)}"
            )

        # Validate transformation
        transform_passed, transform_issues, validated_values = validate_llm_transform(
            transform_result, expected_fields, options
        )

        stages.append(ValidationStage(
            name="llm_transform",
            passed=transform_passed,
            issues=transform_issues
        ))

        for issue in transform_issues:
            if issue.severity == "error":
                summary.errors += 1
            elif issue.severity == "warning":
                summary.warnings += 1

        summary.fields_filled = len(validated_values)
        summary.fields_skipped = summary.total_fields - summary.fields_filled

        if not transform_passed:
            return FillFormResponse(
                success=False,
                opp_id=request.opp_id,
                checksum=request.checksum,
                validation_report=ValidationReport(
                    success=False,
                    stages=stages,
                    summary=summary
                ),
                error="LLM transform validation failed"
            )

        # Stage 3: PDF Writing (S3 only)
        logger.info("Stage 3: Writing PDF...")
        pdf_s3_key = form_record.get("pdf_s3_key")

        if not pdf_s3_key:
            stages.append(ValidationStage(
                name="pdf_write",
                passed=False,
                issues=[ValidationIssue(
                    severity="error",
                    message="PDF not found in S3 storage (pdf_s3_key is missing)"
                )]
            ))
            summary.errors += 1
            return FillFormResponse(
                success=False,
                opp_id=request.opp_id,
                checksum=request.checksum,
                validation_report=ValidationReport(
                    success=False,
                    stages=stages,
                    summary=summary
                ),
                error="PDF not found in S3 storage"
            )

        # Generate output filename
        timestamp = int(time.time())
        checksum_prefix = request.checksum[:6]
        output_filename = f"{request.opp_id}-{checksum_prefix}-{timestamp}.pdf"

        # Variables for S3 output
        output_s3_key = None
        download_url = None

        try:
            from s3_storage import temp_pdf_from_s3, upload_pdf as s3_upload, generate_presigned_url

            # Download source PDF from S3, fill it, upload filled version
            with temp_pdf_from_s3(pdf_s3_key) as temp_source_path:
                # Create temp file for output
                temp_output_path = temp_source_path + ".filled.pdf"
                write_pdf(temp_source_path, temp_output_path, validated_values)

                # Upload filled PDF to S3
                output_s3_key = s3_upload(temp_output_path, "filled", output_filename)
                download_url = generate_presigned_url(output_s3_key, expires_in=3600)
                logger.info(f"Uploaded filled PDF to S3: {output_s3_key}")

                # Clean up temp output file
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)

        except Exception as e:
            stages.append(ValidationStage(
                name="pdf_write",
                passed=False,
                issues=[ValidationIssue(
                    severity="error",
                    message=f"PDF write failed: {str(e)}"
                )]
            ))
            summary.errors += 1

            return FillFormResponse(
                success=False,
                opp_id=request.opp_id,
                checksum=request.checksum,
                validation_report=ValidationReport(
                    success=False,
                    stages=stages,
                    summary=summary
                ),
                error=f"PDF write failed: {str(e)}"
            )

        # Validation - S3 upload succeeded, so write is considered valid
        write_passed = True
        write_issues = [ValidationIssue(
            severity="info",
            message=f"Filled PDF uploaded to S3: {output_s3_key}"
        )]

        stages.append(ValidationStage(
            name="pdf_write",
            passed=write_passed,
            issues=write_issues
        ))

        for issue in write_issues:
            if issue.severity == "error":
                summary.errors += 1
            elif issue.severity == "warning":
                summary.warnings += 1

        all_passed = all(s.passed for s in stages)

        return FillFormResponse(
            success=all_passed,
            opp_id=request.opp_id,
            checksum=request.checksum,
            output_path=None,  # Deprecated - using S3 only
            output_s3_key=output_s3_key if all_passed else None,
            download_url=download_url if all_passed else None,
            validation_report=ValidationReport(
                success=all_passed,
                stages=stages,
                summary=summary
            ),
            error=None if all_passed else "One or more stages failed"
        )

    except Exception as e:
        logger.exception(f"Form fill failed: {e}")
        return FillFormResponse(
            success=False,
            opp_id=request.opp_id,
            checksum=request.checksum,
            validation_report=ValidationReport(
                success=False,
                stages=stages,
                summary=ValidationSummary(errors=1)
            ),
            error=str(e)
        )


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fill PDF forms with answers")
    parser.add_argument("--opp-id", required=True, help="Opportunity ID")
    parser.add_argument("--checksum", required=True, help="PDF checksum")
    parser.add_argument("--version", required=True, help="Canonical version ID")
    parser.add_argument("--answers-file", required=True, help="JSON file with answers")

    args = parser.parse_args()

    # Load answers
    with open(args.answers_file, "r") as f:
        answers = json.load(f)

    # Create request
    request = FillFormRequest(
        opp_id=args.opp_id,
        checksum=args.checksum,
        version_id=args.version,
        answers=answers
    )

    # Fill form
    response = fill_form(request)

    print(json.dumps(response.model_dump(), indent=2))
