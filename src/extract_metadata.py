"""
PDF metadata extraction with caching, checksum validation, and form field mapping.
Self-contained module - all imports local to src/.
"""

import hashlib
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

from gemini_client import get_and_parse_structured_response
from database import (
    load_form_records_db,
    save_form_record_db,
    get_form_by_opp_and_checksum,
)

# Logger Setup
logger = logging.getLogger("extract_metadata")
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
PDF_DIR = DATA_DIR / "pdfs"


# ============================================================================
# Pydantic Schemas
# ============================================================================

class FormFieldInfo(BaseModel):
    """Information about a PDF form field extracted via PyPDFForm."""
    field_name: str = Field(description="The internal field name from the PDF.")
    field_type: str = Field(description="Type of field: text, checkbox, radio, dropdown, signature, etc.")
    field_value: Optional[str] = Field(default=None, description="Current value if any.")
    field_options: Optional[List[str]] = Field(default=None, description="Available options for dropdowns/radios.")


class FieldMapping(BaseModel):
    """Mapping between a PDF form field and a question in the schema."""
    pdf_field_name: str = Field(description="The internal PDF field name from the extractor.")
    pdf_field_type: str = Field(description="The field type from PDF extractor: 'text', 'checkbox', 'radio', 'dropdown', 'signature'.")
    question_number: Optional[str] = Field(default=None, description="The question number this field maps to.")
    question_text_short: Optional[str] = Field(default=None, description="Short text of the mapped question.")
    confidence: str = Field(description="Confidence level: 'high', 'medium', 'low'.")
    mapping_notes: Optional[str] = Field(default=None, description="Notes about the mapping or why it couldn't be mapped.")


class QuestionMapping(BaseModel):
    """Single mapping entry for a question to a canonical question."""
    canonical_version_id: str = Field(description="Version ID of the canonical questions used (e.g., 'c3f1d1c4110f').")
    canonical_question_id: Optional[str] = Field(default=None, description="ID of matched canonical question (e.g., 'cq_001') or null if no match.")
    similarity_score: int = Field(ge=1, le=5, description="Similarity score 1-5 (5=exact match, 1=no match).")


class QuestionDetails(BaseModel):
    """Details for a single question or field in the form."""
    question_id: Optional[str] = Field(default=None, description="Stable form-level question ID (e.g., 'fq_0', 'fq_1'). Generated during extraction, used for answer submission.")
    is_required: bool = Field(description="True if the field is mandatory, False if optional.")
    question_number: Optional[str] = Field(default=None, description="The visible number or identifier (e.g., '1.', 'A', '3.b').")
    question_text_short: str = Field(description="A concise version of the question text.")
    question_text: str = Field(description="The full question text, including context/options.")
    question_type: str = Field(description="Type: 'open ended', 'multiple choice (choose one)', 'multi-choice (choose all)', 'date', 'numeric', 'agreement/authorization'.")
    char_limit: Optional[int] = Field(default=None, description="Estimated max characters allowed. Null if not applicable.")
    is_agreement: bool = Field(description="True if requires initials, signature, or authorization.")
    sub_question_of: Optional[str] = Field(default=None, description="Parent question number if this is a sub-question.")
    unique_structure: bool = Field(description="True if the format is complex/unique and not well-described by the existing fields/types.")
    mapped_pdf_fields: Optional[List[str]] = Field(default=None, description="List of PDF field names that map to this question.")
    mappings: List[QuestionMapping] = Field(default_factory=list, description="Mappings to canonical questions from different versions.")


class FormStructure(BaseModel):
    """The complete structure and metadata extracted from the PDF form."""
    form_description: str = Field(description="Concise description of form's purpose and target audience.")
    is_form: bool = Field(description="True if document contains fillable fields or explicit questions.")
    general_info_needed: List[str] = Field(description="List of required info applying to multiple questions.")
    questions: List[QuestionDetails] = Field(description="Sequential list of all non-'office use only' questions.")
    pdf_form_fields: Optional[List[FormFieldInfo]] = Field(default=None, description="Form fields extracted from PDF.")
    field_mappings: Optional[List[FieldMapping]] = Field(default=None, description="Mappings between PDF fields and questions.")


class ExtractedMetadataRecord(BaseModel):
    """Complete record stored in database."""
    opportunity_id: str
    pdf_checksum: str  # SHA-256 hash
    pdf_filename: str
    pdf_filepath: Optional[str] = None  # Deprecated - kept for backward compat
    pdf_s3_key: str  # S3 key like "od1_/pdfs/123-abc.pdf" (required)
    process_date: str  # ISO 8601 format
    form_structure: FormStructure
    from_cache: bool = False  # True if returned from cache


# ============================================================================
# Statistics Data Class
# ============================================================================

@dataclass
class ExtractionStats:
    """Statistics from extraction operations."""
    total_processed: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    forms_with_pdf_fields: int = 0
    total_pdf_fields_found: int = 0
    total_questions_extracted: int = 0
    successful_field_mappings: int = 0
    unmapped_fields: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_processed": self.total_processed,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "forms_with_pdf_fields": self.forms_with_pdf_fields,
            "total_pdf_fields_found": self.total_pdf_fields_found,
            "total_questions_extracted": self.total_questions_extracted,
            "successful_field_mappings": self.successful_field_mappings,
            "unmapped_fields": self.unmapped_fields,
            "mapping_rate": f"{self.successful_field_mappings}/{self.total_pdf_fields_found}" if self.total_pdf_fields_found > 0 else "N/A",
            "errors": self.errors
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# PyPDFForm Extraction (Standalone, Reusable)
# ============================================================================

def get_pdf_form_schema(pdf_path: str) -> Dict[str, Any]:
    """
    Get the raw PyPDFForm schema for a PDF. Useful for troubleshooting
    and introspection. This is the raw output from PyPDFForm.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Raw schema dict from PyPDFForm, or empty dict if no fields/error
    """
    try:
        from PyPDFForm import PdfWrapper
    except ImportError:
        logger.warning("PyPDFForm not installed.")
        return {}

    try:
        pdf = PdfWrapper(pdf_path)
        return pdf.schema or {}
    except Exception as e:
        logger.error(f"Error getting PDF schema from {pdf_path}: {e}")
        return {}


def get_pdf_field_names(pdf_path: str) -> List[str]:
    """
    Get just the field names from a PDF form. Quick utility for introspection.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of field names
    """
    schema = get_pdf_form_schema(pdf_path)
    properties = schema.get("properties", {})
    return list(properties.keys())


def parse_pdf_form_schema(schema: Dict[str, Any]) -> List[FormFieldInfo]:
    """
    Parse a PyPDFForm schema dict into FormFieldInfo objects.
    Separated for reuse - you can get the schema once and parse it multiple times.

    Args:
        schema: Raw schema dict from get_pdf_form_schema() or PyPDFForm

    Returns:
        List of FormFieldInfo objects
    """
    fields = []

    if not schema:
        return fields

    # PyPDFForm returns JSON Schema format: {"type": "object", "properties": {...}}
    field_definitions = schema.get("properties", {})

    if not field_definitions:
        # Fallback: maybe it's a flat dict of fields
        if "type" not in schema or schema.get("type") != "object":
            field_definitions = schema
        else:
            return fields

    for field_name, field_info in field_definitions.items():
        # Determine field type from JSON Schema type
        json_type = field_info.get("type", "unknown")

        # Map JSON Schema types to our field types
        if json_type == "boolean":
            mapped_type = "checkbox"
        elif json_type == "integer" and "maximum" in field_info:
            # Integer with maximum typically indicates radio buttons or dropdown
            max_val = field_info.get("maximum", 0)
            mapped_type = "radio" if max_val <= 5 else "dropdown"
        elif json_type == "string":
            mapped_type = "text"
        else:
            mapped_type = json_type

        # Get description as potential label
        description = field_info.get("description", field_name)

        # Get enum options if available
        options = None
        if "enum" in field_info:
            options = [str(o) for o in field_info["enum"]]
        elif mapped_type == "radio" and "maximum" in field_info:
            # Radio options are typically 0 to maximum
            options = [str(i) for i in range(field_info["maximum"] + 1)]

        fields.append(FormFieldInfo(
            field_name=field_name,
            field_type=mapped_type,
            field_value=description if description != field_name else None,
            field_options=options
        ))

    return fields


def extract_pdf_form_fields(pdf_path: str) -> List[FormFieldInfo]:
    """
    Extract form field information from a PDF using PyPDFForm.
    Convenience function that combines get_pdf_form_schema + parse_pdf_form_schema.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of FormFieldInfo objects describing each form field
    """
    schema = get_pdf_form_schema(pdf_path)

    if not schema:
        logger.info(f"No form fields found in {pdf_path}")
        return []

    fields = parse_pdf_form_schema(schema)
    logger.info(f"Extracted {len(fields)} form fields from {pdf_path}")
    return fields

    return fields


# ============================================================================
# Core Functions
# ============================================================================

def calculate_pdf_checksum(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Hexadecimal SHA-256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_cache_index(jsonl_path: Path = JSONL_PATH) -> Dict[str, Dict]:
    """
    Load form records from database and create an index by opportunity_id.

    Args:
        jsonl_path: Deprecated - kept for backwards compatibility

    Returns:
        Dictionary mapping opportunity_id to the record dict
    """
    cache_index = {}

    try:
        records = load_form_records_db()
        for record in records:
            opp_id = record.get("opportunity_id")
            if opp_id:
                cache_index[opp_id] = record

        logger.info(f"Loaded {len(cache_index)} records from database cache.")
    except Exception as e:
        logger.error(f"Error loading from database: {e}")

    return cache_index


def check_cache(
    opp_id: str,
    checksum: str,
    cache_index: Dict[str, Dict]
) -> Tuple[bool, Optional[ExtractedMetadataRecord]]:
    """
    Check if a valid cached record exists for the given opportunity_id.

    Args:
        opp_id: Opportunity ID to look up
        checksum: SHA-256 checksum of the current PDF
        cache_index: Pre-loaded cache index

    Returns:
        Tuple of (cache_hit, cached_record or None)
    """
    if opp_id not in cache_index:
        return False, None

    cached_data = cache_index[opp_id]
    cached_checksum = cached_data.get("pdf_checksum", "")

    if cached_checksum == checksum:
        # Valid cache hit - checksum matches
        try:
            record = ExtractedMetadataRecord(**cached_data)
            record.from_cache = True
            return True, record
        except Exception as e:
            logger.warning(f"Failed to parse cached record for {opp_id}: {e}")
            return False, None
    else:
        # Checksum mismatch - PDF has changed
        logger.info(f"Checksum mismatch for {opp_id}. PDF has changed, re-extracting.")
        return False, None


def append_to_jsonl(record: ExtractedMetadataRecord, jsonl_path: Path = JSONL_PATH) -> None:
    """
    Save a new record to the database.

    Args:
        record: The ExtractedMetadataRecord to save
        jsonl_path: Deprecated - kept for backwards compatibility
    """
    save_form_record_db(record.model_dump())
    logger.info(f"Saved record for opportunity_id={record.opportunity_id} to database.")


def _format_form_fields_for_prompt(fields: List[FormFieldInfo]) -> str:
    """Format extracted form fields for inclusion in the LLM prompt."""
    if not fields:
        return "No fillable PDF form fields were detected by the PDF extractor."

    lines = [
        "The following fillable PDF form fields were extracted by PyPDFForm (the PDF extractor):",
        "NOTE: 'Extractor Type' is the raw field type from the PDF structure, which you must preserve in field_mappings.pdf_field_type.",
        ""
    ]

    for i, field in enumerate(fields, 1):
        field_desc = f"{i}. Field Name: '{field.field_name}' | Extractor Type: {field.field_type}"
        if field.field_options:
            field_desc += f" | Options: {field.field_options}"
        if field.field_value:
            field_desc += f" | Description: {field.field_value}"
        lines.append(field_desc)

    return "\n".join(lines)


def _build_extraction_prompt(form_fields: List[FormFieldInfo]) -> str:
    """Build the prompt for LLM extraction with form field mapping."""
    form_fields_text = _format_form_fields_for_prompt(form_fields)

    return f"""Analyze the provided PDF form. Extract all non-'for office use only' questions and the form's metadata, adhering strictly to the required FormStructure schema.

## PDF Form Fields Detected
{form_fields_text}

## Instructions

### Part 1: Question Extraction
1. Identify all questions, fields, and input areas that a user needs to fill out
2. Skip any sections marked as "for office use only" or similar administrative sections
3. For each question, determine:
   - Whether it's required or optional
   - The question number/identifier if visible
   - Both a short and full version of the question text
   - The question type (open ended, multiple choice, date, numeric, etc.)
   - Character limits if applicable
   - Whether it's an agreement/authorization field
   - If it's a sub-question of another question
   - If the format is complex or unique
4. Provide a concise description of the form's purpose
5. List any general information that applies to multiple questions

### Part 2: Field Mapping (if PDF form fields were detected)
For each PDF form field listed above, create a mapping entry:
- **pdf_field_name**: Copy the exact field name from the extractor
- **pdf_field_type**: Copy the exact Extractor Type (text, checkbox, radio, dropdown, signature) - this MUST match what the extractor reported
- **question_number/question_text_short**: Map to the corresponding question you extracted
- **confidence**: 'high' (obvious match), 'medium' (reasonable but ambiguous), 'low' (uncertain)
- **mapping_notes**: Explain any issues or why a field couldn't be mapped

Consider the extractor's field type when determining your question_type:
- Extractor 'text' → typically 'open ended', 'date', or 'numeric' depending on context
- Extractor 'checkbox' → typically part of 'multi-choice (choose all)'
- Extractor 'radio' → typically 'multiple choice (choose one)'
- Extractor 'dropdown' → typically 'multiple choice (choose one)'

If a field cannot be mapped to any question, still include it in field_mappings with null question references.

### Part 3: Populate mapped_pdf_fields
For each question, populate the mapped_pdf_fields list with the PDF field names that map to that question.

Return the structured data according to the FormStructure schema."""


def extract_form_metadata(
    opp_id: str,
    s3_key: str,
    use_cache: bool = True,
    cache_index: Optional[Dict[str, Dict]] = None,
    stats: Optional[ExtractionStats] = None
) -> Tuple[ExtractedMetadataRecord, bool]:
    """
    Extract metadata from a PDF form stored in S3, with optional caching.

    Args:
        opp_id: Opportunity ID associated with this PDF
        s3_key: S3 key for the PDF (required - S3 is the only storage)
        use_cache: Whether to use cached results if available
        cache_index: Pre-loaded cache index (optional, will load if not provided)
        stats: Optional ExtractionStats object to update

    Returns:
        Tuple of (ExtractedMetadataRecord, was_from_cache)
    """
    from s3_storage import temp_pdf_from_s3

    if not s3_key:
        raise ValueError("s3_key is required - S3 is the only storage option")

    # Download PDF from S3 to temp file for processing
    with temp_pdf_from_s3(s3_key) as temp_pdf_path:
        pdf_path = Path(temp_pdf_path)
        pdf_filename = Path(s3_key).name  # Get filename from S3 key

        # Calculate checksum
        checksum = calculate_pdf_checksum(str(pdf_path))
        logger.info(f"PDF checksum for {pdf_filename}: {checksum[:16]}...")

        # Check cache if enabled
        if use_cache:
            if cache_index is None:
                cache_index = load_cache_index()

            cache_hit, cached_record = check_cache(opp_id, checksum, cache_index)
            if cache_hit and cached_record:
                logger.info(f"Cache hit for opportunity_id={opp_id}")
                if stats:
                    stats.cache_hits += 1
                    stats.successful_extractions += 1
                    if cached_record.form_structure.pdf_form_fields:
                        stats.forms_with_pdf_fields += 1
                        stats.total_pdf_fields_found += len(cached_record.form_structure.pdf_form_fields)
                    if cached_record.form_structure.questions:
                        stats.total_questions_extracted += len(cached_record.form_structure.questions)
                    if cached_record.form_structure.field_mappings:
                        mapped = sum(1 for m in cached_record.form_structure.field_mappings if m.question_number or m.question_text_short)
                        stats.successful_field_mappings += mapped
                        stats.unmapped_fields += len(cached_record.form_structure.field_mappings) - mapped
                return cached_record, True

        if stats:
            stats.cache_misses += 1

        # Extract PDF form fields using PyPDFForm
        logger.info(f"Extracting PDF form fields from {pdf_filename}...")
        pdf_form_fields = extract_pdf_form_fields(str(pdf_path))

        # Cache miss or cache disabled - call LLM
        logger.info(f"Extracting metadata from {pdf_filename} using LLM...")

        prompt = _build_extraction_prompt(pdf_form_fields)

        raw_response, parsed_output = get_and_parse_structured_response(
            prompt=prompt,
            output_schema=FormStructure,
            file_paths=[str(pdf_path)]
        )

        # Handle errors
        if isinstance(parsed_output, str) and "error" in parsed_output.lower():
            raise RuntimeError(f"LLM extraction failed: {parsed_output}")

        # Parse the form structure
        if isinstance(parsed_output, FormStructure):
            form_structure = parsed_output
        elif isinstance(parsed_output, dict):
            form_structure = FormStructure(**parsed_output)
        else:
            raise RuntimeError(f"Unexpected output type: {type(parsed_output)}")

        # Ensure PDF form fields are included in the structure
        if pdf_form_fields and not form_structure.pdf_form_fields:
            form_structure.pdf_form_fields = pdf_form_fields

        # Generate stable question_id for each question (fq_0, fq_1, etc.)
        for idx, question in enumerate(form_structure.questions):
            if not question.question_id:
                question.question_id = f"fq_{idx}"

        # Create the record (using S3 key, no local filepath)
        record = ExtractedMetadataRecord(
            opportunity_id=opp_id,
            pdf_checksum=checksum,
            pdf_filename=pdf_filename,
            pdf_filepath=None,  # Deprecated - using S3 only
            pdf_s3_key=s3_key,
            process_date=datetime.now().isoformat(),
            form_structure=form_structure,
            from_cache=False
        )

        # Update stats
        if stats:
            stats.successful_extractions += 1
            if form_structure.pdf_form_fields:
                stats.forms_with_pdf_fields += 1
                stats.total_pdf_fields_found += len(form_structure.pdf_form_fields)
            if form_structure.questions:
                stats.total_questions_extracted += len(form_structure.questions)
            if form_structure.field_mappings:
                mapped = sum(1 for m in form_structure.field_mappings if m.question_number or m.question_text_short)
                stats.successful_field_mappings += mapped
                stats.unmapped_fields += len(form_structure.field_mappings) - mapped

        # Append to cache
        append_to_jsonl(record)

        return record, False


def extract_batch(
    items: List[Tuple[str, str]],
    use_cache: bool = True
) -> Tuple[List[Tuple[Optional[ExtractedMetadataRecord], bool, Optional[str]]], ExtractionStats]:
    """
    Batch process multiple PDFs for metadata extraction.

    Args:
        items: List of (opp_id, s3_key) tuples
        use_cache: Whether to use cached results

    Returns:
        Tuple of (results_list, stats) where results_list contains
        (record, was_from_cache, error_message) tuples
    """
    results = []
    stats = ExtractionStats()

    # Pre-load cache once for efficiency
    cache_index = load_cache_index() if use_cache else {}

    for opp_id, s3_key in items:
        stats.total_processed += 1
        try:
            record, was_cached = extract_form_metadata(
                opp_id=opp_id,
                s3_key=s3_key,
                use_cache=use_cache,
                cache_index=cache_index,
                stats=stats
            )
            results.append((record, was_cached, None))
        except Exception as e:
            logger.error(f"Failed to extract {s3_key}: {e}")
            stats.failed_extractions += 1
            stats.errors.append(f"{opp_id}: {str(e)}")
            results.append((None, False, str(e)))

    logger.info(f"Batch extraction complete. Stats:\n{stats}")

    return results, stats


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python extract_metadata.py <s3_key> <opportunity_id>")
        print("Example: python extract_metadata.py od1_/pdfs/12345-abc.pdf 12345")
        sys.exit(1)

    s3_key = sys.argv[1]
    opp_id = sys.argv[2]

    stats = ExtractionStats()
    stats.total_processed = 1

    try:
        record, from_cache = extract_form_metadata(opp_id=opp_id, s3_key=s3_key, stats=stats)
        print(f"\nExtraction {'(from cache)' if from_cache else '(fresh)'} complete:")
        print(record.model_dump_json(indent=2))
        print(f"\n--- Stats ---")
        print(stats)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
