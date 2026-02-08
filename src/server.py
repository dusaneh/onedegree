"""
Flask API Server for Form Mapping System.

Endpoints:
- POST /api/pdf - Insert a new PDF by URL and opportunity ID
- POST /api/forms/metadata - Get form metadata with canonical overrides
- POST /api/canonical/create - Create new canonical questions version
- POST /api/webhook/canonical - Simulated webhook receiver
"""

import logging
import random
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Any, Dict, List, Optional, Tuple
import requests

from client_fx import download_pdf, DownloadResult
from extract_metadata import (
    extract_form_metadata,
    load_cache_index,
    calculate_pdf_checksum,
    ExtractionStats,
    JSONL_PATH,
)
from form_mapping import (
    load_canonical_questions,
    load_form_records,
    map_single_form,
    map_all_forms,
    update_form_with_mappings,
    save_updated_forms,
    is_already_mapped,
    FormMappingStats,
)
from canonical_questions import (
    list_canonical_versions,
    load_and_simplify_all_forms,
    identify_canonical_questions,
    save_cached_result,
    compute_version_id,
    get_model_config,
    calculate_tokens_progressive,
    GEMINI_2_5_FLASH,
    GEMINI_2_5_PRO,
    GEMINI_3_FLASH,
)
from pdf_writer import (
    fill_form,
    FillFormRequest,
    FillFormResponse,
    FillOptions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================================
# Run Tracking (in-memory for simplicity, use Redis/DB in production)
# ============================================================================

canonical_runs: Dict[str, Dict[str, Any]] = {}
webhook_received: List[Dict[str, Any]] = []  # Store received webhooks for testing
SERVER_START_TIME = datetime.now().isoformat()


def create_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())[:8]


def update_run_status(
    run_id: str,
    status: str,
    **kwargs
) -> None:
    """Update the status of a run."""
    if run_id in canonical_runs:
        canonical_runs[run_id]["status"] = status
        canonical_runs[run_id]["updated_at"] = datetime.now().isoformat()
        canonical_runs[run_id].update(kwargs)
        logger.info(f"[Run {run_id}] Status updated: {status}")


def cleanup_incomplete_runs() -> None:
    """Mark any incomplete runs as failed (called on server startup)."""
    incomplete_statuses = ["loading_forms", "processing", "calling_llm", "mapping_forms"]
    for run_id, run_data in canonical_runs.items():
        if run_data.get("status") in incomplete_statuses:
            logger.warning(f"[Run {run_id}] Marking as failed (incomplete at server startup)")
            run_data["status"] = "failed"
            run_data["error"] = "Server restarted before completion"
            run_data["updated_at"] = datetime.now().isoformat()


def send_webhook(
    webhook_url: str,
    run_id: str,
    payload: Dict[str, Any]
) -> bool:
    """Send webhook notification."""
    try:
        logger.info(f"Sending webhook to {webhook_url} for run {run_id}")
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        logger.info(f"Webhook response: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Webhook failed: {e}")
        return False


def run_canonical_creation_task(
    run_id: str,
    form_ids: Optional[List[str]],
    order: str,
    model_name: str,
    webhook_url: Optional[str]
) -> None:
    """
    Background task to create canonical questions.

    Args:
        run_id: Unique run identifier
        form_ids: Optional list of form IDs to use
        order: 'latest_first', 'oldest_first', or 'random'
        model_name: LLM model to use
        webhook_url: Optional URL to call when complete
    """
    import time
    task_start = time.time()

    try:
        logger.info(f"[Run {run_id}] === CANONICAL CREATION STARTED ===")
        logger.info(f"[Run {run_id}] Model: {model_name}, Order: {order}")
        update_run_status(run_id, "loading_forms")

        # Load forms
        step_start = time.time()
        logger.info(f"[Run {run_id}] Step 1/4: Loading and simplifying forms...")
        forms = load_and_simplify_all_forms(form_ids=form_ids)
        logger.info(f"[Run {run_id}] Step 1/4: Loaded {len(forms)} forms in {time.time() - step_start:.1f}s")

        if not forms:
            update_run_status(run_id, "failed", error="No forms found")
            if webhook_url:
                send_webhook(webhook_url, run_id, {
                    "run_id": run_id,
                    "status": "failed",
                    "error": "No forms found"
                })
            return

        # Apply ordering
        if order == "latest_first":
            forms = list(reversed(forms))
        elif order == "random":
            random.shuffle(forms)

        total_questions = sum(len(f.questions) for f in forms)
        logger.info(f"[Run {run_id}] Total questions across all forms: {total_questions}")

        update_run_status(
            run_id,
            "processing",
            total_forms=len(forms),
            total_questions=total_questions,
            order=order
        )

        # Get model config
        config = get_model_config(model_name)

        # Compute version ID
        version_id = compute_version_id(forms)

        # Calculate token estimates BEFORE calling LLM
        forms_to_process, token_estimate = calculate_tokens_progressive(forms, config)
        logger.info(f"[Run {run_id}] Token estimates - Input: {token_estimate.estimated_input_tokens:,}, Output: {token_estimate.estimated_output_tokens:,}")

        if len(forms_to_process) < len(forms):
            logger.warning(f"[Run {run_id}] Only {len(forms_to_process)}/{len(forms)} forms fit within token limits")

        update_run_status(
            run_id,
            "calling_llm",
            version_id=version_id,
            estimated_input_tokens=token_estimate.estimated_input_tokens,
            estimated_output_tokens=token_estimate.estimated_output_tokens,
            forms_to_process=len(forms_to_process),
            forms_truncated=len(forms) - len(forms_to_process)
        )

        # Run canonical identification (this is the slow LLM call)
        step_start = time.time()
        logger.info(f"[Run {run_id}] Step 2/4: Calling LLM to identify canonical questions...")
        logger.info(f"[Run {run_id}] This may take several minutes for {len(forms_to_process)} forms...")
        result = identify_canonical_questions(forms, config)
        llm_time = time.time() - step_start
        logger.info(f"[Run {run_id}] Step 2/4: LLM call completed in {llm_time:.1f}s")
        logger.info(f"[Run {run_id}] Found {len(result.canonical_questions)} canonical questions")
        logger.info(f"[Run {run_id}] Tokens used - Input: {result.input_token_count}, Output: {result.output_token_count}")

        # Save result
        step_start = time.time()
        logger.info(f"[Run {run_id}] Step 3/4: Saving canonical version...")
        save_cached_result(result)
        logger.info(f"[Run {run_id}] Step 3/4: Saved in {time.time() - step_start:.1f}s")

        # Now map all existing forms to this new canonical version
        update_run_status(run_id, "mapping_forms", version_id=result.version_id)
        step_start = time.time()
        logger.info(f"[Run {run_id}] Step 4/4: Mapping all forms to canonical version {result.version_id}...")

        mapping_stats = map_all_forms(
            force=False,
            canonical_version=result.version_id,
            config=config
        )
        logger.info(f"[Run {run_id}] Step 4/4: Mapping completed in {time.time() - step_start:.1f}s")

        # Prepare stats
        total_time = time.time() - task_start
        stats = {
            "version_id": result.version_id,
            "created_at": result.created_at,
            "model_used": result.model_used,
            "input_token_count": result.input_token_count,
            "output_token_count": result.output_token_count,
            "canonical_questions_count": len(result.canonical_questions),
            "forms_processed": len(forms),
            "processing_notes": result.processing_notes,
            "mapping_stats": mapping_stats.to_dict() if mapping_stats else None,
            "total_time_seconds": round(total_time, 1)
        }

        update_run_status(
            run_id,
            "completed",
            stats=stats,
            version_id=result.version_id
        )

        logger.info(f"[Run {run_id}] === CANONICAL CREATION COMPLETED ===")
        logger.info(f"[Run {run_id}] Version: {result.version_id}")
        logger.info(f"[Run {run_id}] Canonical questions: {len(result.canonical_questions)}")
        logger.info(f"[Run {run_id}] Total time: {total_time:.1f}s")

        # Send webhook if configured
        if webhook_url:
            send_webhook(webhook_url, run_id, {
                "run_id": run_id,
                "status": "completed",
                "version_id": result.version_id,
                "stats": stats
            })

    except Exception as e:
        logger.exception(f"Run {run_id} failed: {e}")
        update_run_status(run_id, "failed", error=str(e))

        if webhook_url:
            send_webhook(webhook_url, run_id, {
                "run_id": run_id,
                "status": "failed",
                "error": str(e)
            })


# ============================================================================
# Helper Functions
# ============================================================================

def get_latest_canonical_version() -> Optional[str]:
    """Get the latest canonical version ID."""
    versions = list_canonical_versions()
    if not versions:
        return None
    # Find the one marked as latest, or return the last one
    for v in versions:
        if v.get("is_latest"):
            return v["version_id"]
    return versions[-1]["version_id"] if versions else None


def find_existing_record(
    opp_id: str,
    checksum: str,
    records: list
) -> Tuple[Optional[Dict], bool]:
    """
    Find existing record by opp_id and check checksum.

    Returns:
        Tuple of (record_or_None, checksum_matches)
    """
    for record in records:
        if record.get("opportunity_id") == opp_id:
            if record.get("pdf_checksum") == checksum:
                return record, True
            else:
                return record, False
    return None, False


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "form-mapping-api"})


@app.route("/api/canonical/versions", methods=["GET"])
def get_canonical_versions():
    """Get list of available canonical question versions."""
    try:
        versions = list_canonical_versions()
        latest = get_latest_canonical_version()
        return jsonify({
            "success": True,
            "latest_version_id": latest,
            "versions": versions
        })
    except Exception as e:
        logger.exception("Failed to get canonical versions")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/canonical/create", methods=["POST"])
def create_canonical():
    """
    Create a new canonical questions version.

    This is an asynchronous operation. Returns immediately with a run_id.
    Optionally sends a webhook when complete.

    Request body:
    {
        "form_ids": ["12345", "67890"],  // Optional: specific forms to use
        "order": "latest_first",          // "latest_first", "oldest_first", "random"
        "model": "gemini-3-flash-preview",  // Optional: model to use
        "webhook_url": "https://..."      // Optional: URL to call when complete
    }

    Response:
    {
        "success": true,
        "run_id": "abc12345",
        "status": "started",
        "message": "Canonical creation started"
    }
    """
    try:
        data = request.get_json() or {}

        # Parse options
        form_ids = data.get("form_ids")  # None means all forms
        order = data.get("order", "latest_first")
        model_name = data.get("model", "gemini-3-flash-preview")
        webhook_url = data.get("webhook_url")

        # Validate order
        valid_orders = ["latest_first", "oldest_first", "random"]
        if order not in valid_orders:
            return jsonify({
                "success": False,
                "error": f"Invalid order. Must be one of: {valid_orders}"
            }), 400

        # Create run
        run_id = create_run_id()
        canonical_runs[run_id] = {
            "run_id": run_id,
            "status": "started",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "form_ids": form_ids,
            "order": order,
            "model": model_name,
            "webhook_url": webhook_url
        }

        logger.info(f"Starting canonical creation run {run_id}, "
                    f"order={order}, model={model_name}, "
                    f"forms={'all' if form_ids is None else len(form_ids)}")

        # Start background task
        thread = threading.Thread(
            target=run_canonical_creation_task,
            args=(run_id, form_ids, order, model_name, webhook_url),
            daemon=True
        )
        thread.start()

        return jsonify({
            "success": True,
            "run_id": run_id,
            "status": "started",
            "message": "Canonical creation started in background"
        })

    except Exception as e:
        logger.exception("Failed to start canonical creation")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/canonical/run/<run_id>", methods=["GET"])
def get_canonical_run_status(run_id: str):
    """
    Get the status of a canonical creation run.

    Response:
    {
        "success": true,
        "run": {
            "run_id": "abc12345",
            "status": "completed",
            "version_id": "c3f1d1c4110f",
            "stats": {...}
        }
    }
    """
    if run_id not in canonical_runs:
        return jsonify({
            "success": False,
            "error": f"Run {run_id} not found"
        }), 404

    return jsonify({
        "success": True,
        "run": canonical_runs[run_id]
    })


@app.route("/api/canonical/runs", methods=["GET"])
def list_canonical_runs():
    """List all canonical creation runs."""
    return jsonify({
        "success": True,
        "runs": list(canonical_runs.values())
    })


@app.route("/api/webhook/canonical", methods=["POST"])
def receive_webhook():
    """
    Simulated webhook receiver for testing.

    This endpoint receives webhooks from canonical creation runs
    and stores them for inspection.
    """
    try:
        data = request.get_json()
        webhook_entry = {
            "received_at": datetime.now().isoformat(),
            "payload": data
        }
        webhook_received.append(webhook_entry)

        logger.info(f"Webhook received: run_id={data.get('run_id')}, "
                    f"status={data.get('status')}")

        return jsonify({
            "success": True,
            "message": "Webhook received"
        })

    except Exception as e:
        logger.exception("Failed to process webhook")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/webhook/received", methods=["GET"])
def list_received_webhooks():
    """List all received webhooks (for testing)."""
    return jsonify({
        "success": True,
        "count": len(webhook_received),
        "webhooks": webhook_received
    })


@app.route("/api/webhook/clear", methods=["POST"])
def clear_received_webhooks():
    """Clear all received webhooks (for testing)."""
    webhook_received.clear()
    return jsonify({
        "success": True,
        "message": "Webhooks cleared"
    })


@app.route("/api/forms/metadata", methods=["POST"])
def get_forms_metadata():
    """
    Get form metadata with canonical overrides.

    Request body:
    {
        "forms": [{"opp_id": "12345", "checksum": "abc123..."}, ...],
        "version_id": "c3f1d1c4110f",
        "min_similarity": 3
    }

    Response includes questions with canonical overrides when similarity >= min_similarity,
    plus debug info showing both canonical and original data.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must be JSON"
            }), 400

        # 1. Parse and validate parameters
        forms_request = data.get("forms", [])
        version_id = data.get("version_id")
        min_similarity = data.get("min_similarity")

        # 2. Validate required params
        if not forms_request:
            return jsonify({
                "success": False,
                "error": "Missing required field: forms"
            }), 400

        if not version_id:
            return jsonify({
                "success": False,
                "error": "Missing required field: version_id"
            }), 400

        if min_similarity is None or not isinstance(min_similarity, int) or not (1 <= min_similarity <= 5):
            return jsonify({
                "success": False,
                "error": "min_similarity must be 1-5"
            }), 400

        # Validate each form entry has opp_id and checksum
        for form_entry in forms_request:
            if not form_entry.get("opp_id") or not form_entry.get("checksum"):
                return jsonify({
                    "success": False,
                    "error": "Each form must have opp_id and checksum"
                }), 400

        # 3. Load canonical questions for version
        try:
            loaded_version_id, canonical_questions = load_canonical_questions(version_id)
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": f"Canonical version not found: {version_id}"
            }), 404
        except FileNotFoundError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

        canonical_by_id = {q["canonical_question_id"]: q for q in canonical_questions}

        # 4. Load form records
        all_records = load_form_records()
        records_by_opp_id = {}
        for record in all_records:
            opp_id = record.get("opportunity_id")
            if opp_id:
                if opp_id not in records_by_opp_id:
                    records_by_opp_id[opp_id] = []
                records_by_opp_id[opp_id].append(record)

        # Build checksum map from request
        checksum_map = {f["opp_id"]: f["checksum"] for f in forms_request}

        # 5. For each requested form: validate checksum, build enriched response
        response_forms = []

        for form_entry in forms_request:
            opp_id = form_entry["opp_id"]
            requested_checksum = form_entry["checksum"]

            # Find matching record(s) for this opp_id
            matching_records = records_by_opp_id.get(opp_id, [])

            if not matching_records:
                return jsonify({
                    "success": False,
                    "error": f"Form not found: {opp_id}"
                }), 404

            # Find record with matching checksum (or use first if none match)
            form_record = None
            checksum_valid = False
            for record in matching_records:
                if record.get("pdf_checksum") == requested_checksum:
                    form_record = record
                    checksum_valid = True
                    break

            # If no checksum match, use the most recent record but mark invalid
            if form_record is None:
                form_record = matching_records[-1]
                checksum_valid = False

            # Build enriched questions
            form_structure = form_record.get("form_structure", {})
            form_description = form_structure.get("form_description", "")
            questions = form_structure.get("questions", [])

            enriched_questions = []
            canonical_count = 0
            original_count = 0

            for idx, q in enumerate(questions):
                # Generate stable question_id (fq_0, fq_1, etc.) - use stored or generate from index
                q_id = q.get("question_id") or f"fq_{idx}"
                # Use original question_number if available, otherwise generate one based on index
                q_num = q.get("question_number") or str(idx + 1)
                original_text = q.get("question_text") or q.get("question_text_short", "")
                original_type = q.get("question_type", "")
                original_char_limit = q.get("char_limit")
                original_is_agreement = q.get("is_agreement", False)
                original_sub_question_of = q.get("sub_question_of")
                original_unique_structure = q.get("unique_structure", False)
                original_is_required = q.get("is_required", False)
                original_mapped_pdf_fields = q.get("mapped_pdf_fields")

                # Find mapping for this canonical version
                mappings = q.get("mappings", [])
                version_mapping = None
                for m in mappings:
                    if m.get("canonical_version_id") == loaded_version_id:
                        version_mapping = m
                        break

                # Determine canonical status and get canonical data
                similarity_score = version_mapping.get("similarity_score") if version_mapping else None
                canonical_question_id = version_mapping.get("canonical_question_id") if version_mapping else None
                is_canonical = False
                canonical_data = None

                if similarity_score is not None and similarity_score >= min_similarity and canonical_question_id:
                    canonical_def = canonical_by_id.get(canonical_question_id)
                    if canonical_def:
                        is_canonical = True
                        canonical_count += 1
                        canonical_data = {
                            "canonical_question_id": canonical_question_id,
                            "question_text": canonical_def.get("question_text", ""),
                            "question_type": canonical_def.get("question_type", ""),
                            "char_limit": canonical_def.get("char_limit"),
                            "is_agreement": canonical_def.get("is_agreement", False),
                            "sub_question_of": canonical_def.get("sub_question_of"),
                            "unique_structure": canonical_def.get("unique_structure", False)
                        }

                if not is_canonical:
                    original_count += 1

                # Build original debug data
                original_data = {
                    "question_id": q_id,
                    "question_number": q_num,
                    "question_text": original_text,
                    "question_type": original_type,
                    "char_limit": original_char_limit,
                    "is_agreement": original_is_agreement,
                    "sub_question_of": original_sub_question_of,
                    "unique_structure": original_unique_structure,
                    "is_required": original_is_required,
                    "mapped_pdf_fields": original_mapped_pdf_fields
                }

                # Build question response - top-level fields from canonical or original
                if is_canonical and canonical_data:
                    question_response = {
                        "question_id": q_id,  # Stable form-level ID for answer submission
                        "question_number": q_num,  # Always from original
                        "question_text": canonical_data["question_text"],
                        "question_type": canonical_data["question_type"],
                        "char_limit": canonical_data["char_limit"],
                        "is_agreement": canonical_data["is_agreement"],
                        "sub_question_of": canonical_data["sub_question_of"],
                        "unique_structure": canonical_data["unique_structure"],
                        "is_canonical": True,
                        "canonical_question_id": canonical_question_id,
                        "similarity_score": similarity_score,
                        "debug": {
                            "canonical": canonical_data,
                            "original": original_data
                        }
                    }
                else:
                    question_response = {
                        "question_id": q_id,  # Stable form-level ID for answer submission
                        "question_number": q_num,
                        "question_text": original_text,
                        "question_type": original_type,
                        "char_limit": original_char_limit,
                        "is_agreement": original_is_agreement,
                        "sub_question_of": original_sub_question_of,
                        "unique_structure": original_unique_structure,
                        "is_canonical": False,
                        "canonical_question_id": None,
                        "similarity_score": similarity_score,
                        "debug": {
                            "canonical": None,
                            "original": original_data
                        }
                    }

                enriched_questions.append(question_response)

            total_questions = len(questions)
            canonical_coverage = (canonical_count / total_questions) if total_questions > 0 else 0.0

            form_response = {
                "opportunity_id": opp_id,
                "checksum": form_record.get("pdf_checksum", ""),
                "checksum_valid": checksum_valid,
                "form_description": form_description,
                "total_questions": total_questions,
                "canonical_count": canonical_count,
                "original_count": original_count,
                "canonical_coverage": round(canonical_coverage, 2),
                "questions": enriched_questions
            }
            response_forms.append(form_response)

        return jsonify({
            "success": True,
            "version_id": loaded_version_id,
            "min_similarity": min_similarity,
            "forms": response_forms
        })

    except Exception as e:
        logger.exception(f"Failed to get forms metadata: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/forms/fill", methods=["POST"])
def fill_form_endpoint():
    """
    Fill a PDF form with provided answers.

    Request body:
    {
        "opp_id": "12345",
        "checksum": "abc123...",
        "version_id": "c3f1d1c4110f",
        "answers": {
            "cq_001": "John",           // Canonical question by ID
            "cq_002": "Doe",             // Canonical question by ID
            "q_5": "Custom answer"       // Non-canonical by question_number
        },
        "options": {
            "truncate_on_char_limit": true,
            "fail_on_missing_optional": false
        }
    }

    Response:
    {
        "success": true,
        "opp_id": "12345",
        "checksum": "abc123...",
        "output_path": "data/filled/12345-abc123-1706000000.pdf",
        "validation_report": {
            "success": true,
            "stages": [...],
            "summary": {...}
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must be JSON"
            }), 400

        # Validate required fields
        required_fields = ["opp_id", "checksum", "version_id", "answers"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400

        # Parse options if provided
        options = None
        if "options" in data:
            try:
                options = FillOptions(**data["options"])
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Invalid options: {str(e)}"
                }), 400

        # Create request object
        try:
            fill_request = FillFormRequest(
                opp_id=str(data["opp_id"]),
                checksum=data["checksum"],
                version_id=data["version_id"],
                answers=data["answers"],
                options=options
            )
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid request: {str(e)}"
            }), 400

        logger.info(f"Filling form: opp_id={fill_request.opp_id}, "
                    f"version={fill_request.version_id}, "
                    f"answers={len(fill_request.answers)}")

        # Call fill_form
        response = fill_form(fill_request)

        # Determine HTTP status code
        if response.success:
            return jsonify(response.model_dump())
        else:
            # Check if it's a "not found" error
            error_msg = response.error or ""
            if "not found" in error_msg.lower():
                return jsonify(response.model_dump()), 404
            elif "checksum mismatch" in error_msg.lower():
                return jsonify(response.model_dump()), 400
            else:
                # Other errors - return 500 for server errors, 400 for validation
                if response.validation_report.stages:
                    # Check what stage failed
                    for stage in response.validation_report.stages:
                        if not stage.passed:
                            if stage.name == "input":
                                return jsonify(response.model_dump()), 400
                            else:
                                return jsonify(response.model_dump()), 500
                return jsonify(response.model_dump()), 500

    except Exception as e:
        logger.exception(f"Failed to fill form: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/examples/<filename>", methods=["GET"])
def serve_example_pdf(filename):
    """
    Serve example PDF files for testing.

    Example: GET /examples/opp1.pdf
    """
    from flask import send_from_directory
    examples_dir = Path(__file__).parent.parent / "examples"

    if not examples_dir.exists():
        return jsonify({"error": "Examples directory not found"}), 404

    file_path = examples_dir / filename
    if not file_path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404

    return send_from_directory(examples_dir, filename)


@app.route("/examples", methods=["GET"])
def list_example_pdfs():
    """List available example PDFs."""
    examples_dir = Path(__file__).parent.parent / "examples"

    if not examples_dir.exists():
        return jsonify({"files": [], "error": "Examples directory not found"})

    files = [f.name for f in examples_dir.glob("*.pdf")]
    return jsonify({"files": sorted(files)})


@app.route("/api/pdf", methods=["POST"])
def insert_pdf():
    """
    Insert a new PDF by URL or local file path and opportunity ID.

    Request body (URL mode):
    {
        "url": "https://example.com/form.pdf",
        "opportunity_id": "12345"
    }

    Request body (local file mode):
    {
        "file_path": "C:/path/to/form.pdf",
        "opportunity_id": "12345"
    }

    Response:
    {
        "success": true,
        "opportunity_id": "12345",
        "checksum": "abc123...",
        "canonical_version_id": "c3f1d1c4110f",
        "from_cache": false,
        "mapping_stats": {...},
        "message": "..."
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must be JSON"
            }), 400

        url = data.get("url")
        file_path = data.get("file_path")
        opp_id = data.get("opportunity_id")

        if not url and not file_path:
            return jsonify({
                "success": False,
                "error": "Missing required field: url or file_path"
            }), 400

        if not opp_id:
            return jsonify({
                "success": False,
                "error": "Missing required field: opportunity_id"
            }), 400

        opp_id = str(opp_id)  # Ensure string

        # Get latest canonical version (may be None if none exist yet)
        latest_version = get_latest_canonical_version()
        if not latest_version:
            logger.info("No canonical version found - will extract metadata without mapping")

        # Upload PDF to S3 - either from URL or local file
        if file_path:
            # Local file mode - upload to S3
            source_path = Path(file_path)
            if not source_path.exists():
                return jsonify({
                    "success": False,
                    "error": f"File not found: {file_path}"
                }), 400

            logger.info(f"Processing local PDF: opp_id={opp_id}, file_path={file_path}")

            # Generate filename for S3
            dest_filename = f"{opp_id}-{source_path.stem[:6]}.pdf"

            # Upload to S3 (required)
            try:
                from s3_storage import upload_pdf as s3_upload
                pdf_s3_key = s3_upload(str(source_path), "pdf", dest_filename)
                logger.info(f"Uploaded to S3: {pdf_s3_key}")
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"S3 upload failed: {str(e)}"
                }), 500

            # Use source file temporarily for checksum calculation
            temp_pdf_path = str(source_path)
        else:
            # URL mode - download_pdf handles S3 upload
            logger.info(f"Processing PDF insert: opp_id={opp_id}, url={url}")
            try:
                download_result: DownloadResult = download_pdf(url, opp_id)
                if not download_result.success:
                    return jsonify({
                        "success": False,
                        "error": f"Failed to download/upload PDF: {download_result.error}"
                    }), 400
                pdf_s3_key = download_result.s3_key
                dest_filename = download_result.filename

                # Download from S3 temporarily to calculate checksum
                from s3_storage import temp_pdf_from_s3
                # We'll calculate checksum inside extract_form_metadata
                temp_pdf_path = None
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Failed to process PDF: {str(e)}"
                }), 400

        # Calculate checksum - download from S3 if needed
        if temp_pdf_path:
            checksum = calculate_pdf_checksum(temp_pdf_path)
        else:
            # Download from S3 to calculate checksum
            from s3_storage import temp_pdf_from_s3
            with temp_pdf_from_s3(pdf_s3_key) as s3_temp_path:
                checksum = calculate_pdf_checksum(s3_temp_path)
        logger.info(f"PDF checksum: {checksum[:16]}...")

        # Load existing records
        records = load_form_records()
        existing_record, checksum_matches = find_existing_record(opp_id, checksum, records)

        response_data = {
            "success": True,
            "opportunity_id": opp_id,
            "checksum": checksum,
            "pdf_s3_key": None,
            "canonical_version_id": latest_version,
            "from_cache": False,
            "mapping_stats": None,
            "message": ""
        }

        if existing_record and checksum_matches:
            # Same opp_id and same checksum - check if mapped to latest version
            logger.info(f"Found cached record for {opp_id} with matching checksum")

            if latest_version and is_already_mapped(existing_record, latest_version):
                # Already mapped to latest - just return stats
                logger.info(f"Already mapped to latest version {latest_version}")

                # Calculate stats from existing mappings
                questions = existing_record.get("form_structure", {}).get("questions", [])
                stats = {
                    "total_questions": len(questions),
                    "questions_with_match": 0,
                    "questions_no_match": 0,
                    "score_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                }

                for q in questions:
                    for m in q.get("mappings", []):
                        if m.get("canonical_version_id") == latest_version:
                            score = m.get("similarity_score", 1)
                            if 1 <= score <= 5:
                                stats["score_distribution"][score] += 1
                            if m.get("canonical_question_id") and score >= 3:
                                stats["questions_with_match"] += 1
                            else:
                                stats["questions_no_match"] += 1
                            break

                response_data["from_cache"] = True
                response_data["pdf_s3_key"] = existing_record.get("pdf_s3_key")
                response_data["mapping_stats"] = stats
                response_data["message"] = "Form already exists and is mapped to latest canonical version"
                return jsonify(response_data)

            else:
                # Exists but not mapped to latest - run mapping if canonical version exists
                if latest_version:
                    logger.info(f"Exists but not mapped to latest version, running mapping...")

                    # Load canonical questions
                    version_id, canonical_questions = load_canonical_questions(latest_version)

                    # Run mapping
                    mappings, form_stats = map_single_form(
                        existing_record,
                        version_id,
                        canonical_questions,
                        GEMINI_2_5_FLASH
                    )

                    # Update record
                    update_form_with_mappings(existing_record, mappings, version_id)

                    # Save all records
                    save_updated_forms(records)

                    response_data["from_cache"] = True
                    response_data["pdf_s3_key"] = existing_record.get("pdf_s3_key")
                    response_data["mapping_stats"] = form_stats.to_dict()
                    response_data["message"] = "Form existed, mapped to latest canonical version"
                else:
                    logger.info("Form exists, no canonical version available for mapping")
                    response_data["from_cache"] = True
                    response_data["pdf_s3_key"] = existing_record.get("pdf_s3_key")
                    response_data["canonical_version_id"] = None
                    response_data["message"] = "Form exists, no canonical version available for mapping"
                return jsonify(response_data)

        elif existing_record and not checksum_matches:
            # Same opp_id but different checksum - preserve old, insert new
            logger.info(f"Found existing record for {opp_id} with different checksum, inserting new version")

            # Extract metadata for new PDF
            stats = ExtractionStats()
            stats.total_processed = 1

            new_record, _ = extract_form_metadata(
                opp_id=opp_id,
                s3_key=pdf_s3_key,
                use_cache=False,  # Force new extraction
                stats=stats
            )

            new_record_dict = new_record.model_dump()
            response_data["pdf_s3_key"] = pdf_s3_key

            # Map to canonical questions if version exists
            if latest_version:
                version_id, canonical_questions = load_canonical_questions(latest_version)

                mappings, form_stats = map_single_form(
                    new_record_dict,
                    version_id,
                    canonical_questions,
                    GEMINI_2_5_FLASH
                )

                update_form_with_mappings(new_record_dict, mappings, version_id)
                response_data["mapping_stats"] = form_stats.to_dict()
            else:
                response_data["canonical_version_id"] = None
                logger.info("No canonical version available, skipping mapping")

            # Reload records (extract_form_metadata appends to file)
            records = load_form_records()

            # Find and update the newly added record
            for i, r in enumerate(records):
                if r.get("opportunity_id") == opp_id and r.get("pdf_checksum") == checksum:
                    records[i] = new_record_dict
                    break

            save_updated_forms(records)

            response_data["message"] = "New version inserted (different checksum), old version preserved"
            return jsonify(response_data)

        else:
            # New form - extract and optionally map
            logger.info(f"New form, extracting metadata...")

            # Extract metadata
            stats = ExtractionStats()
            stats.total_processed = 1

            new_record, _ = extract_form_metadata(
                opp_id=opp_id,
                s3_key=pdf_s3_key,
                use_cache=True,
                stats=stats
            )

            new_record_dict = new_record.model_dump()
            response_data["pdf_s3_key"] = pdf_s3_key

            # Map to canonical questions if version exists
            if latest_version:
                logger.info(f"Mapping to canonical version {latest_version}...")
                version_id, canonical_questions = load_canonical_questions(latest_version)

                mappings, form_stats = map_single_form(
                    new_record_dict,
                    version_id,
                    canonical_questions,
                    GEMINI_2_5_FLASH
                )

                update_form_with_mappings(new_record_dict, mappings, version_id)
                response_data["mapping_stats"] = form_stats.to_dict()
            else:
                response_data["canonical_version_id"] = None
                logger.info("No canonical version available, skipping mapping")

            # Reload and update records
            records = load_form_records()
            for i, r in enumerate(records):
                if r.get("opportunity_id") == opp_id and r.get("pdf_checksum") == checksum:
                    records[i] = new_record_dict
                    break

            save_updated_forms(records)

            response_data["message"] = "New form inserted" + (" and mapped" if latest_version else " (no canonical version)")
            return jsonify(response_data)

    except Exception as e:
        logger.exception(f"Failed to process PDF insert: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# Form Stats API
# ============================================================================

def calculate_form_stats(records: List[Dict], version_filter: Optional[str] = None) -> Dict:
    """
    Calculate aggregate statistics across all forms.

    Args:
        records: List of form records
        version_filter: Optional canonical version to filter by

    Returns:
        Stats dictionary with min/max/avg for questions and field mappings
    """
    form_stats = []

    for record in records:
        opp_id = record.get("opportunity_id", "unknown")
        checksum = record.get("pdf_checksum", "")
        form_structure = record.get("form_structure", {})

        questions = form_structure.get("questions", [])
        pdf_form_fields = form_structure.get("pdf_form_fields") or []

        num_questions = len(questions)
        num_pdf_fields = len(pdf_form_fields)

        # Count questions with mapped PDF fields
        questions_with_pdf_mapping = sum(
            1 for q in questions
            if q.get("mapped_pdf_fields") and len(q.get("mapped_pdf_fields", [])) > 0
        )

        # Calculate PDF field mapping percentage
        pdf_field_mapping_pct = (questions_with_pdf_mapping / num_questions * 100) if num_questions > 0 else 0

        # Get all canonical versions this form is mapped to
        versions_mapped = set()
        for q in questions:
            for m in q.get("mappings", []):
                versions_mapped.add(m.get("canonical_version_id"))

        # Calculate canonical mapping stats
        canonical_mapped_count = 0
        version_matches = None
        version_total = None

        # If version filter is specified, check if form has that version
        if version_filter:
            has_version = version_filter in versions_mapped
            if not has_version:
                continue

            # Calculate match stats for this specific version
            version_matches = 0
            version_total = 0
            for q in questions:
                for m in q.get("mappings", []):
                    if m.get("canonical_version_id") == version_filter:
                        version_total += 1
                        if m.get("canonical_question_id") and m.get("similarity_score", 0) >= 3:
                            version_matches += 1
                            canonical_mapped_count += 1
                        break
        else:
            # Count canonical matches for any version (use first mapping found)
            for q in questions:
                for m in q.get("mappings", []):
                    if m.get("canonical_question_id") and m.get("similarity_score", 0) >= 3:
                        canonical_mapped_count += 1
                        break

        # Calculate canonical mapping percentage
        canonical_mapping_pct = (canonical_mapped_count / num_questions * 100) if num_questions > 0 else 0

        form_stats.append({
            "opportunity_id": opp_id,
            "checksum": checksum[:12] + "..." if checksum else "",
            "num_questions": num_questions,
            "num_pdf_fields": num_pdf_fields,
            "questions_with_pdf_mapping": questions_with_pdf_mapping,
            "pdf_field_mapping_pct": round(pdf_field_mapping_pct, 1),
            "canonical_mapped_count": canonical_mapped_count,
            "canonical_mapping_pct": round(canonical_mapping_pct, 1),
            "versions_mapped": list(versions_mapped),
            "version_matches": version_matches,
            "version_total": version_total,
            "is_form": form_structure.get("is_form", False)
        })

    # Calculate aggregates (only for actual forms with questions)
    forms_with_questions = [f for f in form_stats if f["num_questions"] > 0]

    if forms_with_questions:
        question_counts = [f["num_questions"] for f in forms_with_questions]
        pdf_mapping_pcts = [f["pdf_field_mapping_pct"] for f in forms_with_questions]
        canonical_mapping_pcts = [f["canonical_mapping_pct"] for f in forms_with_questions]

        aggregate = {
            "total_forms": len(form_stats),
            "forms_with_questions": len(forms_with_questions),
            "questions": {
                "min": min(question_counts),
                "max": max(question_counts),
                "avg": round(sum(question_counts) / len(question_counts), 1),
                "total": sum(question_counts)
            },
            "pdf_field_mapping_pct": {
                "min": round(min(pdf_mapping_pcts), 1),
                "max": round(max(pdf_mapping_pcts), 1),
                "avg": round(sum(pdf_mapping_pcts) / len(pdf_mapping_pcts), 1)
            },
            "canonical_mapping_pct": {
                "min": round(min(canonical_mapping_pcts), 1),
                "max": round(max(canonical_mapping_pcts), 1),
                "avg": round(sum(canonical_mapping_pcts) / len(canonical_mapping_pcts), 1)
            }
        }
    else:
        aggregate = {
            "total_forms": len(form_stats),
            "forms_with_questions": 0,
            "questions": {"min": 0, "max": 0, "avg": 0, "total": 0},
            "pdf_field_mapping_pct": {"min": 0, "max": 0, "avg": 0},
            "canonical_mapping_pct": {"min": 0, "max": 0, "avg": 0}
        }

    return {
        "aggregate": aggregate,
        "forms": form_stats
    }


@app.route("/api/forms/stats", methods=["GET"])
def get_form_stats():
    """
    Get aggregate statistics for all forms.

    Query params:
        version: Optional canonical version ID to filter by

    Response:
    {
        "success": true,
        "aggregate": {
            "total_forms": 36,
            "forms_with_questions": 28,
            "questions": {"min": 2, "max": 25, "avg": 8.5, "total": 238},
            "pdf_field_mapping_pct": {"min": 0, "max": 100, "avg": 45.2}
        },
        "forms": [...]
    }
    """
    version_filter = request.args.get("version")

    records = load_form_records()
    stats = calculate_form_stats(records, version_filter)

    return jsonify({
        "success": True,
        "version_filter": version_filter,
        **stats
    })


@app.route("/api/forms/list", methods=["GET"])
def list_forms():
    """
    List all forms with their canonical versions and detailed stats.

    Query params:
        version: Optional canonical version ID to filter by

    Response:
    {
        "success": true,
        "forms": [
            {
                "opportunity_id": "123",
                "checksum": "abc...",
                "filename": "form.pdf",
                "process_date": "2026-01-27T...",
                "question_count": 5,
                "pdf_field_mapped_count": 4,
                "canonical_mapped_count": 3,
                "is_form": true,
                "versions": ["v1", "v2"]
            }
        ]
    }
    """
    version_filter = request.args.get("version")

    records = load_form_records()
    forms = []

    for record in records:
        form_structure = record.get("form_structure", {})
        questions = form_structure.get("questions", [])

        # Get all versions this form is mapped to
        versions = set()
        canonical_mapped_count = 0
        pdf_field_mapped_count = 0

        for q in questions:
            # Count questions with PDF field mappings
            if q.get("mapped_pdf_fields") and len(q.get("mapped_pdf_fields", [])) > 0:
                pdf_field_mapped_count += 1

            # Track versions and canonical mappings
            for m in q.get("mappings", []):
                v_id = m.get("canonical_version_id")
                versions.add(v_id)

                # If filtering by version, count canonical matches for that version
                if version_filter:
                    if v_id == version_filter:
                        if m.get("canonical_question_id") and m.get("similarity_score", 0) >= 3:
                            canonical_mapped_count += 1
                else:
                    # Count canonical matches for any version (use latest mapping)
                    if m.get("canonical_question_id") and m.get("similarity_score", 0) >= 3:
                        canonical_mapped_count += 1
                        break  # Only count once per question

        # Filter by version if specified
        if version_filter and version_filter not in versions:
            continue

        forms.append({
            "opportunity_id": record.get("opportunity_id"),
            "checksum": record.get("pdf_checksum"),
            "filename": record.get("pdf_filename"),
            "filepath": record.get("pdf_filepath"),
            "pdf_s3_key": record.get("pdf_s3_key"),
            "process_date": record.get("process_date"),
            "question_count": len(questions),
            "pdf_field_mapped_count": pdf_field_mapped_count,
            "canonical_mapped_count": canonical_mapped_count,
            "is_form": form_structure.get("is_form", False),
            "versions": sorted(versions)
        })

    return jsonify({
        "success": True,
        "version_filter": version_filter,
        "total": len(forms),
        "forms": forms
    })


@app.route("/api/canonical/stats", methods=["GET"])
def get_canonical_stats():
    """
    Get statistics for each canonical version.

    Response:
    {
        "success": true,
        "versions": [
            {
                "version_id": "abc123",
                "created_at": "2026-01-27T...",
                "canonical_questions_count": 15,
                "forms_mapped": 28,
                "avg_match_rate": 72.5
            }
        ]
    }
    """
    versions = list_canonical_versions()
    records = load_form_records()

    version_stats = []

    for version in versions:
        version_id = version.get("version_id")

        # Count forms mapped to this version and their match rates
        forms_mapped = 0
        total_questions = 0
        total_matches = 0

        for record in records:
            form_structure = record.get("form_structure", {})
            questions = form_structure.get("questions", [])

            form_has_version = False
            for q in questions:
                for m in q.get("mappings", []):
                    if m.get("canonical_version_id") == version_id:
                        form_has_version = True
                        total_questions += 1
                        if m.get("canonical_question_id") and m.get("similarity_score", 0) >= 3:
                            total_matches += 1
                        break

            if form_has_version:
                forms_mapped += 1

        avg_match_rate = (total_matches / total_questions * 100) if total_questions > 0 else 0

        version_stats.append({
            "version_id": version_id,
            "created_at": version.get("created_at"),
            "model_used": version.get("model_used"),
            "canonical_questions_count": len(version.get("canonical_questions", [])),
            "forms_mapped": forms_mapped,
            "total_questions_mapped": total_questions,
            "total_matches": total_matches,
            "avg_match_rate": round(avg_match_rate, 1)
        })

    return jsonify({
        "success": True,
        "versions": version_stats
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    # Cleanup any incomplete runs from previous server instance
    cleanup_incomplete_runs()

    logger.info(f"Starting server on port {port}")
    logger.info(f"Server start time: {SERVER_START_TIME}")
    app.run(host="0.0.0.0", port=port, debug=debug)
