"""
Client Simulation Script for Form Mapping API.

Usage:
    python client_sim.py --run-examples            # Run complete workflow with example PDFs
    python client_sim.py --run-examples --skip-add # Run workflow, skip adding PDFs
    python client_sim.py --url URL --opp-id ID     # Insert PDF from URL
    python client_sim.py --file-path PATH --opp-id ID  # Insert local PDF
    python client_sim.py --health                  # Health check only
    python client_sim.py --versions                # List canonical versions
    python client_sim.py --create-canonical        # Create new canonical version
    python client_sim.py --run-status RUN_ID       # Check run status
    python client_sim.py --list-runs               # List all canonical runs
    python client_sim.py --list-webhooks           # List received webhooks
    python client_sim.py --get-forms --opp-id ID --checksum CHECKSUM --version VER --min-similarity 3
    python client_sim.py --fill-form --opp-id ID --checksum CHECKSUM --version VER --answers-file answers.json
"""

import argparse
import json
import requests
import sys
import time
from typing import Optional

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:5000"


def print_response(response: requests.Response, title: str = "Response"):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print(f"Response:\n{json.dumps(data, indent=2)}")
    except json.JSONDecodeError:
        print(f"Response (raw): {response.text}")
    print()


def health_check(base_url: str) -> bool:
    """Check if server is healthy."""
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        print_response(response, "Health Check")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        print("Make sure the server is running: python src/server.py")
        return False


def get_canonical_versions(base_url: str) -> Optional[dict]:
    """Get list of canonical versions."""
    try:
        response = requests.get(f"{base_url}/api/canonical/versions", timeout=10)
        print_response(response, "Canonical Versions")
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None


def insert_pdf(base_url: str, url: str = None, opportunity_id: str = None, file_path: str = None) -> Optional[dict]:
    """Insert a PDF by URL or local file path and opportunity ID."""
    try:
        payload = {"opportunity_id": opportunity_id}

        if file_path:
            payload["file_path"] = file_path
            print(f"\nInserting local PDF:")
            print(f"  File: {file_path}")
        else:
            payload["url"] = url
            print(f"\nInserting PDF from URL:")
            print(f"  URL: {url}")
        print(f"  Opportunity ID: {opportunity_id}")

        response = requests.post(
            f"{base_url}/api/pdf",
            json=payload,
            timeout=120  # Long timeout for LLM calls
        )

        print_response(response, f"Insert PDF (opp_id={opportunity_id})")

        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None
    except requests.exceptions.Timeout:
        print(f"\nERROR: Request timed out (LLM processing may take a while)")
        return None


def create_canonical(
    base_url: str,
    form_ids: Optional[list] = None,
    order: str = "latest_first",
    model: str = "gemini-2.5-flash",
    webhook_url: Optional[str] = None
) -> Optional[dict]:
    """Start a canonical creation run."""
    try:
        payload = {
            "order": order,
            "model": model
        }
        if form_ids:
            payload["form_ids"] = form_ids
        if webhook_url:
            payload["webhook_url"] = webhook_url

        print(f"\nStarting canonical creation:")
        print(f"  Order: {order}")
        print(f"  Model: {model}")
        print(f"  Forms: {'all' if not form_ids else form_ids}")
        print(f"  Webhook: {webhook_url or 'none'}")

        response = requests.post(
            f"{base_url}/api/canonical/create",
            json=payload,
            timeout=30
        )

        print_response(response, "Create Canonical")

        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None


def get_run_status(base_url: str, run_id: str) -> Optional[dict]:
    """Get the status of a canonical creation run."""
    try:
        response = requests.get(
            f"{base_url}/api/canonical/run/{run_id}",
            timeout=10
        )
        print_response(response, f"Run Status: {run_id}")

        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None


def list_runs(base_url: str) -> Optional[dict]:
    """List all canonical creation runs."""
    try:
        response = requests.get(
            f"{base_url}/api/canonical/runs",
            timeout=10
        )
        print_response(response, "List Runs")

        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None


def list_webhooks(base_url: str) -> Optional[dict]:
    """List all received webhooks."""
    try:
        response = requests.get(
            f"{base_url}/api/webhook/received",
            timeout=10
        )
        print_response(response, "Received Webhooks")

        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None


def clear_webhooks(base_url: str) -> bool:
    """Clear all received webhooks."""
    try:
        response = requests.post(
            f"{base_url}/api/webhook/clear",
            timeout=10
        )
        print_response(response, "Clear Webhooks")
        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return False


def get_forms_metadata(
    base_url: str,
    forms: list,
    version_id: str,
    min_similarity: int
) -> Optional[dict]:
    """
    Get form metadata with canonical overrides.

    Args:
        base_url: Server URL
        forms: List of {opp_id, checksum} dicts
        version_id: Canonical version ID to use
        min_similarity: Minimum similarity score (1-5) for canonical override

    Returns:
        Response dict or None on error
    """
    try:
        payload = {
            "forms": forms,
            "version_id": version_id,
            "min_similarity": min_similarity
        }

        print(f"\nGetting form metadata:")
        print(f"  Version: {version_id}")
        print(f"  Min Similarity: {min_similarity}")
        print(f"  Forms: {len(forms)}")
        for f in forms:
            print(f"    - {f['opp_id']} (checksum: {f['checksum'][:16]}...)")

        response = requests.post(
            f"{base_url}/api/forms/metadata",
            json=payload,
            timeout=60
        )

        print_response(response, "Get Forms Metadata")

        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None
    except requests.exceptions.Timeout:
        print(f"\nERROR: Request timed out")
        return None


def fill_form(
    base_url: str,
    opp_id: str,
    checksum: str,
    version_id: str,
    answers: dict,
    options: Optional[dict] = None
) -> Optional[dict]:
    """
    Fill a PDF form with provided answers.

    Args:
        base_url: Server URL
        opp_id: Opportunity ID
        checksum: PDF checksum
        version_id: Canonical version ID
        answers: Dict of answers keyed by canonical_question_id or "q_{number}"
        options: Optional fill options dict

    Returns:
        Response dict or None on error
    """
    try:
        payload = {
            "opp_id": opp_id,
            "checksum": checksum,
            "version_id": version_id,
            "answers": answers
        }
        if options:
            payload["options"] = options

        print(f"\nFilling form:")
        print(f"  Opportunity ID: {opp_id}")
        print(f"  Checksum: {checksum[:16]}...")
        print(f"  Version: {version_id}")
        print(f"  Answers: {len(answers)} provided")

        response = requests.post(
            f"{base_url}/api/forms/fill",
            json=payload,
            timeout=300  # Long timeout for LLM transformation
        )

        print_response(response, "Fill Form")

        if response.status_code == 200:
            return response.json()
        return response.json() if response.headers.get("content-type", "").startswith("application/json") else None

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        return None
    except requests.exceptions.Timeout:
        print(f"\nERROR: Request timed out (LLM transformation may take a while)")
        return None


def poll_run_status(base_url: str, run_id: str, max_wait: int = 300) -> Optional[dict]:
    """Poll run status until complete or timeout."""
    print(f"\nPolling run {run_id} status (max {max_wait}s)...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{base_url}/api/canonical/run/{run_id}",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                run = data.get("run", {})
                status = run.get("status", "unknown")

                print(f"  Status: {status}")

                if status in ["completed", "failed"]:
                    print_response(response, f"Final Run Status: {run_id}")
                    return data

            time.sleep(5)  # Poll every 5 seconds

        except requests.exceptions.ConnectionError:
            print(f"\nERROR: Cannot connect to server at {base_url}")
            return None

    print(f"\nERROR: Polling timed out after {max_wait}s")
    return None


def run_examples(base_url: str, skip_add: bool = False):
    """
    Run the complete PDF form processing workflow using example PDFs.

    This demonstrates the full pipeline:
    1. List available example PDFs from server
    2. Add initial PDFs (without canonical mapping if none exists)
    3. Create canonical questions from the forms
    4. Add another PDF (will auto-map to canonical)
    5. Get metadata for a form
    6. Create answers programmatically
    7. Fill the form with those answers

    Args:
        base_url: Server URL
        skip_add: If True, skip adding PDFs (useful if already added)
    """
    print("\n" + "=" * 70)
    print("PDF FORM PROCESSING - COMPLETE WORKFLOW EXAMPLE")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Health check and list available example PDFs
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Health check and list available example PDFs")
    print("-" * 70)

    if not health_check(base_url):
        print("ERROR: Server not available. Start server with: python src/server.py")
        return

    try:
        response = requests.get(f"{base_url}/examples", timeout=10)
        if response.status_code == 200:
            files = response.json().get("files", [])
            print(f"Available example PDFs: {files}")
            if not files:
                print("ERROR: No example PDFs found in examples/ folder")
                return
        else:
            print(f"ERROR: Could not list examples: {response.text}")
            return
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # =========================================================================
    # STEP 2: Add initial PDFs (opp1, opp2, opp3)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Add initial PDFs (may not have canonical mapping yet)")
    print("-" * 70)

    pdf_results = {}
    pdfs_to_add = ["opp1.pdf", "opp2.pdf", "opp3.pdf"]

    if skip_add:
        print("Skipping PDF addition (--skip-add flag set)")
    else:
        for pdf_file in pdfs_to_add:
            if pdf_file not in files:
                print(f"  Skipping {pdf_file} (not found in examples)")
                continue

            opp_id = pdf_file.replace(".pdf", "")
            pdf_url = f"{base_url}/examples/{pdf_file}"

            print(f"\n  Adding {pdf_file} as {opp_id}...")
            result = insert_pdf(base_url, url=pdf_url, opportunity_id=opp_id)

            if result and result.get("success"):
                pdf_results[opp_id] = {
                    "checksum": result.get("checksum"),
                    "canonical_version_id": result.get("canonical_version_id"),
                    "mapped": result.get("mapping_stats") is not None
                }
                print(f"    Checksum: {result.get('checksum', '')[:16]}...")
                print(f"    Canonical version: {result.get('canonical_version_id')}")
                print(f"    Mapped: {pdf_results[opp_id]['mapped']}")
            else:
                print(f"    FAILED: {result}")

    # =========================================================================
    # STEP 3: Create canonical questions
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Create canonical questions from all forms")
    print("-" * 70)

    create_result = create_canonical(base_url, webhook_url=f"{base_url}/api/webhook/canonical")

    if not create_result or not create_result.get("success"):
        print("ERROR: Failed to start canonical creation")
        # Continue anyway - might already exist
    else:
        run_id = create_result.get("run_id")
        print(f"  Run ID: {run_id}")
        print("  Waiting for completion...")

        # Poll for completion
        final_result = poll_run_status(base_url, run_id, max_wait=300)
        if final_result and final_result.get("status") == "completed":
            print("  Canonical creation completed!")
        else:
            print("  WARNING: Canonical creation may not have completed")

    # =========================================================================
    # STEP 4: Get canonical version
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Get canonical version ID")
    print("-" * 70)

    versions_result = get_canonical_versions(base_url)
    if not versions_result or not versions_result.get("versions"):
        print("ERROR: No canonical versions found")
        return

    latest_version = None
    for v in versions_result.get("versions", []):
        if v.get("is_latest"):
            latest_version = v.get("version_id")
            break

    if not latest_version:
        latest_version = versions_result["versions"][0].get("version_id")

    print(f"  Using canonical version: {latest_version}")

    # =========================================================================
    # STEP 5: Add another PDF (opp4 if exists, else use opp1)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Add another PDF (will auto-map to canonical)")
    print("-" * 70)

    # Determine which PDF to use for the fill demo
    demo_pdf = "opp4.pdf" if "opp4.pdf" in files else files[0]
    demo_opp_id = demo_pdf.replace(".pdf", "")

    if not skip_add or demo_opp_id not in pdf_results:
        pdf_url = f"{base_url}/examples/{demo_pdf}"
        print(f"\n  Adding {demo_pdf} as {demo_opp_id}...")
        result = insert_pdf(base_url, url=pdf_url, opportunity_id=demo_opp_id)

        if result and result.get("success"):
            pdf_results[demo_opp_id] = {
                "checksum": result.get("checksum"),
                "canonical_version_id": result.get("canonical_version_id"),
                "mapped": result.get("mapping_stats") is not None
            }
            print(f"    Checksum: {result.get('checksum', '')[:16]}...")
            print(f"    Mapped: {pdf_results[demo_opp_id]['mapped']}")
        else:
            print(f"    FAILED: {result}")
            return

    demo_checksum = pdf_results.get(demo_opp_id, {}).get("checksum")
    if not demo_checksum:
        print(f"ERROR: No checksum found for {demo_opp_id}")
        return

    # =========================================================================
    # STEP 6: Get form metadata
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"STEP 6: Get metadata for {demo_opp_id}")
    print("-" * 70)

    forms = [{"opp_id": demo_opp_id, "checksum": demo_checksum}]
    metadata_result = get_forms_metadata(base_url, forms=forms, version_id=latest_version, min_similarity=3)

    if not metadata_result or not metadata_result.get("success"):
        print("ERROR: Failed to get metadata")
        return

    form_data = metadata_result.get("forms", [{}])[0]
    questions = form_data.get("questions", [])

    print(f"  Total questions: {len(questions)}")
    print(f"  Canonical questions: {form_data.get('canonical_count', 0)}")

    # =========================================================================
    # STEP 7: Create answers programmatically
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 7: Create answers programmatically")
    print("-" * 70)

    # Build answers based on question types
    answers = {}
    sample_values = {
        "open ended": "Sample text answer",
        "date": "01/15/2026",
        "numeric": "12345",
        "multiple choice (choose one)": True,
        "multi-choice (choose all)": ["Option 1"],
        "agreement/authorization": True,
    }

    # Sample data for common canonical questions
    canonical_answers = {
        "cq_001": "John",           # First Name
        "cq_002": "Smith",          # Last Name
        "cq_003": "Robert",         # Middle Name
        "cq_004": "01/15/1985",     # DOB
        "cq_006": "123 Main Street", # Address
        "cq_007": "Los Angeles",    # City
        "cq_008": "90012",          # Zip
        "cq_010": "(213) 555-1234", # Phone Day
        "cq_011": "(213) 555-5678", # Phone Cell
        "cq_012": "john@email.com", # Email
        "cq_014": "Male",           # Gender
        "cq_015": "Asian",          # Ethnicity
        "cq_016": "Single",         # Marital Status
        "cq_060": "English",        # Language
    }

    for q in questions[:15]:  # Limit to first 15 questions for demo
        q_num = q.get("question_number")
        q_type = q.get("question_type", "open ended")
        canonical_id = q.get("canonical_question_id")

        if canonical_id and canonical_id in canonical_answers:
            # Use predefined canonical answer
            answers[canonical_id] = canonical_answers[canonical_id]
        elif q_num:
            # Use sample value based on type
            key = f"q_{q_num}"
            answers[key] = sample_values.get(q_type, "Sample answer")

    print(f"  Created {len(answers)} answers:")
    for key, value in list(answers.items())[:10]:
        print(f"    {key}: {value}")
    if len(answers) > 10:
        print(f"    ... and {len(answers) - 10} more")

    # Save answers to file
    answers_file = "answers_example.json"
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent=2)
    print(f"\n  Saved answers to: {answers_file}")

    # =========================================================================
    # STEP 8: Fill the form
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 8: Fill the form with answers")
    print("-" * 70)

    fill_result = fill_form(
        base_url,
        opp_id=demo_opp_id,
        checksum=demo_checksum,
        version_id=latest_version,
        answers=answers,
        options={"truncate_on_char_limit": True, "fail_on_missing_optional": False}
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("=" * 70)

    if fill_result and fill_result.get("success"):
        print(f"  SUCCESS!")
        print(f"  Output PDF: {fill_result.get('output_path')}")
        report = fill_result.get("validation_report", {})
        summary = report.get("summary", {})
        print(f"  Fields filled: {summary.get('fields_filled', 0)} / {summary.get('total_fields', 0)}")
        print(f"  Warnings: {summary.get('warnings', 0)}")
        print(f"  Errors: {summary.get('errors', 0)}")
    else:
        print(f"  Form fill returned: {fill_result}")

    print(f"\n  Answers file saved: {answers_file}")
    print(f"  You can modify {answers_file} and re-run the fill command:")
    print(f"  python src/client_sim.py --fill-form --opp-id {demo_opp_id} --checksum {demo_checksum} --version {latest_version} --answers-file {answers_file}")


def run_demo_tests(base_url: str):
    """Run demo tests with sample PDFs."""
    print("\n" + "="*60)
    print("FORM MAPPING API - DEMO TESTS")
    print("="*60)

    # 1. Health check
    print("\n[1/4] Health Check...")
    if not health_check(base_url):
        print("Server not available. Exiting.")
        sys.exit(1)

    # 2. Get canonical versions
    print("\n[2/4] Getting Canonical Versions...")
    versions = get_canonical_versions(base_url)
    if not versions or not versions.get("success"):
        print("WARNING: No canonical versions found. Some tests may fail.")

    # 3. Test PDF insert (new form)
    print("\n[3/4] Testing PDF Insert (sample form)...")
    test_url = "https://pacsla.org/wp-content/uploads/2021/10/Referral-Form.pdf"
    test_opp_id = "test_12345"

    result = insert_pdf(base_url, test_url, test_opp_id)
    if result and result.get("success"):
        print("SUCCESS: PDF inserted and mapped")
    else:
        print("WARNING: PDF insert may have failed")

    # 4. Test cached retrieval (same form again)
    print("\n[4/4] Testing Cached Retrieval (same form)...")
    result2 = insert_pdf(base_url, test_url, test_opp_id)
    if result2 and result2.get("from_cache"):
        print("SUCCESS: Returned cached result as expected")
    else:
        print("NOTE: Form was reprocessed (may be expected)")

    print("\n" + "="*60)
    print("DEMO TESTS COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Client simulation for Form Mapping API"
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help=f"Server URL (default: {DEFAULT_SERVER_URL})"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run health check only"
    )
    parser.add_argument(
        "--versions",
        action="store_true",
        help="List canonical versions only"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="PDF URL to insert"
    )
    parser.add_argument(
        "--file-path",
        type=str,
        help="Local PDF file path to insert (alternative to --url)"
    )
    parser.add_argument(
        "--opp-id",
        type=str,
        help="Opportunity ID for the PDF"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo tests with sample PDFs"
    )

    # Canonical creation arguments
    parser.add_argument(
        "--create-canonical",
        action="store_true",
        help="Start a new canonical creation run"
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["latest_first", "oldest_first", "random"],
        default="latest_first",
        help="Form processing order for canonical creation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model to use for canonical creation"
    )
    parser.add_argument(
        "--webhook",
        type=str,
        help="Webhook URL for canonical creation completion notification"
    )
    parser.add_argument(
        "--form-ids",
        type=str,
        nargs="+",
        help="Specific form IDs to use for canonical creation"
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll and wait for canonical creation to complete"
    )

    # Run status arguments
    parser.add_argument(
        "--run-status",
        type=str,
        metavar="RUN_ID",
        help="Get status of a canonical creation run"
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all canonical creation runs"
    )
    parser.add_argument(
        "--list-webhooks",
        action="store_true",
        help="List all received webhooks"
    )
    parser.add_argument(
        "--clear-webhooks",
        action="store_true",
        help="Clear all received webhooks"
    )

    # Get forms metadata arguments
    parser.add_argument(
        "--get-forms",
        action="store_true",
        help="Get form metadata with canonical overrides"
    )
    parser.add_argument(
        "--checksum",
        type=str,
        help="Checksum of the form (for --get-forms with single --opp-id)"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Canonical version ID (for --get-forms)"
    )
    parser.add_argument(
        "--min-similarity",
        type=int,
        default=3,
        help="Minimum similarity score 1-5 for canonical override (default: 3)"
    )
    parser.add_argument(
        "--forms-file",
        type=str,
        help="JSON file with forms array [{opp_id, checksum}, ...] (for --get-forms)"
    )

    # Fill form arguments
    parser.add_argument(
        "--fill-form",
        action="store_true",
        help="Fill a PDF form with answers"
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        help="JSON file with answers keyed by canonical_question_id or 'q_{number}'"
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        default=True,
        help="Truncate text that exceeds char limits (default: true)"
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Fail instead of truncating text that exceeds char limits"
    )

    # Run examples workflow
    parser.add_argument(
        "--run-examples",
        action="store_true",
        help="Run the complete workflow using example PDFs from server"
    )
    parser.add_argument(
        "--skip-add",
        action="store_true",
        help="Skip adding PDFs in --run-examples (if already added)"
    )

    args = parser.parse_args()
    base_url = args.server.rstrip("/")

    if args.health:
        health_check(base_url)
    elif args.versions:
        get_canonical_versions(base_url)
    elif args.get_forms:
        # Build forms list from arguments
        forms = []

        if args.forms_file:
            # Load forms from JSON file
            try:
                with open(args.forms_file, "r") as f:
                    forms = json.load(f)
                if not isinstance(forms, list):
                    print("ERROR: forms-file must contain a JSON array")
                    sys.exit(1)
            except FileNotFoundError:
                print(f"ERROR: File not found: {args.forms_file}")
                sys.exit(1)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON in forms-file: {e}")
                sys.exit(1)
        elif args.opp_id and args.checksum:
            # Single form from command line
            forms = [{"opp_id": args.opp_id, "checksum": args.checksum}]
        else:
            print("ERROR: --get-forms requires either (--opp-id and --checksum) or --forms-file")
            sys.exit(1)

        if not args.version:
            print("ERROR: --get-forms requires --version")
            sys.exit(1)

        get_forms_metadata(
            base_url,
            forms=forms,
            version_id=args.version,
            min_similarity=args.min_similarity
        )
    elif args.fill_form:
        # Fill a form with answers
        if not args.opp_id:
            print("ERROR: --fill-form requires --opp-id")
            sys.exit(1)
        if not args.checksum:
            print("ERROR: --fill-form requires --checksum")
            sys.exit(1)
        if not args.version:
            print("ERROR: --fill-form requires --version")
            sys.exit(1)
        if not args.answers_file:
            print("ERROR: --fill-form requires --answers-file")
            sys.exit(1)

        # Load answers from JSON file
        try:
            with open(args.answers_file, "r") as f:
                answers = json.load(f)
            if not isinstance(answers, dict):
                print("ERROR: answers-file must contain a JSON object (dict)")
                sys.exit(1)
        except FileNotFoundError:
            print(f"ERROR: File not found: {args.answers_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in answers-file: {e}")
            sys.exit(1)

        # Build options
        options = {
            "truncate_on_char_limit": not args.no_truncate,
            "fail_on_missing_optional": False
        }

        fill_form(
            base_url,
            opp_id=args.opp_id,
            checksum=args.checksum,
            version_id=args.version,
            answers=answers,
            options=options
        )
    elif (args.url or args.file_path) and args.opp_id:
        insert_pdf(base_url, url=args.url, opportunity_id=args.opp_id, file_path=args.file_path)
    elif args.url or args.file_path:
        print("ERROR: --url or --file-path requires --opp-id")
        sys.exit(1)
    elif args.create_canonical:
        # Start canonical creation
        webhook = args.webhook
        if not webhook and args.poll:
            # Use local webhook endpoint for testing
            webhook = f"{base_url}/api/webhook/canonical"

        result = create_canonical(
            base_url,
            form_ids=args.form_ids,
            order=args.order,
            model=args.model,
            webhook_url=webhook
        )

        if result and result.get("success") and args.poll:
            run_id = result.get("run_id")
            poll_run_status(base_url, run_id)
    elif args.run_status:
        get_run_status(base_url, args.run_status)
    elif args.list_runs:
        list_runs(base_url)
    elif args.list_webhooks:
        list_webhooks(base_url)
    elif args.clear_webhooks:
        clear_webhooks(base_url)
    elif args.run_examples:
        run_examples(base_url, skip_add=args.skip_add)
    else:
        # Default: run demo tests
        run_demo_tests(base_url)


if __name__ == "__main__":
    main()
