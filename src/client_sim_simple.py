"""
Simple API client for PDF form processing.
Copy-paste these cells into a notebook and run one at a time.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:5000"

# =============================================================================
# STEP 1: Add PDFs (run this cell multiple times with different files)
# =============================================================================

def add_pdf(file_path: str, opp_id: str):
    """Add a local PDF file to the system."""
    response = requests.post(
        f"{BASE_URL}/api/pdf",
        json={"file_path": file_path, "opportunity_id": opp_id},
        timeout=120
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

# Example usage - run these one at a time:
# add_pdf("C:/pp2/od_map/examples/opp1.pdf", "opp1")
# add_pdf("C:/pp2/od_map/examples/opp2.pdf", "opp2")
# add_pdf("C:/pp2/od_map/examples/opp3.pdf", "opp3")


# =============================================================================
# STEP 2: Create canonical questions (maps all existing forms)
# =============================================================================

def create_canonical():
    """Create canonical questions from all forms. Watch server logs for progress."""
    response = requests.post(
        f"{BASE_URL}/api/canonical/create",
        json={"webhook_url": f"{BASE_URL}/api/webhook/canonical"},
        timeout=300
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

# Run: create_canonical()
# Then watch server logs - it will print when webhook is called


# =============================================================================
# STEP 3: Check canonical version exists
# =============================================================================

def get_versions():
    """List canonical versions to get the version_id."""
    response = requests.get(f"{BASE_URL}/api/canonical/versions", timeout=10)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    # Return the latest version_id
    versions = result.get("versions", [])
    if versions:
        latest = [v for v in versions if v.get("is_latest")]
        return latest[0]["version_id"] if latest else versions[0]["version_id"]
    return None

# Run: version_id = get_versions()


# =============================================================================
# STEP 4: Add a new PDF (will auto-map to canonical)
# =============================================================================

# Use add_pdf() again - now it will auto-map to canonical
# add_pdf("C:/pp2/od_map/examples/opp4.pdf", "opp4")
# Save the checksum from the output!


# =============================================================================
# STEP 5: Get form metadata
# =============================================================================

def get_metadata(opp_id: str, checksum: str, version_id: str):
    """Get form metadata with canonical mappings."""
    response = requests.post(
        f"{BASE_URL}/api/forms/metadata",
        json={
            "forms": [{"opp_id": opp_id, "checksum": checksum}],
            "version_id": version_id,
            "min_similarity": 3
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

# Example: get_metadata("opp4", "YOUR_CHECKSUM_HERE", version_id)


# =============================================================================
# STEP 6: Fill the form
# =============================================================================

def fill_form(opp_id: str, checksum: str, version_id: str, answers: dict):
    """Fill a PDF form with answers and save to examples/filled/."""
    response = requests.post(
        f"{BASE_URL}/api/forms/fill",
        json={
            "opp_id": opp_id,
            "checksum": checksum,
            "version_id": version_id,
            "answers": answers,
            "options": {"truncate_on_char_limit": True}
        },
        timeout=300
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))

    # Copy filled PDF to examples/filled if successful
    if result.get("success") and result.get("output_path"):
        import shutil
        src = Path(result["output_path"])
        dest_dir = Path("C:/pp2/od_map/examples/filled")
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        print(f"\nCopied to: {dest}")

    return result

# Example answers - use canonical question IDs (cq_001) or form question IDs (fq_0)
# answers = {
#     "cq_001": "John",        # Canonical: First Name (if mapped)
#     "cq_002": "Smith",       # Canonical: Last Name (if mapped)
#     "fq_0": "01/15/1985",    # Form question 0 (by question_id from metadata)
#     "fq_1": "Some answer",   # Form question 1
# }
# fill_form("opp4", "YOUR_CHECKSUM_HERE", version_id, answers)


# =============================================================================
# QUICK REFERENCE - Run these in order:
# =============================================================================
#
# 1. Start server:  python src/server.py
#
# 2. Add PDFs:
#    add_pdf("C:/pp2/od_map/examples/opp1.pdf", "opp1")
#    add_pdf("C:/pp2/od_map/examples/opp2.pdf", "opp2")
#    add_pdf("C:/pp2/od_map/examples/opp3.pdf", "opp3")
#
# 3. Create canonical:
#    create_canonical()
#
# 4. Get version ID:
#    version_id = get_versions()
#
# 5. Add new PDF (save the checksum!):
#    result = add_pdf("C:/pp2/od_map/examples/opp1.pdf", "opp4")
#    checksum = result["checksum"]
#
# 6. Get metadata:
#    get_metadata("opp4", checksum, version_id)
#
# 7. Fill form:
#    answers = {"cq_001": "John", "cq_002": "Smith", "fq_0": "01/25/2026"}
#    fill_form("opp4", checksum, version_id, answers)
