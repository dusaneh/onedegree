"""
PDF download client for fetching PDFs from URLs.
Self-contained module with no external dependencies outside standard library and requests.
"""

import os
import logging
import requests
import hashlib
import random
import string
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

# Logger Setup
logger = logging.getLogger("client_fx")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_DIR = DATA_DIR / "pdfs"
CHUNK_SIZE = 8192
REQUEST_TIMEOUT = 30
UNIQUE_ID_LENGTH = 6  # Length of unique ID suffix


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DownloadItem:
    """Represents a PDF to download."""
    url: str
    opportunity_id: str
    filename: Optional[str] = None


@dataclass
class DownloadResult:
    """Result of a PDF download attempt."""
    url: str
    opportunity_id: str
    success: bool
    filepath: Optional[str] = None
    filename: Optional[str] = None  # Just the filename without path
    s3_key: Optional[str] = None  # S3 key if uploaded to S3
    error: Optional[str] = None


# ============================================================================
# Unique ID Generation
# ============================================================================

def _get_existing_ids(output_dir: Path) -> Set[str]:
    """
    Get set of existing unique IDs from filenames in the output directory.

    Expected format: {opp_id}-{unique_id}.pdf

    Args:
        output_dir: Directory to scan for existing PDFs

    Returns:
        Set of unique IDs currently in use
    """
    existing_ids = set()

    if not output_dir.exists():
        return existing_ids

    for file in output_dir.iterdir():
        if file.is_file() and file.suffix.lower() == ".pdf":
            # Parse filename: {opp_id}-{unique_id}.pdf
            stem = file.stem
            if "-" in stem:
                parts = stem.rsplit("-", 1)
                if len(parts) == 2:
                    existing_ids.add(parts[1])

    return existing_ids


def _generate_unique_id(existing_ids: Set[str], length: int = UNIQUE_ID_LENGTH) -> str:
    """
    Generate a unique alphanumeric ID that doesn't exist in the set.

    Args:
        existing_ids: Set of IDs already in use
        length: Length of the ID to generate

    Returns:
        A unique alphanumeric ID
    """
    chars = string.ascii_lowercase + string.digits
    max_attempts = 1000

    for _ in range(max_attempts):
        new_id = ''.join(random.choices(chars, k=length))
        if new_id not in existing_ids:
            return new_id

    # Fallback: use longer ID if too many collisions
    return ''.join(random.choices(chars, k=length + 4))


def generate_unique_filename(opp_id: str, output_dir: Path = DEFAULT_PDF_DIR) -> str:
    """
    Generate a unique filename for a PDF that won't conflict with existing files.

    Format: {opp_id}-{unique_id}.pdf

    Args:
        opp_id: Opportunity ID
        output_dir: Directory where PDFs are stored

    Returns:
        Unique filename string
    """
    output_dir = Path(output_dir)
    existing_ids = _get_existing_ids(output_dir)
    unique_id = _generate_unique_id(existing_ids)

    return f"{opp_id}-{unique_id}.pdf"


# ============================================================================
# Core Functions
# ============================================================================

def validate_pdf_url(url: str) -> Tuple[bool, str, str]:
    """
    Validate that a URL points to a PDF by checking Content-Type.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, content_type, error_message)
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type:
            return True, content_type, ""
        else:
            return False, content_type, f"Content-Type is not PDF: {content_type}"

    except requests.RequestException as e:
        return False, "", f"Request failed: {e}"


def _normalize_url(url: str) -> str:
    """Ensure URL has a scheme."""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def download_pdf(
    url: str,
    opp_id: str,
    output_dir: Path = DEFAULT_PDF_DIR,
    filename: Optional[str] = None,
    validate_first: bool = True
) -> DownloadResult:
    """
    Download a PDF from a URL and upload to S3.

    Downloads to a temp file, uploads to S3, then cleans up local file.
    S3 is the only storage - local files are not kept.

    Args:
        url: URL to download from
        opp_id: Opportunity ID associated with this PDF
        output_dir: Directory for temp file during download
        filename: Optional filename override (if None, generates unique filename)
        validate_first: Whether to validate Content-Type before downloading

    Returns:
        DownloadResult with success status, s3_key, or error
    """
    import tempfile

    url = _normalize_url(url)

    # Validate URL first if requested
    if validate_first:
        is_valid, content_type, error = validate_pdf_url(url)
        if not is_valid:
            logger.warning(f"URL validation failed for {url}: {error}")
            # Continue anyway - some servers don't report correct Content-Type

    # Generate unique filename if not provided
    if not filename:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = generate_unique_filename(opp_id, output_dir)

    # Use temp file for download
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")

    try:
        logger.info(f"Downloading {url} to temp file")

        response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        # Write file in chunks
        with os.fdopen(temp_fd, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        logger.info(f"Successfully downloaded to temp file")

        # Upload to S3 (required)
        try:
            from s3_storage import upload_pdf as s3_upload
            s3_key = s3_upload(temp_path, "pdf", filename)
            logger.info(f"Uploaded to S3: {s3_key}")
        except Exception as e:
            error_msg = f"S3 upload failed: {e}"
            logger.error(error_msg)
            return DownloadResult(
                url=url,
                opportunity_id=opp_id,
                success=False,
                error=error_msg
            )

        return DownloadResult(
            url=url,
            opportunity_id=opp_id,
            success=True,
            filepath=None,  # No local file kept
            filename=filename,
            s3_key=s3_key
        )

    except requests.RequestException as e:
        error_msg = f"Download failed: {e}"
        logger.error(f"{error_msg} for {url}")
        return DownloadResult(
            url=url,
            opportunity_id=opp_id,
            success=False,
            error=error_msg
        )
    except IOError as e:
        error_msg = f"File write failed: {e}"
        logger.error(f"{error_msg}")
        return DownloadResult(
            url=url,
            opportunity_id=opp_id,
            success=False,
            error=error_msg
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def download_pdfs(
    items: List[DownloadItem],
    output_dir: Path = DEFAULT_PDF_DIR,
    skip_existing_opp_ids: bool = False
) -> List[DownloadResult]:
    """
    Download multiple PDFs with unique filenames.

    Args:
        items: List of DownloadItem objects
        output_dir: Directory to save PDFs
        skip_existing_opp_ids: If True, skip downloads where opp_id already has a file

    Returns:
        List of DownloadResult objects
    """
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build map of existing opp_ids to files if skipping
    existing_opp_files = {}
    if skip_existing_opp_ids:
        for file in output_dir.iterdir():
            if file.is_file() and file.suffix.lower() == ".pdf":
                stem = file.stem
                if "-" in stem:
                    opp_id_part = stem.rsplit("-", 1)[0]
                    existing_opp_files[opp_id_part] = file

    for item in items:
        # Skip if opp_id already has a file
        if skip_existing_opp_ids and item.opportunity_id in existing_opp_files:
            existing_file = existing_opp_files[item.opportunity_id]
            logger.info(f"Skipping existing file for opp_id {item.opportunity_id}: {existing_file}")
            results.append(DownloadResult(
                url=item.url,
                opportunity_id=item.opportunity_id,
                success=True,
                filepath=str(existing_file),
                filename=existing_file.name
            ))
            continue

        # Download with unique filename
        result = download_pdf(
            url=item.url,
            opp_id=item.opportunity_id,
            output_dir=output_dir,
            filename=item.filename,  # Will generate unique if None
            validate_first=True
        )
        results.append(result)

    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Download complete: {successful}/{len(items)} successful")

    return results


def get_pdf_files(directory_path: str = None) -> List[str]:
    """
    List all PDF files in a directory.

    Args:
        directory_path: Path to directory (defaults to data/pdfs)

    Returns:
        List of absolute paths to PDF files
    """
    if directory_path is None:
        directory = DEFAULT_PDF_DIR
    else:
        directory = Path(directory_path)

    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    pdf_files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() == ".pdf":
            pdf_files.append(str(file.absolute()))

    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return sorted(pdf_files)


def get_opp_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract opportunity ID from a filename.

    Expected format: {opp_id}-{unique_id}.pdf

    Args:
        filename: The filename (with or without path)

    Returns:
        Opportunity ID or None if not parseable
    """
    stem = Path(filename).stem
    if "-" in stem:
        return stem.rsplit("-", 1)[0]
    return None


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python client_fx.py <url> <opportunity_id>")
        print("Downloads PDF from URL and uploads to S3.")
        sys.exit(1)

    url = sys.argv[1]
    opp_id = sys.argv[2]

    result = download_pdf(url, opp_id)
    if result.success:
        print(f"S3 Key: {result.s3_key}")
        print(f"Filename: {result.filename}")
    else:
        print(f"Failed: {result.error}")
        sys.exit(1)
