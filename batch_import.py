"""
Batch PDF Import Script.

Reads PDFs from a CSV file and processes them in parallel via the API server.
Writes back results to the CSV with proper file locking.

Usage:
    python batch_import.py [options]

Options:
    --csv PATH          Path to CSV file (default: src/data/pdf_import.csv)
    --workers N         Number of parallel workers (default: 4)
    --api-url URL       API server URL (default: http://localhost:5000)
    --dry-run           Show what would be processed without actually processing

CSV Format:
    Required columns: access_value (PDF URL), oppid (opportunity ID)
    Added columns: processed_at, status, checksum, error
"""

import argparse
import csv
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s"
)
logger = logging.getLogger(__name__)

# File lock for CSV writing
csv_lock = threading.Lock()

# Default paths
DEFAULT_CSV_PATH = Path(__file__).parent / "data" / "pdf_import.csv"
DEFAULT_API_URL = "http://localhost:5000"


def check_api_health(api_url: str) -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{api_url}/api/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def process_single_pdf(
    row: Dict[str, str],
    row_index: int,
    api_url: str
) -> Dict[str, Any]:
    """
    Process a single PDF via the API.

    Args:
        row: CSV row data
        row_index: Index of the row in the CSV (for logging)
        api_url: API server URL

    Returns:
        Result dict with status, checksum, error, etc.
    """
    url = row.get("access_value", "").strip()
    opp_id = row.get("oppid", "").strip()

    result = {
        "row_index": row_index,
        "oppid": opp_id,
        "url": url,
        "processed_at": datetime.now().isoformat(),
        "status": "pending",
        "checksum": "",
        "canonical_version_id": "",
        "error": ""
    }

    # Validate inputs - skip empty rows silently
    if not url and not opp_id:
        result["status"] = "skipped"
        result["error"] = "Empty row"
        return result  # Silent skip for empty rows

    if not url:
        result["status"] = "skipped"
        result["error"] = "Missing access_value (URL)"
        logger.warning(f"Row {row_index}: Skipped - missing URL")
        return result

    if not opp_id:
        result["status"] = "skipped"
        result["error"] = "Missing oppid"
        logger.warning(f"Row {row_index}: Skipped - missing oppid")
        return result

    # Check if already processed
    if row.get("status") == "success":
        result["status"] = "skipped"
        result["error"] = "Already processed"
        result["checksum"] = row.get("checksum", "")
        result["canonical_version_id"] = row.get("canonical_version_id", "")
        logger.info(f"Row {row_index}: Skipped - already processed (opp_id={opp_id})")
        return result

    # Call the API
    try:
        logger.info(f"Row {row_index}: Processing opp_id={opp_id}, url={url[:60]}...")

        response = requests.post(
            f"{api_url}/api/pdf",
            json={
                "url": url,
                "opportunity_id": opp_id
            },
            timeout=120  # 2 minute timeout for PDF processing
        )

        data = response.json()

        if response.status_code == 200 and data.get("success"):
            result["status"] = "success"
            result["checksum"] = data.get("checksum", "")
            result["canonical_version_id"] = data.get("canonical_version_id", "")
            logger.info(f"Row {row_index}: Success - opp_id={opp_id}, checksum={result['checksum'][:12]}...")
        else:
            result["status"] = "failed"
            result["error"] = data.get("error", f"HTTP {response.status_code}")
            logger.error(f"Row {row_index}: Failed - opp_id={opp_id}, error={result['error']}")

    except requests.Timeout:
        result["status"] = "failed"
        result["error"] = "Request timeout (>120s)"
        logger.error(f"Row {row_index}: Timeout - opp_id={opp_id}")
    except requests.RequestException as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Row {row_index}: Request error - opp_id={opp_id}, error={e}")
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Row {row_index}: Unexpected error - opp_id={opp_id}, error={e}")

    return result


def update_csv_row(
    csv_path: Path,
    row_index: int,
    result: Dict[str, Any],
    fieldnames: List[str]
) -> None:
    """
    Update a single row in the CSV file with thread-safe locking.

    Args:
        csv_path: Path to CSV file
        row_index: Index of the row to update
        result: Result data to write
        fieldnames: CSV column names
    """
    with csv_lock:
        # Read all rows
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Update the specific row
        if 0 <= row_index < len(rows):
            rows[row_index]["processed_at"] = result["processed_at"]
            rows[row_index]["status"] = result["status"]
            rows[row_index]["checksum"] = result["checksum"]
            rows[row_index]["canonical_version_id"] = result["canonical_version_id"]
            rows[row_index]["error"] = result["error"]

        # Write all rows back
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def ensure_csv_columns(csv_path: Path) -> List[str]:
    """
    Ensure the CSV has the required output columns.
    Returns the complete list of fieldnames.
    """
    required_output_columns = ["processed_at", "status", "checksum", "canonical_version_id", "error"]

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Add missing columns
    fieldnames = list(existing_fieldnames)
    for col in required_output_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    # If we added columns, rewrite the file
    if fieldnames != list(existing_fieldnames):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Ensure all columns exist in each row
                for col in required_output_columns:
                    if col not in row:
                        row[col] = ""
                writer.writerow(row)
        logger.info(f"Added output columns to CSV: {required_output_columns}")

    return fieldnames


def load_csv_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Load all rows from the CSV file."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def run_batch_import(
    csv_path: Path,
    api_url: str,
    workers: int,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Run the batch import process.

    Args:
        csv_path: Path to CSV file
        api_url: API server URL
        workers: Number of parallel workers
        dry_run: If True, only show what would be processed

    Returns:
        Stats dict with counts of success, failed, skipped
    """
    # Validate CSV exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.info("Creating sample CSV file...")
        create_sample_csv(csv_path)
        logger.info(f"Sample CSV created at: {csv_path}")
        logger.info("Please populate it with your PDF data and run again.")
        return {"success": 0, "failed": 0, "skipped": 0, "total": 0}

    # Check API health
    if not dry_run:
        if not check_api_health(api_url):
            logger.error(f"API server not reachable at {api_url}")
            logger.error("Please start the server with: python src/server.py")
            return {"success": 0, "failed": 0, "skipped": 0, "total": 0}
        logger.info(f"API server is healthy at {api_url}")

    # Ensure CSV has output columns
    fieldnames = ensure_csv_columns(csv_path)

    # Load rows
    rows, _ = load_csv_rows(csv_path)
    total_rows = len(rows)

    # Filter to only rows with data (non-empty access_value and oppid)
    rows_with_data = [
        (i, row) for i, row in enumerate(rows)
        if row.get("access_value", "").strip() and row.get("oppid", "").strip()
    ]
    total = len(rows_with_data)
    empty_rows = total_rows - total

    if total == 0:
        logger.warning("CSV file has no rows with data")
        return {"success": 0, "failed": 0, "skipped": 0, "total": 0}

    logger.info(f"Found {total} PDFs to process with {workers} workers (skipping {empty_rows} empty rows)")

    # Dry run - just show what would be processed
    if dry_run:
        pending = 0
        already_done = 0
        for i, row in rows_with_data:
            status = row.get("status", "")
            if status == "success":
                already_done += 1
            else:
                pending += 1
                logger.info(f"  Would process: row {i}, oppid={row.get('oppid')}, url={row.get('access_value', '')[:50]}...")

        logger.info(f"Dry run complete: {pending} pending, {already_done} already processed")
        return {"success": 0, "failed": 0, "skipped": already_done, "total": total}

    # Process in parallel
    stats = {"success": 0, "failed": 0, "skipped": 0, "total": total}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="Worker") as executor:
        # Submit all tasks (rows_with_data contains (original_index, row) tuples)
        futures = {
            executor.submit(process_single_pdf, row, row_index, api_url): row_index
            for row_index, row in rows_with_data
        }

        # Process results as they complete
        for future in as_completed(futures):
            row_index = futures[future]
            try:
                result = future.result()

                # Update CSV immediately after each result
                update_csv_row(csv_path, row_index, result, fieldnames)

                # Update stats
                status = result["status"]
                if status == "success":
                    stats["success"] += 1
                elif status == "failed":
                    stats["failed"] += 1
                else:
                    stats["skipped"] += 1

            except Exception as e:
                logger.error(f"Row {row_index}: Worker exception - {e}")
                stats["failed"] += 1

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Batch import complete!")
    logger.info(f"  Total:   {stats['total']}")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Failed:  {stats['failed']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Time:    {elapsed:.1f}s")
    logger.info("=" * 60)

    return stats


def create_sample_csv(csv_path: Path) -> None:
    """Create a sample CSV file with example structure."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "access_value",
            "oppid",
            "processed_at",
            "status",
            "checksum",
            "canonical_version_id",
            "error"
        ])
        # Add example row (commented out)
        writer.writerow([
            "https://example.com/form.pdf",
            "12345",
            "",
            "",
            "",
            "",
            ""
        ])

    logger.info(f"Created sample CSV at: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch import PDFs from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process with default settings (4 workers)
    python batch_import.py

    # Process with 8 parallel workers
    python batch_import.py --workers 8

    # Use custom CSV file
    python batch_import.py --csv /path/to/my_pdfs.csv

    # Dry run to see what would be processed
    python batch_import.py --dry-run

    # Connect to API on different port
    python batch_import.py --api-url http://localhost:8000
        """
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to CSV file (default: {DEFAULT_CSV_PATH})"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )

    args = parser.parse_args()

    # Validate workers
    if args.workers < 1:
        logger.error("Workers must be at least 1")
        sys.exit(1)

    if args.workers > 20:
        logger.warning(f"Using {args.workers} workers - this may overwhelm the API server")

    # Run the batch import
    stats = run_batch_import(
        csv_path=args.csv,
        api_url=args.api_url,
        workers=args.workers,
        dry_run=args.dry_run
    )

    # Exit with error code if any failures
    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
