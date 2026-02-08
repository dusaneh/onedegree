"""
Migration script to upload existing local PDFs to S3.

This script:
1. Queries forms where pdf_s3_key is NULL but pdf_filepath exists
2. Uploads each PDF to S3
3. Updates the database with the S3 key
4. Optionally migrates filled PDFs from data/filled/

Usage:
    python migrate_pdfs_to_s3.py --dry-run     # Preview what would be migrated
    python migrate_pdfs_to_s3.py               # Perform actual migration
    python migrate_pdfs_to_s3.py --verify      # Verify all PDFs are accessible in S3
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from database import get_session, Od1Form, init_db
from s3_storage import (
    upload_pdf,
    file_exists,
    list_pdfs,
    test_connection,
    get_s3_key,
)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("migrate_pdfs_to_s3")

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
FILLED_DIR = DATA_DIR / "filled"


def get_forms_needing_migration() -> List[Tuple[int, str, str, str]]:
    """
    Get list of forms that need S3 migration.

    Returns:
        List of (form_id, opportunity_id, pdf_filepath, pdf_filename) tuples
    """
    session = get_session()
    try:
        # Find forms with local path but no S3 key
        forms = session.query(
            Od1Form.id,
            Od1Form.opportunity_id,
            Od1Form.pdf_filepath,
            Od1Form.pdf_filename
        ).filter(
            Od1Form.pdf_s3_key.is_(None),
            Od1Form.pdf_filepath.isnot(None)
        ).all()

        return [(f.id, f.opportunity_id, f.pdf_filepath, f.pdf_filename) for f in forms]
    finally:
        session.close()


def update_form_s3_key(form_id: int, s3_key: str) -> bool:
    """
    Update a form's S3 key in the database.

    Args:
        form_id: Database ID of the form
        s3_key: S3 key to set

    Returns:
        True if updated successfully
    """
    session = get_session()
    try:
        form = session.query(Od1Form).filter_by(id=form_id).first()
        if form:
            form.pdf_s3_key = s3_key
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to update form {form_id}: {e}")
        return False
    finally:
        session.close()


def get_local_filled_pdfs() -> List[Path]:
    """Get list of filled PDFs in local directory."""
    if not FILLED_DIR.exists():
        return []
    return list(FILLED_DIR.glob("*.pdf"))


def migrate_forms(dry_run: bool = True) -> Dict[str, int]:
    """
    Migrate form PDFs to S3.

    Args:
        dry_run: If True, only preview what would be migrated

    Returns:
        Stats dict with counts
    """
    stats = {
        "total": 0,
        "success": 0,
        "skipped_not_found": 0,
        "skipped_already_exists": 0,
        "failed": 0
    }

    forms = get_forms_needing_migration()
    stats["total"] = len(forms)

    logger.info(f"Found {len(forms)} forms needing S3 migration")

    for form_id, opp_id, filepath, filename in forms:
        if not filepath or not os.path.exists(filepath):
            logger.warning(f"Skipping {opp_id}: file not found at {filepath}")
            stats["skipped_not_found"] += 1
            continue

        # Generate S3 key
        s3_key = get_s3_key("pdf", filename or Path(filepath).name)

        # Check if already in S3
        if file_exists(s3_key):
            logger.info(f"Already in S3: {opp_id} -> {s3_key}")
            if not dry_run:
                update_form_s3_key(form_id, s3_key)
            stats["skipped_already_exists"] += 1
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would upload: {filepath} -> {s3_key}")
            stats["success"] += 1
        else:
            try:
                uploaded_key = upload_pdf(filepath, "pdf", filename)
                update_form_s3_key(form_id, uploaded_key)
                logger.info(f"Uploaded: {opp_id} -> {uploaded_key}")
                stats["success"] += 1
            except Exception as e:
                logger.error(f"Failed to upload {opp_id}: {e}")
                stats["failed"] += 1

    return stats


def migrate_filled_pdfs(dry_run: bool = True) -> Dict[str, int]:
    """
    Migrate filled PDFs to S3.

    Args:
        dry_run: If True, only preview what would be migrated

    Returns:
        Stats dict with counts
    """
    stats = {
        "total": 0,
        "success": 0,
        "skipped_already_exists": 0,
        "failed": 0
    }

    filled_pdfs = get_local_filled_pdfs()
    stats["total"] = len(filled_pdfs)

    logger.info(f"Found {len(filled_pdfs)} filled PDFs to migrate")

    for pdf_path in filled_pdfs:
        s3_key = get_s3_key("filled", pdf_path.name)

        # Check if already in S3
        if file_exists(s3_key):
            logger.info(f"Already in S3: {pdf_path.name}")
            stats["skipped_already_exists"] += 1
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would upload: {pdf_path} -> {s3_key}")
            stats["success"] += 1
        else:
            try:
                upload_pdf(str(pdf_path), "filled", pdf_path.name)
                logger.info(f"Uploaded filled: {pdf_path.name}")
                stats["success"] += 1
            except Exception as e:
                logger.error(f"Failed to upload {pdf_path.name}: {e}")
                stats["failed"] += 1

    return stats


def verify_s3_migration() -> Dict[str, int]:
    """
    Verify that all forms with S3 keys are accessible.

    Returns:
        Stats dict with counts
    """
    stats = {
        "total": 0,
        "accessible": 0,
        "missing": 0
    }

    session = get_session()
    try:
        # Get forms with S3 keys
        forms = session.query(
            Od1Form.id,
            Od1Form.opportunity_id,
            Od1Form.pdf_s3_key
        ).filter(
            Od1Form.pdf_s3_key.isnot(None)
        ).all()

        stats["total"] = len(forms)
        logger.info(f"Verifying {len(forms)} forms with S3 keys...")

        for form_id, opp_id, s3_key in forms:
            if file_exists(s3_key):
                stats["accessible"] += 1
            else:
                logger.warning(f"MISSING: {opp_id} -> {s3_key}")
                stats["missing"] += 1

        return stats
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate local PDFs to S3"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be migrated without actually uploading"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all PDFs with S3 keys are accessible"
    )
    parser.add_argument(
        "--filled-only",
        action="store_true",
        help="Only migrate filled PDFs, not source forms"
    )
    parser.add_argument(
        "--forms-only",
        action="store_true",
        help="Only migrate source form PDFs, not filled PDFs"
    )

    args = parser.parse_args()

    # Test S3 connection first
    logger.info("Testing S3 connection...")
    if not test_connection():
        logger.error("S3 connection failed. Check your BUCKETEER_* environment variables.")
        sys.exit(1)
    logger.info("S3 connection successful")

    # Initialize database
    init_db()

    if args.verify:
        logger.info("=== Verifying S3 Migration ===")
        stats = verify_s3_migration()
        logger.info(f"Verification complete:")
        logger.info(f"  Total forms with S3 keys: {stats['total']}")
        logger.info(f"  Accessible: {stats['accessible']}")
        logger.info(f"  Missing: {stats['missing']}")

        if stats["missing"] > 0:
            logger.warning("Some PDFs are missing from S3!")
            sys.exit(1)
        else:
            logger.info("All PDFs verified successfully!")
        return

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
    else:
        logger.info("=== MIGRATION MODE ===")

    total_stats = {
        "forms": {"total": 0, "success": 0, "failed": 0},
        "filled": {"total": 0, "success": 0, "failed": 0}
    }

    # Migrate form PDFs
    if not args.filled_only:
        logger.info("\n--- Migrating Form PDFs ---")
        form_stats = migrate_forms(dry_run=args.dry_run)
        total_stats["forms"] = form_stats
        logger.info(f"Form migration: {form_stats['success']}/{form_stats['total']} "
                   f"(skipped: {form_stats['skipped_not_found'] + form_stats['skipped_already_exists']}, "
                   f"failed: {form_stats['failed']})")

    # Migrate filled PDFs
    if not args.forms_only:
        logger.info("\n--- Migrating Filled PDFs ---")
        filled_stats = migrate_filled_pdfs(dry_run=args.dry_run)
        total_stats["filled"] = filled_stats
        logger.info(f"Filled migration: {filled_stats['success']}/{filled_stats['total']} "
                   f"(skipped: {filled_stats['skipped_already_exists']}, "
                   f"failed: {filled_stats['failed']})")

    # Summary
    logger.info("\n=== Migration Summary ===")
    if not args.filled_only:
        logger.info(f"Forms: {total_stats['forms']['success']} uploaded, "
                   f"{total_stats['forms'].get('skipped_not_found', 0)} not found, "
                   f"{total_stats['forms'].get('skipped_already_exists', 0)} already in S3, "
                   f"{total_stats['forms']['failed']} failed")
    if not args.forms_only:
        logger.info(f"Filled: {total_stats['filled']['success']} uploaded, "
                   f"{total_stats['filled'].get('skipped_already_exists', 0)} already in S3, "
                   f"{total_stats['filled']['failed']} failed")

    if args.dry_run:
        logger.info("\nThis was a dry run. Run without --dry-run to perform actual migration.")

    # Exit with error if any failures
    total_failed = total_stats["forms"].get("failed", 0) + total_stats["filled"].get("failed", 0)
    if total_failed > 0:
        logger.warning(f"\n{total_failed} files failed to migrate!")
        sys.exit(1)


if __name__ == "__main__":
    main()
