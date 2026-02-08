"""
Migration script to move data from JSONL files to PostgreSQL database.

Usage:
    python migrate_to_db.py [--dry-run] [--forms-only] [--canonical-only]

Options:
    --dry-run       Show what would be migrated without making changes
    --forms-only    Only migrate form records
    --canonical-only Only migrate canonical versions
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from database import (
    init_db,
    test_connection,
    save_form_record_db,
    save_canonical_version_db,
    set_latest_canonical_version_db,
    load_form_records_db,
    list_canonical_versions_db,
    get_session,
    Od1Form,
    Od1CanonicalVersion
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("migrate")

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
FORM_JSONL_PATH = DATA_DIR / "form_metadata.jsonl"
CANONICAL_CACHE_PATH = DATA_DIR / "canonical_mappings.json"


def load_jsonl_forms() -> list:
    """Load form records from JSONL file."""
    records = []
    if not FORM_JSONL_PATH.exists():
        logger.warning(f"Form metadata file not found: {FORM_JSONL_PATH}")
        return records

    with open(FORM_JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")

    logger.info(f"Loaded {len(records)} form records from JSONL")
    return records


def load_canonical_versions() -> list:
    """Load canonical versions from cache file."""
    if not CANONICAL_CACHE_PATH.exists():
        logger.warning(f"Canonical cache file not found: {CANONICAL_CACHE_PATH}")
        return []

    with open(CANONICAL_CACHE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both {"versions": [...]} and direct list formats
    if isinstance(data, dict) and "versions" in data:
        versions = data["versions"]
    elif isinstance(data, list):
        versions = data
    else:
        versions = [data]

    logger.info(f"Loaded {len(versions)} canonical versions from cache")
    return versions


def migrate_forms(dry_run: bool = False) -> dict:
    """Migrate form records to database."""
    stats = {
        "total": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": 0
    }

    records = load_jsonl_forms()
    stats["total"] = len(records)

    if not records:
        logger.info("No form records to migrate")
        return stats

    for record in records:
        opp_id = record.get("opportunity_id", "unknown")
        checksum = record.get("pdf_checksum", "")[:12]

        if dry_run:
            logger.info(f"[DRY RUN] Would migrate form: {opp_id} ({checksum}...)")
            stats["migrated"] += 1
            continue

        try:
            save_form_record_db(record)
            stats["migrated"] += 1
        except Exception as e:
            logger.error(f"Failed to migrate form {opp_id}: {e}")
            stats["errors"] += 1

    return stats


def migrate_canonical_versions(dry_run: bool = False) -> dict:
    """Migrate canonical versions to database."""
    stats = {
        "total": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": 0
    }

    versions = load_canonical_versions()
    stats["total"] = len(versions)

    if not versions:
        logger.info("No canonical versions to migrate")
        return stats

    latest_version = None

    for version in versions:
        version_id = version.get("version_id", "unknown")

        if dry_run:
            logger.info(f"[DRY RUN] Would migrate canonical version: {version_id}")
            stats["migrated"] += 1
            continue

        try:
            save_canonical_version_db(version)
            stats["migrated"] += 1
            # Track latest by created_at
            if latest_version is None or version.get("created_at", "") > latest_version.get("created_at", ""):
                latest_version = version
        except Exception as e:
            logger.error(f"Failed to migrate version {version_id}: {e}")
            stats["errors"] += 1

    # Set latest version
    if latest_version and not dry_run:
        set_latest_canonical_version_db(latest_version.get("version_id"))
        logger.info(f"Set latest canonical version: {latest_version.get('version_id')}")

    return stats


def verify_migration() -> dict:
    """Verify migration by comparing counts."""
    jsonl_forms = load_jsonl_forms()
    jsonl_canonical = load_canonical_versions()

    db_forms = load_form_records_db()
    db_canonical = list_canonical_versions_db()

    return {
        "jsonl_forms": len(jsonl_forms),
        "db_forms": len(db_forms),
        "forms_match": len(jsonl_forms) == len(db_forms),
        "jsonl_canonical": len(jsonl_canonical),
        "db_canonical": len(db_canonical),
        "canonical_match": len(jsonl_canonical) == len(db_canonical)
    }


def main():
    parser = argparse.ArgumentParser(description="Migrate JSONL data to PostgreSQL database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    parser.add_argument("--forms-only", action="store_true", help="Only migrate form records")
    parser.add_argument("--canonical-only", action="store_true", help="Only migrate canonical versions")
    parser.add_argument("--verify", action="store_true", help="Verify migration by comparing counts")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║         JSONL to PostgreSQL Migration Script                 ║
║         Tables prefixed with: od1_                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Test database connection
    print("Testing database connection...")
    if not test_connection():
        print("ERROR: Could not connect to database. Check DATABASE_URL in .env")
        sys.exit(1)
    print("Database connection successful!\n")

    # Initialize tables
    print("Initializing database tables...")
    init_db()
    print("Tables created/verified!\n")

    if args.verify:
        print("Verifying migration...")
        verification = verify_migration()
        print(f"""
Verification Results:
---------------------
JSONL Forms:      {verification['jsonl_forms']}
Database Forms:   {verification['db_forms']}
Forms Match:      {'✓' if verification['forms_match'] else '✗'}

JSONL Canonical:  {verification['jsonl_canonical']}
Database Canonical: {verification['db_canonical']}
Canonical Match:  {'✓' if verification['canonical_match'] else '✗'}
        """)
        return

    if args.dry_run:
        print("=== DRY RUN MODE - No changes will be made ===\n")

    # Migrate forms
    if not args.canonical_only:
        print("Migrating form records...")
        form_stats = migrate_forms(dry_run=args.dry_run)
        print(f"""
Form Migration Results:
-----------------------
Total records:    {form_stats['total']}
Migrated:         {form_stats['migrated']}
Skipped:          {form_stats['skipped']}
Errors:           {form_stats['errors']}
        """)

    # Migrate canonical versions
    if not args.forms_only:
        print("Migrating canonical versions...")
        canonical_stats = migrate_canonical_versions(dry_run=args.dry_run)
        print(f"""
Canonical Version Migration Results:
------------------------------------
Total versions:   {canonical_stats['total']}
Migrated:         {canonical_stats['migrated']}
Skipped:          {canonical_stats['skipped']}
Errors:           {canonical_stats['errors']}
        """)

    if not args.dry_run:
        print("\n=== Migration Complete ===")
        print("Run with --verify to confirm data was migrated correctly")
    else:
        print("\n=== Dry Run Complete ===")
        print("Run without --dry-run to perform actual migration")


if __name__ == "__main__":
    main()
