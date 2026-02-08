"""
S3 Storage Module for PDF files.
Uses AWS S3 via Heroku Bucketeer for storing source and filled PDFs.
All keys use 'od1_' prefix since the bucket is shared with other apps.
"""

import os
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Logger setup
logger = logging.getLogger("s3_storage")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# S3 Configuration from Bucketeer
AWS_ACCESS_KEY_ID = os.environ.get("BUCKETEER_AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("BUCKETEER_AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.environ.get("BUCKETEER_BUCKET_NAME")
AWS_REGION = os.environ.get("BUCKETEER_AWS_REGION", "us-east-1")

# S3 key prefixes
PREFIX = "od1_"
PDF_PREFIX = f"{PREFIX}/pdfs/"
FILLED_PREFIX = f"{PREFIX}/filled/"

# Lazy-initialized S3 client
_s3_client = None


def get_s3_client():
    """
    Get or create boto3 S3 client.
    Uses lazy initialization for efficiency.

    Returns:
        boto3.client: S3 client instance

    Raises:
        ValueError: If required environment variables are not set
    """
    global _s3_client

    if _s3_client is None:
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
            raise ValueError(
                "Missing required S3 environment variables. "
                "Ensure BUCKETEER_AWS_ACCESS_KEY_ID, BUCKETEER_AWS_SECRET_ACCESS_KEY, "
                "and BUCKETEER_BUCKET_NAME are set."
            )

        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        logger.info(f"S3 client initialized for bucket: {BUCKET_NAME}")

    return _s3_client


def get_s3_key(key_type: str, filename: str) -> str:
    """
    Generate S3 key for a file.

    Args:
        key_type: Type of PDF - 'pdf' for source PDFs, 'filled' for filled PDFs
        filename: The filename (without path)

    Returns:
        Full S3 key with prefix
    """
    if key_type == "pdf":
        return f"{PDF_PREFIX}{filename}"
    elif key_type == "filled":
        return f"{FILLED_PREFIX}{filename}"
    else:
        raise ValueError(f"Invalid key_type: {key_type}. Must be 'pdf' or 'filled'.")


def upload_pdf(local_path: str, key_type: str, filename: Optional[str] = None) -> str:
    """
    Upload a PDF file to S3.

    Args:
        local_path: Path to the local PDF file
        key_type: Type of PDF - 'pdf' for source PDFs, 'filled' for filled PDFs
        filename: Optional filename to use in S3 (defaults to local filename)

    Returns:
        S3 key of the uploaded file

    Raises:
        FileNotFoundError: If local file doesn't exist
        ClientError: If S3 upload fails
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    if filename is None:
        filename = local_path.name

    s3_key = get_s3_key(key_type, filename)
    client = get_s3_client()

    logger.info(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}")

    client.upload_file(
        str(local_path),
        BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "application/pdf"}
    )

    logger.info(f"Successfully uploaded to S3: {s3_key}")
    return s3_key


def download_to_temp(s3_key: str) -> str:
    """
    Download a file from S3 to a temporary file.

    Args:
        s3_key: S3 key of the file to download

    Returns:
        Path to the temporary file (caller is responsible for cleanup)

    Raises:
        ClientError: If S3 download fails
    """
    client = get_s3_client()

    # Create temp file with .pdf extension
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    logger.info(f"Downloading s3://{BUCKET_NAME}/{s3_key} to {temp_path}")

    try:
        client.download_file(BUCKET_NAME, s3_key, temp_path)
        logger.info(f"Successfully downloaded from S3: {s3_key}")
        return temp_path
    except ClientError as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


@contextmanager
def temp_pdf_from_s3(s3_key: str):
    """
    Context manager to download a PDF from S3 to a temp file.
    Automatically cleans up the temp file when done.

    Args:
        s3_key: S3 key of the file to download

    Yields:
        Path to the temporary PDF file

    Example:
        with temp_pdf_from_s3("od1_/pdfs/form.pdf") as temp_path:
            # Use temp_path - file is automatically deleted after
            process_pdf(temp_path)
    """
    temp_path = download_to_temp(s3_key)
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug(f"Cleaned up temp file: {temp_path}")


def generate_presigned_url(s3_key: str, expires_in: int = 3600) -> str:
    """
    Generate a presigned URL for downloading a file from S3.

    Args:
        s3_key: S3 key of the file
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Presigned URL for downloading the file
    """
    client = get_s3_client()

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": s3_key},
        ExpiresIn=expires_in
    )

    logger.info(f"Generated presigned URL for {s3_key} (expires in {expires_in}s)")
    return url


def delete_pdf(s3_key: str) -> bool:
    """
    Delete a file from S3.

    Args:
        s3_key: S3 key of the file to delete

    Returns:
        True if deleted successfully, False if file didn't exist
    """
    client = get_s3_client()

    try:
        client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Deleted from S3: {s3_key}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning(f"File not found in S3: {s3_key}")
            return False
        raise


def file_exists(s3_key: str) -> bool:
    """
    Check if a file exists in S3.

    Args:
        s3_key: S3 key of the file to check

    Returns:
        True if file exists, False otherwise
    """
    client = get_s3_client()

    try:
        client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def list_pdfs(key_type: str = "pdf") -> list:
    """
    List all PDFs in the specified prefix.

    Args:
        key_type: 'pdf' for source PDFs, 'filled' for filled PDFs

    Returns:
        List of S3 keys
    """
    client = get_s3_client()
    prefix = PDF_PREFIX if key_type == "pdf" else FILLED_PREFIX

    try:
        response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [obj["Key"] for obj in response["Contents"]]
    except ClientError as e:
        logger.error(f"Failed to list S3 objects: {e}")
        raise


def test_connection() -> bool:
    """
    Test S3 connection by listing objects.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = get_s3_client()
        client.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1)
        logger.info("S3 connection test successful")
        return True
    except Exception as e:
        logger.error(f"S3 connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test S3 connection when run directly
    print("Testing S3 connection...")

    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
        print("Error: Missing required environment variables")
        print(f"  BUCKETEER_AWS_ACCESS_KEY_ID: {'set' if AWS_ACCESS_KEY_ID else 'NOT SET'}")
        print(f"  BUCKETEER_AWS_SECRET_ACCESS_KEY: {'set' if AWS_SECRET_ACCESS_KEY else 'NOT SET'}")
        print(f"  BUCKETEER_BUCKET_NAME: {'set' if BUCKET_NAME else 'NOT SET'}")
        exit(1)

    print(f"Bucket: {BUCKET_NAME}")
    print(f"Region: {AWS_REGION}")

    if test_connection():
        print("Connection successful!")

        # List existing PDFs
        pdfs = list_pdfs("pdf")
        print(f"Source PDFs in bucket: {len(pdfs)}")
        for pdf in pdfs[:5]:
            print(f"  - {pdf}")
        if len(pdfs) > 5:
            print(f"  ... and {len(pdfs) - 5} more")

        filled = list_pdfs("filled")
        print(f"Filled PDFs in bucket: {len(filled)}")
    else:
        print("Connection failed!")
        exit(1)
