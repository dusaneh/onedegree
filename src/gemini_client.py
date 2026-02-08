"""
Self-contained Gemini API client for PDF metadata extraction.
All API keys loaded from src/.env file.
"""

from google import genai
from google.genai import types
import json
import re
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
from dotenv import load_dotenv

# Load API key from .env in same directory
load_dotenv(Path(__file__).parent / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL = 'gemini-2.5-flash'
GEMINI_3_FLASH = 'gemini-3-flash-preview'

# Logger Setup
logger = logging.getLogger("gemini_client")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def get_gemini_response(
    prompt: str,
    file_paths: Optional[List[str]] = None,
    response_schema: Optional[BaseModel] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None
) -> Tuple[Any, str]:
    """
    Sends a prompt and optional file(s) using the google-genai Client.
    Uploads files, calls API, and cleans up uploaded files.

    Args:
        prompt: The text prompt to send to the model
        file_paths: Optional list of file paths to upload and include
        response_schema: Optional Pydantic BaseModel class for structured output
        model: Optional model name to use (defaults to GEMINI_MODEL)
        max_output_tokens: Optional max output tokens (default 8192, max varies by model)

    Returns:
        Tuple of (raw_response, parsed_output)
    """
    selected_model = model or GEMINI_MODEL
    uploaded_files = []
    client = None

    try:
        if not GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set. Cannot call Gemini API.")
            return None, "An error occurred: API key not set"

        # Initialize the Client
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Handle File Uploads
        if file_paths:
            logger.info(f"Preparing to upload {len(file_paths)} files...")
            for path in file_paths:
                if os.path.exists(path):
                    uploaded_file = client.files.upload(file=path)
                    uploaded_files.append(uploaded_file)
                    logger.info(f"Successfully uploaded: {uploaded_file.display_name} ({uploaded_file.mime_type})")
                else:
                    logger.warning(f"File not found at path: {path}. Skipping.")

        # Build Contents and Configuration
        contents = uploaded_files + [prompt]

        config = types.GenerateContentConfig()

        # Set max output tokens (default to 65535 for large responses)
        if max_output_tokens:
            config.max_output_tokens = max_output_tokens
        else:
            config.max_output_tokens = 65535  # Maximum for most Gemini models
        logger.info(f"[Gemini] Max output tokens: {config.max_output_tokens:,}")

        # Configure thinking based on model version
        # Gemini 3 uses thinking_level, Gemini 2.5 uses thinking_budget
        if "gemini-3" in selected_model:
            # Gemini 3: use MINIMAL thinking level for structured output
            config.thinking_config = types.ThinkingConfig(thinking_level="MINIMAL")
            logger.info(f"[Gemini] Using Gemini 3 with thinking_level=MINIMAL")
        else:
            # Gemini 2.5: disable thinking with budget=0 for structured output
            config.thinking_config = types.ThinkingConfig(thinking_budget=0)
            logger.info(f"[Gemini] Using Gemini 2.5 with thinking_budget=0")

        if response_schema:
            # Enforce JSON output and provide the schema
            config.response_mime_type = "application/json"
            config.response_schema = response_schema
            logger.info(f"Using enforced JSON schema: {response_schema.__name__}")

        # Generate content
        import time
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        logger.info(f"[Gemini] Calling {selected_model}...")
        logger.info(f"[Gemini] Prompt length: {len(prompt):,} chars")
        logger.info(f"[Gemini] Prompt preview: {prompt_preview}")
        logger.info(f"[Gemini] Waiting for response (this may take several minutes for large prompts)...")

        call_start = time.time()
        response_raw = client.models.generate_content(
            model=selected_model,
            contents=contents,
            config=config
        )
        call_duration = time.time() - call_start

        logger.info(f"[Gemini] Response received in {call_duration:.1f}s")

        # Log response metadata if available
        if hasattr(response_raw, 'usage_metadata'):
            usage = response_raw.usage_metadata
            input_tokens = getattr(usage, 'prompt_token_count', 'N/A')
            output_tokens = getattr(usage, 'candidates_token_count', 'N/A')
            logger.info(f"[Gemini] Tokens used - Input: {input_tokens}, Output: {output_tokens}")

        # Check for direct Pydantic-parsed object
        if response_schema and hasattr(response_raw, 'parsed') and response_raw.parsed is not None:
            logger.info(f"[Gemini] Response parsed directly to Pydantic object")
            return response_raw, response_raw.parsed

        response_text = response_raw.text
        logger.info(f"[Gemini] Response text length: {len(response_text):,} chars")
        return response_raw, response_text

    except Exception as e:
        logger.exception(f"An error occurred with gemini call: {e}")
        return None, f"An error occurred: {e}"

    finally:
        # Cleanup: Delete uploaded files
        if uploaded_files and client:
            logger.info("Cleaning up uploaded files...")
            for f in uploaded_files:
                try:
                    client.files.delete(name=f.name)
                    logger.debug(f"Deleted file: {f.name}")
                except Exception as e:
                    logger.error(f"Failed to delete file {f.name}: {e}")


def get_and_parse_structured_response(
    prompt: str,
    output_schema: Optional[Union[Dict, List, BaseModel]] = None,
    file_paths: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None
) -> Tuple[Any, Union[Dict, List, str, BaseModel]]:
    """
    Calls get_gemini_response and prioritizes Pydantic parsing.
    Falls back to manual JSON parsing if needed.

    Args:
        prompt: The text prompt to send
        output_schema: Expected schema (Pydantic BaseModel class, dict, or list)
        file_paths: Optional list of file paths to include
        model: Optional model name to use (defaults to GEMINI_MODEL)
        max_output_tokens: Optional max output tokens (default 16384)

    Returns:
        Tuple of (raw_response, parsed_output)
    """
    is_pydantic = isinstance(output_schema, type) and issubclass(output_schema, BaseModel)

    # Pass the schema class to the API call for direct parsing by the SDK
    schema_class = output_schema if is_pydantic else None

    raw_response, parsed_output = get_gemini_response(
        prompt=prompt,
        file_paths=file_paths,
        response_schema=schema_class,
        model=model,
        max_output_tokens=max_output_tokens
    )

    # SUCCESS: Pydantic object returned directly by SDK
    if is_pydantic and isinstance(parsed_output, BaseModel):
        logger.info("Structured output parsed directly by SDK using Pydantic.")
        return raw_response, parsed_output

    # ERROR/FAILURE: API error string
    if isinstance(parsed_output, str) and "error occurred" in parsed_output.lower():
        logger.error(f"Gemini API call failed.")
        return raw_response, parsed_output

    # FALLBACK: Manual JSON parsing (if model returned text instead of Pydantic object)
    if isinstance(parsed_output, str):
        json_text = parsed_output.strip()

        # Clean up code fences and trailing commas
        if json_text.startswith("```json"):
            json_text = json_text.removeprefix("```json").strip()
        if json_text.startswith("```"):
            json_text = json_text.removeprefix("```").strip()
        if json_text.endswith("```"):
            json_text = json_text.removesuffix("```").strip()
        json_text = re.sub(r',\s*([\}\]])', r'\1', json_text)

        try:
            parsed_data = json.loads(json_text)
            expected_type = type(output_schema) if output_schema is not None else dict

            if not isinstance(parsed_data, expected_type):
                logger.warning(f"Manual parsing type mismatch. Expected {expected_type.__name__}.")

            logger.info("Successfully parsed structured response via manual JSON.")
            return raw_response, parsed_data

        except json.JSONDecodeError as e:
            # Try to repair truncated JSON by adding missing closing brackets
            logger.warning(f"JSON parse failed, attempting repair: {e}")
            repaired_json = repair_truncated_json(json_text)
            if repaired_json:
                try:
                    parsed_data = json.loads(repaired_json)
                    logger.info("Successfully parsed JSON after repair.")
                    return raw_response, parsed_data
                except json.JSONDecodeError as e2:
                    logger.error(f"JSON repair also failed: {e2}")
                    # Log the problematic JSON for debugging
                    logger.error(f"Original JSON (last 500 chars): ...{json_text[-500:]}")
            return raw_response, f"Manual JSON Decode Error: {e}"


def repair_truncated_json(json_text: str) -> Optional[str]:
    """
    Attempt to repair truncated JSON by adding missing closing brackets.

    This handles common LLM truncation where the output is cut off mid-JSON.
    """
    if not json_text:
        return None

    # Count open vs close brackets
    open_braces = json_text.count('{')
    close_braces = json_text.count('}')
    open_brackets = json_text.count('[')
    close_brackets = json_text.count(']')

    # If balanced, no repair needed
    if open_braces == close_braces and open_brackets == close_brackets:
        return None

    # Try to fix by removing incomplete trailing content and adding closers
    repaired = json_text.rstrip()

    # Remove trailing incomplete entries (e.g., `"key":` or `"key": "incomplete`)
    # Remove trailing comma
    repaired = re.sub(r',\s*$', '', repaired)

    # Remove incomplete key-value pairs at the end
    repaired = re.sub(r',?\s*"[^"]*":\s*"[^"]*$', '', repaired)  # "key": "incomplete value
    repaired = re.sub(r',?\s*"[^"]*":\s*$', '', repaired)  # "key":
    repaired = re.sub(r',?\s*"[^"]*$', '', repaired)  # "incomplete key

    # Recalculate brackets needed
    open_braces = repaired.count('{')
    close_braces = repaired.count('}')
    open_brackets = repaired.count('[')
    close_brackets = repaired.count(']')

    # Add missing closers in reverse order (] before } typically)
    missing_brackets = open_brackets - close_brackets
    missing_braces = open_braces - close_braces

    if missing_brackets > 0:
        repaired += ']' * missing_brackets
    if missing_braces > 0:
        repaired += '}' * missing_braces

    logger.info(f"JSON repair: added {missing_brackets} ']' and {missing_braces} '}}'")

    return repaired

    return raw_response, parsed_output
