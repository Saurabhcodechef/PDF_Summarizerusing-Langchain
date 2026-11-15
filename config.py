"""Application configuration: environment variables and tunable defaults.

Everything the app needs to boot lives here so the UI layer never reads
``os.environ`` directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# ``override=False`` keeps real environment variables authoritative.
load_dotenv(PROJECT_ROOT / ".env", override=False)

API_KEY_ENV = "GEMINI_API_KEY"

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- Chunking ---------------------------------------------------------------
# Sizing is in characters so the splitter stays dependency-free. 9000 chars is
# roughly 2k tokens of English prose, a comfortable amount of context per call.
DEFAULT_CHUNK_CHARS = 9000
DEFAULT_CHUNK_OVERLAP = 600
MIN_CHUNK_CHARS = 1500
MAX_CHUNK_CHARS = 40_000

# --- Model behaviour -------------------------------------------------------
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_CONCURRENCY = 3

# Above this many tokens a single stuffed prompt gets expensive and lossy, so
# switch to map-reduce.
SINGLE_PASS_TOKEN_LIMIT = 80_000

# Longest text we will ever hand to one prompt in single-pass mode.
HARD_INPUT_TOKEN_LIMIT = 400_000

# --- Retrieval -------------------------------------------------------------
EMBEDDING_BATCH_SIZE = 32
RETRIEVAL_TOP_K = 6

MAX_UPLOAD_MB = 200


def get_api_key() -> str | None:
    """Return the configured Gemini API key, or ``None`` when unset."""
    key = os.getenv(API_KEY_ENV, "").strip()
    return key or None


def require_api_key() -> str:
    """Return the API key or raise a message the UI can show verbatim."""
    key = get_api_key()
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Add it to the .env file in "
            f"{PROJECT_ROOT} and restart the app."
        )
    return key
