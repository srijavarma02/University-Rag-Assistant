"""Central configuration for the iot_rag application.

All settings are loaded from environment variables (or a ``.env`` file)
with sensible defaults, so nothing is hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env if present (safe if file is missing)
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root: app/config.py  ->  parent is project root
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR: Path = ROOT_DIR / os.getenv("DOCUMENTS_DIR", "documents")
DATA_DIR: Path = ROOT_DIR / os.getenv("DATA_DIR", "data")
CHROMA_PATH: Path = ROOT_DIR / os.getenv("CHROMA_PATH", "data/chroma")

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.3"))

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
SYLLABUS_DIR: Path = DOCUMENTS_DIR / "syllabus"
CALENDAR_DIR: Path = DOCUMENTS_DIR / "academic_calendar"
