"""Embeddings: call Ollama's embedding API with batching and retries.

Uses the ``nomic-embed-text`` model by default.  Texts are sent in
batches to avoid overwhelming the Ollama server, and failed batches
are retried with exponential backoff.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import ollama

from app.config import EMBED_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

# How many texts to embed in a single Ollama request.
BATCH_SIZE = 16

# Retry settings.
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds; doubled each retry


def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
) -> list[list[float]]:
    """Embed a list of texts using Ollama.

    Parameters
    ----------
    texts:
        The texts to embed.
    model:
        The Ollama embedding model name.
    batch_size:
        Number of texts per Ollama request.
    max_retries:
        Number of retry attempts per batch on failure.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text (same length and order).

    Raises
    ------
    ConnectionError
        If Ollama is unreachable after all retries.
    """
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_num = start // batch_size

        for attempt in range(1, max_retries + 1):
            try:
                response = ollama.embed(model=model, input=batch)
                all_embeddings.extend(response["embeddings"])
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < max_retries:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedding batch %d failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs …",
                        batch_num,
                        attempt,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Embedding batch %d failed after %d attempts: %s",
                        batch_num,
                        max_retries,
                        exc,
                    )
                    raise

    return all_embeddings


def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single text and return the embedding vector."""
    return embed_texts([text], model=model)[0]
