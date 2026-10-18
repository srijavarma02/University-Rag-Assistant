"""LLM module: call Ollama chat for grounded answers.

Supports both a blocking call and a streaming generator.  Temperature
is low (0.1) so answers stay faithful to the context.
"""
from __future__ import annotations

import logging
from typing import Iterator

import ollama

from app.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


def chat(
    messages: list[dict[str, str]],
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """Send *messages* to the Ollama chat API and return the full answer."""
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def chat_stream(
    messages: list[dict[str, str]],
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> Iterator[str]:
    """Stream the answer token-by-token from the Ollama chat API.

    Yields text fragments as they arrive.
    """
    stream = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
