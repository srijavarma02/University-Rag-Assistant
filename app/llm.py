"""LLM module: call Ollama chat for grounded answers.

Supports both a blocking call and a streaming generator.  Temperature
is low (0.1) so answers stay faithful to the context.
"""
from __future__ import annotations

import logging
from typing import Iterator

import ollama

from app.config import (
    LLM_KEEP_ALIVE,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


def _options(temperature: float) -> dict:
    """Build the Ollama options dict, honouring memory-saving limits."""
    opts: dict = {"temperature": temperature}
    if LLM_NUM_CTX > 0:
        opts["num_ctx"] = LLM_NUM_CTX
    if LLM_NUM_PREDICT > 0:
        opts["num_predict"] = LLM_NUM_PREDICT
    return opts


def chat(
    messages: list[dict[str, str]],
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """Send *messages* to the Ollama chat API and return the full answer."""
    response = ollama.chat(
        model=model,
        messages=messages,
        options=_options(temperature),
        keep_alive=LLM_KEEP_ALIVE,
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
        options=_options(temperature),
        keep_alive=LLM_KEEP_ALIVE,
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
