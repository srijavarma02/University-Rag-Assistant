"""Answer orchestration: retrieve → threshold check → LLM → response.

This is the single entry point the API and CLI call to get a grounded,
cited answer for a question.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterator, Optional

from app.config import SCORE_THRESHOLD, TOP_K
from app.llm import chat, chat_stream
from app.prompts import NOT_FOUND_REPLY, build_messages
from app.retrieval import retrieve
from app.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A citation source attached to an answer."""

    source_file: str
    page: int
    doc_type: str
    snippet: str
    course_code: Optional[str] = None
    event_type: Optional[str] = None


@dataclass
class AnswerResponse:
    """The full response for a question."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _to_sources(results: list[SearchResult]) -> list[Source]:
    """Convert retrieval results into Source objects for the response."""
    sources: list[Source] = []
    for r in results:
        meta = r.metadata
        sources.append(
            Source(
                source_file=meta.get("source_file", "unknown"),
                page=meta.get("page_start", 0),
                doc_type=meta.get("doc_type", "unknown"),
                snippet=r.text[:200].replace("\n", " "),
                course_code=meta.get("course_code"),
                event_type=meta.get("event_type"),
            )
        )
    return sources


def _best_score(results: list[SearchResult]) -> float:
    """Return the highest raw vector similarity among results.

    The retrieval step overwrites ``score`` with the fused RRF score,
    so we look at ``_vector_score`` if present, else fall back to score.
    """
    best = 0.0
    for r in results:
        s = r.metadata.get("_vector_score", r.score)
        if s > best:
            best = s
    return best


def answer_question(
    question: str,
    store: VectorStore,
    top_k: int = TOP_K,
    sidebar_year: Optional[int] = None,
    sidebar_semester: Optional[int] = None,
    score_threshold: float = SCORE_THRESHOLD,
    today: Optional[date] = None,
) -> AnswerResponse:
    """Answer *question* using retrieval + the grounded LLM.

    If the best retrieval score is below *score_threshold*, the LLM is
    skipped and the not-found reply is returned.
    """
    retrieval = retrieve(
        question,
        store,
        top_k=top_k,
        sidebar_year=sidebar_year,
        sidebar_semester=sidebar_semester,
        today=today,
    )
    results = retrieval.results

    debug = dict(retrieval.debug)
    debug["n_results"] = len(results)
    debug["best_score"] = _best_score(results) if results else 0.0

    # Threshold gate: if nothing relevant, skip the LLM
    if not results or debug["best_score"] < score_threshold:
        logger.info(
            "Best score %.3f below threshold %.3f — returning not-found reply",
            debug["best_score"],
            score_threshold,
        )
        return AnswerResponse(
            answer=NOT_FOUND_REPLY,
            sources=[],
            debug=debug,
        )

    messages = build_messages(question, results, today=today)
    answer_text = chat(messages)

    return AnswerResponse(
        answer=answer_text.strip(),
        sources=_to_sources(results),
        debug=debug,
    )


def answer_question_stream(
    question: str,
    store: VectorStore,
    top_k: int = TOP_K,
    sidebar_year: Optional[int] = None,
    sidebar_semester: Optional[int] = None,
    score_threshold: float = SCORE_THRESHOLD,
    today: Optional[date] = None,
) -> tuple[Iterator[str], list[Source], dict[str, Any]]:
    """Streaming variant of :func:`answer_question`.

    Returns a tuple of ``(token_iterator, sources, debug)``.  If the
    score is below threshold, the iterator yields only the not-found
    reply and sources is empty.
    """
    retrieval = retrieve(
        question,
        store,
        top_k=top_k,
        sidebar_year=sidebar_year,
        sidebar_semester=sidebar_semester,
        today=today,
    )
    results = retrieval.results

    debug = dict(retrieval.debug)
    debug["n_results"] = len(results)
    debug["best_score"] = _best_score(results) if results else 0.0

    if not results or debug["best_score"] < score_threshold:
        def _not_found() -> Iterator[str]:
            yield NOT_FOUND_REPLY

        return _not_found(), [], debug

    messages = build_messages(question, results, today=today)
    sources = _to_sources(results)
    return chat_stream(messages), sources, debug
