"""Prompt construction for the grounded LLM.

The system prompt instructs the model to answer only from the supplied
context, cite sources, state exact dates for calendar answers, and give
the exact not-found reply when the context is insufficient.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from app.vectorstore.base import SearchResult

# The exact phrase the model must use when the answer is not in the docs.
NOT_FOUND_REPLY = (
    "I couldn't find this in the IoT syllabus or academic calendar."
)

OUT_OF_SCOPE_REPLY = (
    "That question is outside the two documents I can answer from "
    "(the IoT syllabus and the academic calendar)."
)


SYSTEM_PROMPT = """You are the IoT Department Assistant. You answer student \
questions using ONLY the context provided below, which comes from two \
sources: the IoT (Internet of Things) B.Tech syllabus and the academic \
calendar.

RULES:
1. Answer ONLY from the provided context. Never use outside knowledge or \
invent facts.
2. Cite your sources inline using the format [file, p.X] after the relevant \
statement, where file is the source file name and X is the page number.
3. For calendar/date questions:
   - Always state exact dates and, when available, the day of the week.
   - Today's date is {today}. Use it to interpret words like "next", \
"upcoming", or "when does X start". Events marked (upcoming) are in the \
future; events marked (passed) are in the past.
4. Be concise and direct. Use bullet points for lists of courses or events.
5. If the context does NOT contain enough information to answer, reply with \
EXACTLY this sentence and nothing else:
   "{not_found}"
6. If the question is about something other than the IoT syllabus or academic \
calendar (for example fees, hostel rules, exam results, admissions), reply \
with EXACTLY this sentence and nothing else:
   "{out_of_scope}"
"""


def _format_source(result: SearchResult) -> str:
    """Format a single retrieved chunk as a context block with a citation tag."""
    meta = result.metadata
    source_file = meta.get("source_file", "unknown")
    page = meta.get("page_start", "?")

    header_parts = [f"[source: {source_file}, p.{page}]"]
    if meta.get("doc_type"):
        header_parts.append(f"type={meta['doc_type']}")
    if meta.get("course_code"):
        header_parts.append(f"course={meta['course_code']}")
    if meta.get("event_type"):
        temporal = meta.get("_temporal")
        tag = f"event={meta['event_type']}"
        if temporal:
            tag += f" ({temporal})"
        header_parts.append(tag)

    header = " ".join(header_parts)
    return f"{header}\n{result.text}"


def build_context(results: list[SearchResult]) -> str:
    """Join retrieved chunks into a single context string."""
    if not results:
        return "(no relevant context found)"
    blocks = [_format_source(r) for r in results]
    return "\n\n---\n\n".join(blocks)


def build_messages(
    question: str,
    results: list[SearchResult],
    today: Optional[date] = None,
) -> list[dict[str, str]]:
    """Build the chat message list for the LLM.

    Returns a list of ``{"role": ..., "content": ...}`` dicts.
    """
    if today is None:
        today = date.today()

    system = SYSTEM_PROMPT.format(
        today=today.isoformat(),
        not_found=NOT_FOUND_REPLY,
        out_of_scope=OUT_OF_SCOPE_REPLY,
    )

    context = build_context(results)
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, with inline citations."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
