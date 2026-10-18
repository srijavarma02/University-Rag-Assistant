"""Hybrid retrieval: vector search + BM25 keyword search with RRF.

Also handles:
  - auto-detecting year/semester filters from the question
  - routing the query to syllabus, calendar, or both
  - detecting course codes and boosting exact matches
  - annotating calendar events as upcoming/passed relative to today

The main entry point is :func:`retrieve`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from app.config import TOP_K
from app.embeddings import embed_text
from app.ingestion.metadata import (
    COURSE_CODE_RE,
    year_sem_to_overall,
)
from app.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


# ===========================================================================
# Filter auto-detection
# ===========================================================================
# Maps spoken forms to year of study (1–4)
_YEAR_WORD_PATTERNS: list[tuple[str, int]] = [
    (r"\bfirst\s+year\b", 1),
    (r"\bsecond\s+year\b", 2),
    (r"\bthird\s+year\b", 3),
    (r"\bfourth\s+year\b", 4),
    (r"\b1st\s+year\b", 1),
    (r"\b2nd\s+year\b", 2),
    (r"\b3rd\s+year\b", 3),
    (r"\b4th\s+year\b", 4),
    (r"\bi\s+year\b", 1),
    (r"\bii\s+year\b", 2),
    (r"\biii\s+year\b", 3),
    (r"\biv\s+year\b", 4),
    (r"\byear\s+1\b", 1),
    (r"\byear\s+2\b", 2),
    (r"\byear\s+3\b", 3),
    (r"\byear\s+4\b", 4),
]

# Semester-in-year words
_SEM_IN_YEAR_PATTERNS: list[tuple[str, int]] = [
    (r"\b1st\s+sem", 1),
    (r"\b2nd\s+sem", 2),
    (r"\bfirst\s+sem", 1),
    (r"\bsecond\s+sem", 2),
    (r"\bsem\s*1\b", 1),
    (r"\bsem\s*2\b", 2),
]

# Overall semester (1–8) e.g. "sem 5", "5th semester", "semester 7"
_OVERALL_SEM_PATTERNS: list[tuple[str, int]] = [
    (r"\bsem(?:ester)?\s*([1-8])\b", 0),
    (r"\b([1-8])(?:st|nd|rd|th)\s+semester\b", 0),
]

# Word numbers for "third semester" etc.
_SEM_WORD_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}


@dataclass
class QueryFilters:
    """Filters auto-detected from a question (or set via the sidebar)."""

    year_of_study: Optional[int] = None
    semester_in_year: Optional[int] = None
    overall_semester: Optional[int] = None
    doc_type: Optional[str] = None
    course_codes: list[str] = field(default_factory=list)

    def to_chroma_where(self) -> Optional[dict[str, Any]]:
        """Build a ChromaDB ``where`` filter from the active filters.

        Only the most reliable fields are applied so recall stays high:
        doc_type and year_of_study.  Semester is used for ranking, not
        hard filtering, to avoid dropping summary chunks.
        """
        conditions: list[dict[str, Any]] = []
        if self.doc_type:
            conditions.append({"doc_type": self.doc_type})
        if self.year_of_study:
            conditions.append({"year_of_study": self.year_of_study})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


def detect_filters(question: str) -> QueryFilters:
    """Auto-detect year, semester, and course codes from a question."""
    q = question.lower()
    filters = QueryFilters()

    # Year of study
    for pattern, year in _YEAR_WORD_PATTERNS:
        if re.search(pattern, q):
            filters.year_of_study = year
            break

    # Overall semester (sem 1–8)
    for pattern, _ in _OVERALL_SEM_PATTERNS:
        m = re.search(pattern, q)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 8:
                filters.overall_semester = val
                break

    # Word-based semester ("third semester")
    if filters.overall_semester is None:
        m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\s+semester\b", q)
        if m:
            filters.overall_semester = _SEM_WORD_MAP[m.group(1)]

    # Semester within a year (only if no overall sem found)
    if filters.overall_semester is None:
        for pattern, sem in _SEM_IN_YEAR_PATTERNS:
            if re.search(pattern, q):
                filters.semester_in_year = sem
                break

    # If we have year + semester_in_year, compute overall semester
    if filters.year_of_study and filters.semester_in_year and not filters.overall_semester:
        filters.overall_semester = year_sem_to_overall(
            filters.year_of_study, filters.semester_in_year
        )
    # If we have overall semester, derive year
    elif filters.overall_semester and not filters.year_of_study:
        filters.year_of_study = (filters.overall_semester - 1) // 2 + 1

    # Course codes (uppercase the question for the code regex)
    codes = COURSE_CODE_RE.findall(question.upper())
    filters.course_codes = list(dict.fromkeys(codes))  # dedupe, keep order

    return filters


# ===========================================================================
# Query routing
# ===========================================================================
_CALENDAR_KEYWORDS = {
    "exam", "exams", "holiday", "holidays", "vacation", "date", "dates",
    "schedule", "last working day", "reopen", "reopening", "break",
    "commencement", "when", "deadline", "feedback", "assessment",
    "sessional", "sankranti", "dasara", "deepavali", "semester end",
    "start", "begin", "upcoming", "next",
}

_SYLLABUS_KEYWORDS = {
    "syllabus", "unit", "units", "topic", "topics", "credit", "credits",
    "textbook", "textbooks", "course", "courses", "subject", "subjects",
    "lab", "laboratory", "elective", "electives", "outcome", "outcomes",
    "objective", "objectives", "code", "ltp", "l-t-p",
}


def route_query(question: str) -> Optional[str]:
    """Decide whether to search syllabus, calendar, or both.

    Returns ``"syllabus"``, ``"calendar"``, or *None* (search both).
    """
    q = question.lower()
    cal_score = sum(1 for kw in _CALENDAR_KEYWORDS if kw in q)
    syl_score = sum(1 for kw in _SYLLABUS_KEYWORDS if kw in q)

    if cal_score > syl_score:
        return "calendar"
    if syl_score > cal_score:
        return "syllabus"
    return None  # unclear → search both


# ===========================================================================
# BM25 keyword search
# ===========================================================================
def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase alphanumeric words + course codes."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Thin wrapper around rank-bm25 over a corpus of chunk texts."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.tokenized = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized) if texts else None

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(corpus_index, score)`` for the top matches."""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:top_k] if s > 0]


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================
RRF_K = 60  # standard RRF constant


def reciprocal_rank_fusion(
    vector_ranking: list[str],
    keyword_ranking: list[str],
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse two ranked lists of chunk IDs into combined RRF scores.

    RRF score = sum over lists of 1 / (k + rank).
    """
    scores: dict[str, float] = {}
    for ranking in (vector_ranking, keyword_ranking):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ===========================================================================
# Calendar annotation
# ===========================================================================
def annotate_calendar_results(
    results: list[SearchResult], today: Optional[date] = None
) -> list[SearchResult]:
    """Mark calendar events as upcoming/passed and sort upcoming by date.

    Adds ``_temporal`` ("upcoming"/"passed"/"unknown") and ``_days_away``
    to each result's metadata.  Non-calendar results are returned
    unchanged.
    """
    if today is None:
        today = date.today()

    calendar_results = [r for r in results if r.metadata.get("doc_type") == "calendar"]
    other_results = [r for r in results if r.metadata.get("doc_type") != "calendar"]

    for r in calendar_results:
        start = r.metadata.get("start_date")
        if start:
            try:
                event_date = date.fromisoformat(start)
                days = (event_date - today).days
                r.metadata["_temporal"] = "upcoming" if days >= 0 else "passed"
                r.metadata["_days_away"] = days
            except ValueError:
                r.metadata["_temporal"] = "unknown"
        else:
            r.metadata["_temporal"] = "unknown"

    # Sort calendar: upcoming first (soonest), then unknown, then passed
    def sort_key(r: SearchResult) -> tuple:
        temporal = r.metadata.get("_temporal", "unknown")
        days = r.metadata.get("_days_away", 99999)
        order = {"upcoming": 0, "unknown": 1, "passed": 2}
        return (order.get(temporal, 1), abs(days) if days is not None else 99999)

    calendar_results.sort(key=sort_key)
    return calendar_results + other_results


# ===========================================================================
# Main retrieval
# ===========================================================================
@dataclass
class RetrievalResult:
    """Full output of a retrieval call."""

    results: list[SearchResult]
    filters: QueryFilters
    routed_doc_type: Optional[str]
    debug: dict[str, Any] = field(default_factory=dict)


def retrieve(
    question: str,
    store: VectorStore,
    top_k: int = TOP_K,
    sidebar_year: Optional[int] = None,
    sidebar_semester: Optional[int] = None,
    today: Optional[date] = None,
) -> RetrievalResult:
    """Run hybrid retrieval for *question*.

    Parameters
    ----------
    question:
        The user's natural-language question.
    store:
        The vector store to search.
    top_k:
        Number of final results to return.
    sidebar_year, sidebar_semester:
        Explicit filters from the UI; these override auto-detection.
    today:
        Reference date for calendar annotation (defaults to today).

    Returns
    -------
    RetrievalResult
    """
    # 1. Detect filters, then let the sidebar override
    filters = detect_filters(question)
    if sidebar_year is not None:
        filters.year_of_study = sidebar_year
    if sidebar_semester is not None:
        filters.semester_in_year = sidebar_semester
        if filters.year_of_study:
            filters.overall_semester = year_sem_to_overall(
                filters.year_of_study, sidebar_semester
            )

    # 2. Route the query
    routed = route_query(question)
    if filters.doc_type is None:
        filters.doc_type = routed

    where = filters.to_chroma_where()

    # 3. Vector search (fetch extra for fusion)
    fetch_k = max(top_k * 4, 20)
    query_embedding = embed_text(question)
    vector_results = store.search(query_embedding, top_k=fetch_k, filters=where)

    # 4. BM25 keyword search over the same candidate pool
    #    Build the BM25 corpus from the filtered texts.
    corpus_results = vector_results  # reuse candidates for keyword scoring
    id_to_result = {r.id: r for r in corpus_results}
    corpus_texts = [r.text for r in corpus_results]
    corpus_ids = [r.id for r in corpus_results]

    bm25 = BM25Index(corpus_texts)
    keyword_hits = bm25.search(question, top_k=fetch_k)
    keyword_ranking = [corpus_ids[i] for i, _ in keyword_hits]
    vector_ranking = [r.id for r in vector_results]

    # 5. Reciprocal rank fusion
    fused_scores = reciprocal_rank_fusion(vector_ranking, keyword_ranking)

    # 6. Course-code exact-match boost
    if filters.course_codes:
        for cid, res in id_to_result.items():
            code = (res.metadata.get("course_code") or "").upper()
            if code in filters.course_codes:
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0

    # 7. Rank by fused score, attach the fused score to each result
    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    final: list[SearchResult] = []
    for cid in ranked_ids:
        res = id_to_result.get(cid)
        if res is not None:
            res.score = fused_scores[cid]
            final.append(res)

    # 8. Calendar annotation (upcoming/passed) if calendar results present
    final = annotate_calendar_results(final, today=today)

    # 9. Trim to top_k
    final = final[:top_k]

    debug = {
        "detected_year": filters.year_of_study,
        "detected_semester_in_year": filters.semester_in_year,
        "detected_overall_semester": filters.overall_semester,
        "course_codes": filters.course_codes,
        "routed_doc_type": routed,
        "chroma_where": where,
        "n_vector_candidates": len(vector_results),
        "n_keyword_hits": len(keyword_hits),
    }

    return RetrievalResult(
        results=final,
        filters=filters,
        routed_doc_type=routed,
        debug=debug,
    )
