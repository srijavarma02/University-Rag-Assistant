"""Unit tests for retrieval helpers (filter detection, routing, RRF)."""
from __future__ import annotations

from datetime import date

from app.ingestion.chunkers import Chunk
from app.retrieval import (
    BM25Index,
    QueryFilters,
    annotate_calendar_results,
    detect_filters,
    reciprocal_rank_fusion,
    route_query,
)
from app.vectorstore.base import SearchResult


# ===========================================================================
# Filter detection
# ===========================================================================
class TestDetectFilters:
    def test_year_word(self):
        assert detect_filters("subjects in third year").year_of_study == 3
        assert detect_filters("first year courses").year_of_study == 1

    def test_year_ordinal(self):
        assert detect_filters("2nd year syllabus").year_of_study == 2
        assert detect_filters("4th year electives").year_of_study == 4

    def test_year_roman(self):
        assert detect_filters("II year subjects").year_of_study == 2

    def test_overall_semester(self):
        f = detect_filters("what is in sem 5")
        assert f.overall_semester == 5
        assert f.year_of_study == 3  # sem 5 → year 3

    def test_ordinal_semester(self):
        f = detect_filters("subjects in 7th semester")
        assert f.overall_semester == 7
        assert f.year_of_study == 4

    def test_word_semester(self):
        f = detect_filters("courses in third semester")
        assert f.overall_semester == 3

    def test_year_plus_sem_in_year(self):
        f = detect_filters("3rd year 1st sem subjects")
        assert f.year_of_study == 3
        assert f.semester_in_year == 1
        assert f.overall_semester == 5

    def test_course_code(self):
        f = detect_filters("credits for 22PC1IN401")
        assert "22PC1IN401" in f.course_codes

    def test_course_code_lowercase(self):
        f = detect_filters("topics in 25bs1mt101")
        assert "25BS1MT101" in f.course_codes

    def test_no_filters(self):
        f = detect_filters("tell me about the course")
        assert f.year_of_study is None
        assert f.overall_semester is None


class TestQueryFiltersWhere:
    def test_empty(self):
        assert QueryFilters().to_chroma_where() is None

    def test_doc_type_only(self):
        f = QueryFilters(doc_type="calendar")
        assert f.to_chroma_where() == {"doc_type": "calendar"}

    def test_year_only(self):
        f = QueryFilters(year_of_study=2)
        assert f.to_chroma_where() == {"year_of_study": 2}

    def test_both(self):
        f = QueryFilters(doc_type="syllabus", year_of_study=3)
        where = f.to_chroma_where()
        assert "$and" in where
        assert {"doc_type": "syllabus"} in where["$and"]
        assert {"year_of_study": 3} in where["$and"]


# ===========================================================================
# Query routing
# ===========================================================================
class TestRouteQuery:
    def test_calendar(self):
        assert route_query("when do the exams start?") == "calendar"
        assert route_query("when is the next holiday?") == "calendar"

    def test_syllabus(self):
        assert route_query("what are the units in operating systems?") == "syllabus"
        assert route_query("credits for this course") == "syllabus"

    def test_ambiguous(self):
        # No strong keywords either way
        assert route_query("tell me something") is None


# ===========================================================================
# BM25
# ===========================================================================
class TestBM25Index:
    def test_search(self):
        texts = [
            "operating systems process management",
            "data structures and algorithms",
            "computer networks protocols",
        ]
        idx = BM25Index(texts)
        hits = idx.search("operating systems", top_k=2)
        assert len(hits) >= 1
        # First hit should be the operating systems doc (index 0)
        assert hits[0][0] == 0

    def test_empty_corpus(self):
        idx = BM25Index([])
        assert idx.search("anything", top_k=5) == []

    def test_no_match(self):
        idx = BM25Index(["hello world"])
        hits = idx.search("zzz nonexistent", top_k=5)
        assert hits == []


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================
class TestRRF:
    def test_basic_fusion(self):
        vector = ["a", "b", "c"]
        keyword = ["b", "a", "d"]
        scores = reciprocal_rank_fusion(vector, keyword)
        # 'a' and 'b' appear in both → higher scores
        assert scores["a"] > scores["c"]
        assert scores["b"] > scores["d"]

    def test_top_item_in_both(self):
        vector = ["x", "y"]
        keyword = ["x", "z"]
        scores = reciprocal_rank_fusion(vector, keyword)
        # x is rank-1 in both → highest
        assert max(scores, key=scores.get) == "x"

    def test_empty(self):
        assert reciprocal_rank_fusion([], []) == {}


# ===========================================================================
# Calendar annotation
# ===========================================================================
def _cal_result(rid, start_date, event_type="holiday"):
    return SearchResult(
        id=rid,
        text=f"event {rid}",
        score=0.5,
        metadata={
            "doc_type": "calendar",
            "event_type": event_type,
            "start_date": start_date,
        },
    )


class TestAnnotateCalendar:
    def test_upcoming_and_passed(self):
        today = date(2026, 9, 1)
        results = [
            _cal_result("past", "2026-08-15"),
            _cal_result("future", "2026-10-20"),
        ]
        annotated = annotate_calendar_results(results, today=today)
        by_id = {r.id: r for r in annotated}
        assert by_id["past"].metadata["_temporal"] == "passed"
        assert by_id["future"].metadata["_temporal"] == "upcoming"

    def test_upcoming_sorted_first(self):
        today = date(2026, 9, 1)
        results = [
            _cal_result("past", "2026-08-01"),
            _cal_result("soon", "2026-09-10"),
        ]
        annotated = annotate_calendar_results(results, today=today)
        # Upcoming event should come before passed
        assert annotated[0].id == "soon"

    def test_unknown_date(self):
        results = [_cal_result("nodate", None)]
        annotated = annotate_calendar_results(results, today=date(2026, 9, 1))
        assert annotated[0].metadata["_temporal"] == "unknown"

    def test_non_calendar_unchanged(self):
        syl = SearchResult(
            id="s1", text="syllabus", score=0.9,
            metadata={"doc_type": "syllabus"},
        )
        annotated = annotate_calendar_results([syl], today=date(2026, 9, 1))
        assert annotated[0].id == "s1"
        assert "_temporal" not in annotated[0].metadata
