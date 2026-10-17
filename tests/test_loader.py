"""Unit tests for the loader's pure functions (no PDF required)."""
from __future__ import annotations

from pathlib import Path

from app.ingestion.loader import _detect_doc_type, format_tables


class TestDetectDocType:
    def test_syllabus(self):
        p = Path("documents/syllabus/I-B.Tech-IOT-R25.pdf")
        assert _detect_doc_type(p) == "syllabus"

    def test_calendar(self):
        p = Path("documents/academic_calendar/B.Tech-I-Year-2026-2027.pdf")
        assert _detect_doc_type(p) == "calendar"

    def test_unknown(self):
        p = Path("documents/random/notes.pdf")
        assert _detect_doc_type(p) == "unknown"


class TestFormatTables:
    def test_single_table(self):
        tables = [[["Name", "Credits"], ["CN", "3"], ["DBMS", "4"]]]
        result = format_tables(tables)
        assert "Name | Credits" in result
        assert "CN | 3" in result
        assert "DBMS | 4" in result

    def test_none_cells(self):
        tables = [[["A", None, "C"], ["1", "2", "3"]]]
        result = format_tables(tables)
        assert "A |  | C" in result
        assert "1 | 2 | 3" in result

    def test_multiple_tables(self):
        tables = [[["a", "b"]], [["c", "d"]]]
        result = format_tables(tables)
        assert "a | b" in result
        assert "c | d" in result

    def test_empty(self):
        assert format_tables([]) == ""
