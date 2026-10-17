"""Unit tests for the text cleaner."""
from __future__ import annotations

from app.ingestion.cleaner import (
    collapse_whitespace,
    clean_document,
    clean_page_text,
    find_repeated_lines,
    fix_hyphenated_words,
    is_page_number,
)


# ---------------------------------------------------------------------------
# is_page_number
# ---------------------------------------------------------------------------
class TestIsPageNumber:
    def test_plain_number(self):
        assert is_page_number("12") is True

    def test_with_dashes(self):
        assert is_page_number("- 12 -") is True

    def test_page_prefix(self):
        assert is_page_number("Page 5") is True
        assert is_page_number("page 42") is True

    def test_not_page_number(self):
        assert is_page_number("Embedded Systems") is False
        assert is_page_number("Unit 3: Computer Networks") is False
        assert is_page_number("") is False

    def test_pipe_footer(self):
        assert is_page_number("12 | Dept of CSE") is True
        assert is_page_number("Dept of CSE | 12") is True


# ---------------------------------------------------------------------------
# fix_hyphenated_words
# ---------------------------------------------------------------------------
class TestFixHyphenatedWords:
    def test_basic_join(self):
        assert fix_hyphenated_words("exam-\nple") == "example"

    def test_multiple(self):
        text = "inter-\nnational network-\ning"
        assert fix_hyphenated_words(text) == "international networking"

    def test_no_change(self):
        assert fix_hyphenated_words("hello world") == "hello world"

    def test_preserves_legitimate_hyphen(self):
        # "well-known" should stay as-is when not at line end
        assert fix_hyphenated_words("well-known\nconcept") == "well-known\nconcept"


# ---------------------------------------------------------------------------
# collapse_whitespace
# ---------------------------------------------------------------------------
class TestCollapseWhitespace:
    def test_strips_trailing(self):
        assert collapse_whitespace("hello   \nworld  ") == "hello\nworld"

    def test_collapses_blank_lines(self):
        text = "a\n\n\n\nb"
        assert collapse_whitespace(text) == "a\n\nb"

    def test_strips_outer(self):
        assert collapse_whitespace("  \nhello\n  ") == "hello"


# ---------------------------------------------------------------------------
# find_repeated_lines
# ---------------------------------------------------------------------------
class TestFindRepeatedLines:
    def test_detects_header(self):
        pages = [
            "UNIVERSITY HEADER\ncontent A",
            "UNIVERSITY HEADER\ncontent B",
            "UNIVERSITY HEADER\ncontent C",
            "UNIVERSITY HEADER\ncontent D",
        ]
        repeated = find_repeated_lines(pages)
        assert "UNIVERSITY HEADER" in repeated

    def test_ignores_short_lists(self):
        # Fewer than 3 pages — should return empty set
        pages = ["HEADER\ncontent A", "HEADER\ncontent B"]
        assert find_repeated_lines(pages) == set()

    def test_page_numbers_not_treated_as_headers(self):
        pages = ["Page 1\ncontent"] * 5
        repeated = find_repeated_lines(pages)
        assert "Page 1" not in repeated


# ---------------------------------------------------------------------------
# clean_page_text
# ---------------------------------------------------------------------------
class TestCleanPageText:
    def test_removes_page_number(self):
        text = "Some content\n12\nMore content"
        result = clean_page_text(text, set())
        assert "12" not in result.split("\n")
        assert "Some content" in result
        assert "More content" in result

    def test_removes_repeated_header(self):
        text = "RUNNING HEADER\nMain content here"
        result = clean_page_text(text, {"RUNNING HEADER"})
        assert "RUNNING HEADER" not in result
        assert "Main content here" in result

    def test_fixes_hyphenation(self):
        text = "comput-\ner networks"
        result = clean_page_text(text, set())
        assert "computer networks" in result


# ---------------------------------------------------------------------------
# clean_document (integration)
# ---------------------------------------------------------------------------
class TestCleanDocument:
    def test_preserves_content(self):
        pages = [
            "HEADER\nUnit 1: Introduction\nTopics A, B, C\n12",
            "HEADER\nUnit 2: Advanced\nTopics D, E, F\n13",
            "HEADER\nUnit 3: Expert\nTopics G, H, I\n14",
            "HEADER\nUnit 4: Master\nTopics J, K, L\n15",
        ]
        cleaned = clean_document(pages)
        assert len(cleaned) == 4
        # Headers and page numbers removed
        for page in cleaned:
            assert "HEADER" not in page
            assert not any(line.strip() in {"12", "13", "14", "15"} for line in page.split("\n"))
        # Content preserved
        assert "Unit 1: Introduction" in cleaned[0]
        assert "Topics D, E, F" in cleaned[1]

    def test_empty_pages(self):
        cleaned = clean_document(["", "", ""])
        assert all(p == "" for p in cleaned)
