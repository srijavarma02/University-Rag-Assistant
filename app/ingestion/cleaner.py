"""Text cleaner: removes repetitive headers/footers, page numbers,
and fixes hyphenated line breaks.

The cleaner preserves headings, numbering, and table structure.  It only
removes content that is clearly boilerplate (repeated on most pages) or
obvious page-number lines.

Typical usage::

    from app.ingestion.cleaner import clean_document
    cleaned_pages = clean_document([page.text for page in doc.pages])
"""
from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Individual cleaning primitives
# ---------------------------------------------------------------------------
def is_page_number(line: str) -> bool:
    """Return *True* if *line* is a standalone page number.

    Matches patterns like ``12``, ``Page 12``, ``- 12 -``, ``12 |``.
    """
    stripped = line.strip()
    if not stripped:
        return False
    patterns = [
        r"^[-\s]*page\s*\d+\s*[-\s]*$",  # "Page 12", "- Page 12 -"
        r"^[-\s]*\d+\s*[-\s]*$",         # "12", "- 12 -"
        r"^\d+\s*\|.*",                  # "12 | some footer"
        r"^.*\|\s*\d+\s*$",             # "footer | 12"
    ]
    return any(re.match(p, stripped, re.IGNORECASE) for p in patterns)


def find_repeated_lines(page_texts: list[str]) -> set[str]:
    """Detect lines that appear near the top or bottom of most pages.

    These are almost certainly running headers or footers and should be
    removed.  A line qualifies if it shows up on at least 60 % of pages
    (minimum 2 pages).
    """
    if len(page_texts) < 3:
        return set()

    line_counter: Counter[str] = Counter()
    for page_text in page_texts:
        lines = page_text.strip().split("\n")
        # Examine first 3 and last 3 non-empty lines
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            continue
        candidates = set()
        for line in non_empty[:3] + non_empty[-3:]:
            stripped = line.strip()
            if stripped and not is_page_number(stripped):
                candidates.add(stripped)
        for line in candidates:
            line_counter[line] += 1

    threshold = max(2, len(page_texts) * 0.6)
    return {line for line, count in line_counter.items() if count >= threshold}


def fix_hyphenated_words(text: str) -> str:
    """Join words broken across lines by a trailing hyphen.

    ``"exam-\\nple"`` becomes ``"example"``.
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Strip trailing whitespace per line and collapse excessive blank lines."""
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Page-level and document-level cleaning
# ---------------------------------------------------------------------------
def clean_page_text(text: str, repeated_lines: set[str]) -> str:
    """Clean a single page: drop page numbers and repeated headers/footers,
    fix hyphenation, and normalise whitespace.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if is_page_number(stripped):
            continue
        if stripped in repeated_lines:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = fix_hyphenated_words(text)
    text = collapse_whitespace(text)
    return text


def clean_document(page_texts: list[str]) -> list[str]:
    """Clean all pages of a document.

    First detects repeated header/footer lines across the whole document,
    then applies per-page cleaning.

    Parameters
    ----------
    page_texts:
        List of raw text strings, one per page.

    Returns
    -------
    list[str]
        Cleaned text, one string per page (same length as input).
    """
    repeated = find_repeated_lines(page_texts)
    return [clean_page_text(pt, repeated) for pt in page_texts]
