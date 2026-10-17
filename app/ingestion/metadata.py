"""Metadata detection: year, semester, course codes, dates, event types.

All regexes are configurable module-level constants so they can be tuned
without changing the function logic.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from dateutil import parser as date_parser

# ---------------------------------------------------------------------------
# Roman numeral / word-to-int helpers
# ---------------------------------------------------------------------------
ROMAN_TO_INT: dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8,
}

WORD_TO_INT: dict[str, int] = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4,
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
}

# ---------------------------------------------------------------------------
# Year / semester regexes (configurable)
# ---------------------------------------------------------------------------
# Matches: "IV YEAR", "B.TECH. IV YEAR", "III B.Tech", "Fourth Year", "1st Year"
YEAR_PATTERNS: list[str] = [
    r"(?:B\.?\s*TECH\.?\s+)?([IVX]+)\s+YEAR",
    r"([IVX]+)\s+B\.?\s*TECH",
    r"(FIRST|SECOND|THIRD|FOURTH)\s+YEAR",
    r"(\d)(?:st|nd|rd|th)?\s+YEAR",
    r"B\.?\s*Tech\.?\s+([IVX]+)\s+Year",
]

# Matches: "VII SEMESTER", "B.Tech. V Semester", "5th Semester", "Semester 5"
SEMESTER_PATTERNS: list[str] = [
    r"(?:B\.?\s*Tech\.?\s+)?([IVX]+)\s+SEMESTER",
    r"(\d)(?:st|nd|rd|th)?\s+[Ss]emester",
    r"[Ss]emester\s+(\d)",
]

# ---------------------------------------------------------------------------
# Course code regexes
# ---------------------------------------------------------------------------
# Pattern: 2-digit year prefix + 2 letters + 1 digit + 2 letters + 3 digits
# e.g. 22PC1IN401, 25BS1MT101, 22PE1CS402
COURSE_CODE_RE: re.Pattern[str] = re.compile(
    r"\b(\d{2}[A-Z]{2}\d[A-Z]{2}\d{3})\b"
)

# Course detail header: "(22PC1AM202) DATA ENGINEERING"
COURSE_DETAIL_RE: re.Pattern[str] = re.compile(
    r"\((\d{2}[A-Z]{2}\d[A-Z]{2}\d{3})\)\s+(.+?)$",
    re.MULTILINE,
)

# Bare course code on its own line (course list pages)
BARE_CODE_RE: re.Pattern[str] = re.compile(
    r"^(\d{2}[A-Z]{2}\d[A-Z]{2}\d{3})\s*$",
    re.MULTILINE,
)

# Unit heading: "UNIT-I:", "UNIT-II:", "UNIT-V:"
UNIT_RE: re.Pattern[str] = re.compile(
    r"^(UNIT[-\s]*([IVX]+)\s*:?)\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)

# Academic year: "2026 — 2027", "2026-2027", "Academic Year 2026 — 2027"
ACADEMIC_YEAR_RE: re.Pattern[str] = re.compile(
    r"(?:Academic\s+Year\s+)?(\d{4})\s*[-–—]\s*(\d{2,4})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Year / semester detection
# ---------------------------------------------------------------------------
def _resolve_token(token: str) -> Optional[int]:
    """Resolve a roman numeral, word, or digit to an integer."""
    token = token.strip().upper()
    if token in ROMAN_TO_INT:
        return ROMAN_TO_INT[token]
    if token in WORD_TO_INT:
        return WORD_TO_INT[token]
    if token.isdigit():
        return int(token)
    return None


def detect_year(text: str) -> Optional[int]:
    """Detect year of study (1–4) from text headings."""
    for pattern in YEAR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = _resolve_token(match.group(1))
            if val and 1 <= val <= 4:
                return val
    return None


def detect_semester(text: str) -> Optional[int]:
    """Detect overall semester number (1–8) from text headings."""
    for pattern in SEMESTER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = _resolve_token(match.group(1))
            if val and 1 <= val <= 8:
                return val
    return None


def overall_semester_to_year_sem(overall_sem: int) -> tuple[int, int]:
    """Convert overall semester (1–8) to (year_of_study, semester_in_year)."""
    year = (overall_sem - 1) // 2 + 1
    sem_in_year = (overall_sem - 1) % 2 + 1
    return year, sem_in_year


def year_sem_to_overall(year: int, sem_in_year: int) -> int:
    """Convert (year, semester_in_year) to overall semester (1–8)."""
    return (year - 1) * 2 + sem_in_year


# ---------------------------------------------------------------------------
# Course detection
# ---------------------------------------------------------------------------
def detect_course_code(text: str) -> Optional[str]:
    """Find the first course code (e.g. ``22PC1IN401``) in *text*."""
    match = COURSE_CODE_RE.search(text)
    return match.group(1) if match else None


def detect_course_detail_header(
    text: str,
) -> Optional[tuple[str, str]]:
    """Detect a course detail header like ``(22PC1AM202) DATA ENGINEERING``.

    Returns ``(course_code, course_title)`` or *None*.
    """
    match = COURSE_DETAIL_RE.search(text)
    if match:
        return match.group(1), match.group(2).strip()
    return None


def detect_course_type(code: str, title: str = "") -> str:
    """Classify a course as ``theory``, ``lab``, or ``elective``.

    Uses the 6th character of the code (position index 5):
      ``1`` → theory, ``2`` → lab/practical, ``3`` → drawing,
      ``4`` → project, ``6`` → mandatory/non-credit.
    Also checks if ``PE`` (Professional Elective) appears in the code,
    or if "Laboratory" appears in the title.
    """
    title_upper = title.upper()
    if "PE" in code[:4]:
        return "elective"
    if "LABORATORY" in title_upper or "LAB" in title_upper:
        return "lab"
    if len(code) >= 5 and code[4] == "2":
        return "lab"
    if len(code) >= 5 and code[4] == "4":
        return "project"
    return "theory"


# ---------------------------------------------------------------------------
# Academic year
# ---------------------------------------------------------------------------
def detect_academic_year(text: str) -> Optional[str]:
    """Detect academic year like ``2026-2027`` from text."""
    match = ACADEMIC_YEAR_RE.search(text)
    if match:
        start = match.group(1)
        end = match.group(2)
        # Normalise 2-digit end to 4-digit
        if len(end) == 2:
            end = start[:2] + end
        return f"{start}-{end}"
    return None


# ---------------------------------------------------------------------------
# Date parsing (for calendar OCR text)
# ---------------------------------------------------------------------------
# Month names for date extraction
_MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

# Matches "15 August 2026" or "15-18 September 2026"
_DATE_RANGE_RE: re.Pattern[str] = re.compile(
    rf"(\d{{1,2}})\s*[-–—to]+\s*(\d{{1,2}})\s+({_MONTHS})\s+(\d{{4}})",
    re.IGNORECASE,
)
# Matches "15 August 2026"
_SINGLE_DATE_RE: re.Pattern[str] = re.compile(
    rf"(\d{{1,2}})\s+({_MONTHS})\s+(\d{{4}})",
    re.IGNORECASE,
)
# Matches "31 August 2026 - 02 September 2026"
_CROSS_MONTH_RANGE_RE: re.Pattern[str] = re.compile(
    rf"(\d{{1,2}})\s+({_MONTHS})\s+(\d{{4}})"
    rf"\s*[-–—to]+\s*"
    rf"(\d{{1,2}})\s+({_MONTHS})\s+(\d{{4}})",
    re.IGNORECASE,
)


def clean_ocr_date(text: str) -> str:
    """Remove common OCR artefacts from a date string.

    Handles superscript ordinals (``°``, ``"``, ``#``, ``*``), stray
    letters mixed with digits (``O1`` → ``01``), and normalises dashes.
    """
    # O → 0 when it looks like a digit
    text = re.sub(r"\bO(\d)", r"0\1", text)
    # Remove superscript / ordinal markers
    for ch in '°"#*_|':
        text = text.replace(ch, "")
    text = re.sub(r"(\d)(?:st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
    # Normalise dashes
    text = text.replace("—", "-").replace("–", "-")
    # Remove leading zeros from day numbers (015 → 15, 08 → 8)
    text = re.sub(r"\b0+(\d{1,2})\b", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_date_range(text: str) -> Optional[tuple[str, str]]:
    """Parse a date or date range from messy OCR text.

    Returns ``(start_date_iso, end_date_iso)`` or *None*.
    Both are ``YYYY-MM-DD`` strings.
    """
    cleaned = clean_ocr_date(text)

    # Try cross-month range first: "31 August 2026 - 02 September 2026"
    match = _CROSS_MONTH_RANGE_RE.search(cleaned)
    if match:
        try:
            d1 = date_parser.parse(f"{match.group(1)} {match.group(2)} {match.group(3)}")
            d2 = date_parser.parse(f"{match.group(4)} {match.group(5)} {match.group(6)}")
            return d1.date().isoformat(), d2.date().isoformat()
        except (ValueError, date_parser.ParserError):
            pass

    # Try same-month range: "15-18 September 2026"
    match = _DATE_RANGE_RE.search(cleaned)
    if match:
        try:
            d1 = date_parser.parse(f"{match.group(1)} {match.group(3)} {match.group(4)}")
            d2 = date_parser.parse(f"{match.group(2)} {match.group(3)} {match.group(4)}")
            # If start > end, this is likely a garbled single date, not a range
            if d1.date() > d2.date():
                return d1.date().isoformat(), d1.date().isoformat()
            return d1.date().isoformat(), d2.date().isoformat()
        except (ValueError, date_parser.ParserError):
            pass

    # Try single date: "15 August 2026"
    match = _SINGLE_DATE_RE.search(cleaned)
    if match:
        try:
            d = date_parser.parse(f"{match.group(1)} {match.group(2)} {match.group(3)}")
            return d.date().isoformat(), d.date().isoformat()
        except (ValueError, date_parser.ParserError):
            pass

    return None


# ---------------------------------------------------------------------------
# Event classification (for calendar)
# ---------------------------------------------------------------------------
def classify_event_type(text: str) -> str:
    """Classify a calendar event into one of the known types.

    Returns one of: ``commencement``, ``internal_exam``, ``end_exam``,
    ``holiday``, ``vacation``, ``last_working_day``, ``other``.
    """
    t = text.lower()
    if "commencement" in t or "reopening" in t or "reopen" in t or "start of class" in t:
        return "commencement"
    if "end exam" in t or "semester end" in t or "see" in t or "end examination" in t:
        return "end_exam"
    if "cie" in t or "class assessment" in t or "sessional" in t or "internal" in t:
        return "internal_exam"
    if "holiday" in t or "break" in t or "festival" in t or "dasara" in t or "sankranti" in t or "deepavali" in t:
        return "holiday"
    if "vacation" in t:
        return "vacation"
    if "last working" in t or "last date" in t or "end of class" in t:
        return "last_working_day"
    return "other"
