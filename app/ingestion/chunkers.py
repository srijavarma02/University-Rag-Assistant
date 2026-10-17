"""Chunkers: split syllabus and calendar documents into search chunks.

Syllabus → one chunk per course (split by unit if too long) + one
summary chunk per semester.

Calendar → one chunk per event + one summary chunk per semester.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.ingestion.loader import DocumentContent
from app.ingestion.metadata import (
    classify_event_type,
    detect_academic_year,
    detect_course_code,
    detect_course_detail_header,
    detect_course_type,
    detect_semester,
    detect_year,
    overall_semester_to_year_sem,
    parse_date_range,
    year_sem_to_overall,
    BARE_CODE_RE,
    UNIT_RE,
    _CROSS_MONTH_RANGE_RE,
    _DATE_RANGE_RE,
    _SINGLE_DATE_RE,
    _MONTHS,
)

# Rough token estimate: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4
MAX_COURSE_CHARS = 1200 * CHARS_PER_TOKEN  # ≈ 4800 chars

# Known section headers in course detail pages (used to trim noise)
SECTION_HEADERS = {
    "COURSE OBJECTIVES",
    "COURSE OUTCOMES",
    "COURSE ARTICULATION MATRIX",
    "TEACHING SCHEME",
    "EVALUATION SCHEME",
    "TEXT BOOKS",
    "REFERENCES",
    "UNIT-I",
    "UNIT-II",
    "UNIT-III",
    "UNIT-IV",
    "UNIT-V",
}


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single search chunk with metadata."""

    id: str
    text: str
    doc_type: str  # 'syllabus' or 'calendar'
    chunk_kind: str  # 'course', 'course_unit', 'semester_summary', 'event', 'calendar_summary'
    year_of_study: Optional[int] = None
    semester_in_year: Optional[int] = None
    overall_semester: Optional[int] = None
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    event_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    academic_year: Optional[str] = None
    source_file: str = ""
    page_start: int = 0
    page_end: int = 0


def _make_id(source_file: str, text: str) -> str:
    """Generate a stable ID from source file and text hash."""
    h = hashlib.md5(f"{source_file}:{text[:200]}".encode()).hexdigest()[:12]
    return f"{source_file}_{h}"


# ---------------------------------------------------------------------------
# Syllabus chunker
# ---------------------------------------------------------------------------
def _split_by_units(text: str, code: str, title: str) -> list[str]:
    """Split course text by UNIT headings, repeating the code+title header."""
    # Find all unit heading positions
    matches = list(UNIT_RE.finditer(text))
    if not matches:
        return [text]

    header = f"({code}) {title}\n"
    pieces: list[str] = []

    # Text before first unit (objectives, outcomes, etc.) — keep as intro
    pre_unit = text[: matches[0].start()].strip()
    if pre_unit:
        pieces.append(f"{header}{pre_unit}")

    # Each unit
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        unit_text = text[start:end].strip()
        pieces.append(f"{header}{unit_text}")

    return pieces


def _extract_course_list(
    page_text: str,
) -> list[tuple[str, str]]:
    """Extract (course_code, course_title) pairs from a course list page.

    Course list pages have bare course codes followed by their titles on
    the next line.
    """
    pairs: list[tuple[str, str]] = []
    lines = page_text.split("\n")
    i = 0
    while i < len(lines):
        m = BARE_CODE_RE.match(lines[i].strip())
        if m:
            code = m.group(1)
            # Next non-empty line is the title
            title = ""
            if i + 1 < len(lines):
                title = lines[i + 1].strip()
            pairs.append((code, title))
            i += 2
        else:
            i += 1
    return pairs


def chunk_syllabus(
    doc: DocumentContent,
    cleaned_pages: list[str],
) -> list[Chunk]:
    """Chunk a syllabus document into course and semester-summary chunks.

    Walks through pages tracking the current year/semester from headings.
    When a ``(CODE) TITLE`` course detail header is found, collects all
    text until the next course header.  Long courses are split by unit.
    Course list pages yield semester-summary chunks.
    """
    chunks: list[Chunk] = []

    current_year: Optional[int] = None
    current_sem: Optional[int] = None
    current_academic_year: Optional[str] = None

    # Accumulator for the current course being collected
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    course_pages: list[str] = []
    course_page_start: int = 0

    # Course list entries for the current semester
    semester_courses: list[tuple[str, str]] = []
    sem_summary_year: Optional[int] = None
    sem_summary_sem: Optional[int] = None

    def flush_course(end_page: int) -> None:
        """Save the accumulated course as chunk(s)."""
        nonlocal course_code, course_title, course_pages, course_page_start
        if not course_code or not course_pages:
            course_code = None
            course_title = None
            course_pages = []
            return

        full_text = "\n".join(course_pages).strip()
        ctype = detect_course_type(course_code, course_title or "")

        def make_chunk(text: str, kind: str = "course") -> Chunk:
            y, s = _resolve_year_sem()
            return Chunk(
                id=_make_id(doc.source_file, text),
                text=text,
                doc_type="syllabus",
                chunk_kind=kind,
                year_of_study=y,
                semester_in_year=s,
                overall_semester=current_sem,
                course_code=course_code,
                course_title=course_title,
                academic_year=current_academic_year,
                source_file=doc.source_file,
                page_start=course_page_start + 1,
                page_end=end_page,
            )

        def _resolve_year_sem() -> tuple[Optional[int], Optional[int]]:
            if current_sem:
                y, s = overall_semester_to_year_sem(current_sem)
                return y, s
            return current_year, None

        if len(full_text) > MAX_COURSE_CHARS:
            pieces = _split_by_units(full_text, course_code, course_title or "")
            for piece in pieces:
                chunks.append(make_chunk(piece, "course_unit"))
        else:
            chunks.append(make_chunk(full_text, "course"))

        course_code = None
        course_title = None
        course_pages = []
        course_page_start = 0

    def flush_semester_summary() -> None:
        """Create a semester-summary chunk from collected course list entries."""
        nonlocal semester_courses
        if not semester_courses:
            return
        y = sem_summary_year
        s = sem_summary_sem
        if current_sem:
            y, s = overall_semester_to_year_sem(current_sem)

        lines = [f"Semester Summary (Year {y}, Semester {s})"]
        for code, title in semester_courses:
            lines.append(f"  {code} – {title}")
        text = "\n".join(lines)

        chunks.append(
            Chunk(
                id=_make_id(doc.source_file, text),
                text=text,
                doc_type="syllabus",
                chunk_kind="semester_summary",
                year_of_study=y,
                semester_in_year=s,
                overall_semester=current_sem,
                academic_year=current_academic_year,
                source_file=doc.source_file,
                page_start=0,
                page_end=0,
            )
        )
        semester_courses = []

    for page_idx, page_text in enumerate(cleaned_pages):
        # Update year/semester from headings on this page
        year = detect_year(page_text)
        sem = detect_semester(page_text)
        acyear = detect_academic_year(page_text)

        if year:
            # Year changed → flush any pending course and semester summary
            if year != current_year:
                flush_course(page_idx)
                flush_semester_summary()
            current_year = year
        if sem:
            if sem != current_sem:
                flush_course(page_idx)
                flush_semester_summary()
            current_sem = sem
        if acyear:
            current_academic_year = acyear

        # Check for course detail header
        header = detect_course_detail_header(page_text)
        if header:
            # New course starts → flush previous course
            if course_code:
                flush_course(page_idx)
            course_code = header[0]
            course_title = header[1]
            course_pages = [page_text]
            course_page_start = page_idx
            continue

        # If we're inside a course, keep accumulating
        if course_code:
            course_pages.append(page_text)
            continue

        # Otherwise this might be a course list page → extract entries
        entries = _extract_course_list(page_text)
        if entries and len(entries) >= 2:
            semester_courses = entries
            sem_summary_year = current_year
            sem_summary_sem = current_sem

    # Flush remaining course and summary
    flush_course(len(cleaned_pages))
    flush_semester_summary()

    return chunks


# ---------------------------------------------------------------------------
# Calendar chunker
# ---------------------------------------------------------------------------
def _is_date_line(line: str) -> bool:
    """Heuristic: does this line look like it contains a date?"""
    cleaned = line.strip()
    if not cleaned:
        return False
    # Check for month names
    for month in re.split(r"\|", _MONTHS):
        if month.lower() in cleaned.lower():
            return True
    return False


# Keywords that signal the start of a new calendar event
_EVENT_START_KEYWORDS = (
    "student induction", "commencement", "class review", "cie",
    "last date", "student feedback", "registration", "semester end",
    "see:", "preparation", "course end", "end of class", "semester break",
    "summer break", "holiday", "vacation", "reopening", "reopen",
    "practical examination", "sessional examination", "class assessment",
    "internship", "project", "survey", "induction",
)

# Connector words that indicate a continuation line, not a new event
_CONTINUATION_HINTS = ("answer scripts", "exam branch", "marks in", "fee payment",
                       "holidays &", "for semester")


def _is_new_event(line: str) -> bool:
    """Heuristic: does this line start a new event or continue the previous one?"""
    lower = line.lower().strip()
    if lower.startswith(_EVENT_START_KEYWORDS):
        return True
    if lower.startswith(_CONTINUATION_HINTS):
        return False
    # Short lines (< 25 chars) that don't start with a keyword are continuations
    if len(lower) < 25:
        return False
    # Default: treat as continuation
    return False


def _extract_event_name(line: str) -> str:
    """Extract a clean event name from an OCR line."""
    # Remove leading/trailing noise characters
    name = line.strip()
    name = re.sub(r"^[|_\-\s]+", "", name)
    name = re.sub(r"[|_\-\s]+$", "", name)
    # Remove trailing OCR artefacts like "|*", "—|*", "=k"
    name = re.sub(r"[\u2014\u2013][-|*]+\s*$", "", name)
    name = re.sub(r"\s*[|*=]+\s*$", "", name)
    name = re.sub(r"\s*—\s*[|*]+\s*$", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _detect_year_from_filename(filename: str) -> Optional[int]:
    """Extract year of study from a calendar file name.

    e.g. ``B.Tech-I-Year-...`` → 1, ``B.Tech-IV-Year-...`` → 4
    """
    m = re.search(r"B\.?Tech[-_]([IVX]+)[-_]Year", filename, re.IGNORECASE)
    if m:
        from app.ingestion.metadata import _resolve_token
        return _resolve_token(m.group(1))
    m = re.search(r"B\.?Tech[-_](\d)[-_]Year", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# OCR common substitutions for roman numerals in calendar text
_OCR_ROMAN_MAP = {"|": "I", "Il": "II", "IlI": "III", "lV": "IV"}


def _fix_ocr_roman(text: str) -> str:
    """Fix common OCR misreads of roman numerals."""
    for bad, good in _OCR_ROMAN_MAP.items():
        text = text.replace(bad, good)
    return text


def chunk_calendar(
    doc: DocumentContent,
    cleaned_pages: list[str],
) -> list[Chunk]:
    """Chunk a calendar document into event and summary chunks.

    Parses messy OCR text to extract individual events with dates.
    """
    chunks: list[Chunk] = []
    full_text = "\n".join(cleaned_pages)

    # Detect academic year
    academic_year = detect_academic_year(full_text)

    # Detect year of study: try text first, fall back to file name
    year_of_study = detect_year(full_text)
    if not year_of_study:
        year_of_study = _detect_year_from_filename(doc.source_file)

    # Semester marker (tolerant of OCR artefacts like "Il", "|")
    sem_marker_re = re.compile(
        r"([IVXIl|]+|\d)\s*[-–—]\s*SEMESTER", re.IGNORECASE
    )

    # Header lines to skip (case-insensitive substrings)
    skip_substrings = [
        "ACADEMIC CALENDAR", "VNRVJIET", "VIGNANA", "VIGNAN",
        "JYOTHI", "AUTONOMOUS", "CBCS", "APPLICABLE TO",
        "EVENT / ACTIVITY", "REGULATIONS", "ENGINEERING",
        "TECHNOLOGY", "INSTITUTE", "ADMITTED BATCH", "PROGRAMMES",
        "VALLURUPALLI", "NAGESWARA", "RAO",
    ]

    current_sem: Optional[int] = None
    lines = full_text.split("\n")
    current_event_name: Optional[str] = None
    date_buffer: list[str] = []

    def flush_event() -> None:
        """Create a chunk from the accumulated event name and dates."""
        nonlocal current_event_name, date_buffer
        if not current_event_name:
            date_buffer = []
            return

        date_text = " ".join(date_buffer).strip()
        date_range = parse_date_range(date_text) if date_text else None

        y, s = _calendar_year_sem()
        etype = classify_event_type(current_event_name)

        event_text = current_event_name
        if date_text:
            event_text += f" | Dates: {date_text}"
        if date_range:
            start, end = date_range
            event_text += f" | Start: {start} | End: {end}"

        chunks.append(
            Chunk(
                id=_make_id(doc.source_file, event_text),
                text=event_text,
                doc_type="calendar",
                chunk_kind="event",
                year_of_study=y,
                semester_in_year=s,
                overall_semester=current_sem,
                event_type=etype,
                start_date=date_range[0] if date_range else None,
                end_date=date_range[1] if date_range else None,
                academic_year=academic_year,
                source_file=doc.source_file,
                page_start=1,
                page_end=doc.total_pages,
            )
        )

        current_event_name = None
        date_buffer = []

    def _calendar_year_sem() -> tuple[Optional[int], Optional[int]]:
        if current_sem:
            y, s = overall_semester_to_year_sem(current_sem)
            return y, s
        return year_of_study, None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Fix OCR roman numerals for semester detection
        fixed = _fix_ocr_roman(stripped)

        # Check for semester marker
        sem_match = sem_marker_re.search(fixed)
        if sem_match:
            token = sem_match.group(1)
            from app.ingestion.metadata import _resolve_token

            val = _resolve_token(token)
            if val and 1 <= val <= 8:
                flush_event()
                current_sem = val
                continue

        # Skip header / boilerplate lines
        upper = stripped.upper()
        if any(s in upper for s in skip_substrings):
            continue
        if "ACADEMIC YEAR" in upper:
            continue
        if "B.TECH" in upper and "YEAR" in upper:
            continue
        # Skip very short lines (likely noise)
        if len(stripped) < 4:
            continue

        # Is this a date line?
        if _is_date_line(stripped):
            date_buffer.append(stripped)
        else:
            # This is an event-name line
            if _is_new_event(stripped):
                # A new event starts → flush the previous one
                if current_event_name:
                    flush_event()
                current_event_name = _extract_event_name(stripped)
            else:
                # Continuation of the current event name (multi-line)
                if current_event_name:
                    current_event_name += " " + _extract_event_name(stripped)
                else:
                    current_event_name = _extract_event_name(stripped)

    flush_event()

    # Create a semester summary chunk
    if chunks:
        summary_lines = [f"Calendar Summary (Year {year_of_study})"]
        for ch in chunks:
            summary_lines.append(f"  {ch.text}")
        summary_text = "\n".join(summary_lines)
        chunks.append(
            Chunk(
                id=_make_id(doc.source_file, summary_text),
                text=summary_text,
                doc_type="calendar",
                chunk_kind="calendar_summary",
                year_of_study=year_of_study,
                academic_year=academic_year,
                source_file=doc.source_file,
                page_start=1,
                page_end=doc.total_pages,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Top-level chunking
# ---------------------------------------------------------------------------
def chunk_document(
    doc: DocumentContent,
    cleaned_pages: list[str],
) -> list[Chunk]:
    """Chunk a single document based on its type."""
    if doc.doc_type == "syllabus":
        return chunk_syllabus(doc, cleaned_pages)
    elif doc.doc_type == "calendar":
        return chunk_calendar(doc, cleaned_pages)
    else:
        return []
