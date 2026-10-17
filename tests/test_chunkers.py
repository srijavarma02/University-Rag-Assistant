"""Unit tests for metadata detection and chunkers."""
from __future__ import annotations

from app.ingestion.chunkers import (
    Chunk,
    _split_by_units,
    _extract_course_list,
    _detect_year_from_filename,
    chunk_syllabus,
    chunk_calendar,
)
from app.ingestion.loader import DocumentContent, PageContent
from app.ingestion.metadata import (
    classify_event_type,
    clean_ocr_date,
    detect_academic_year,
    detect_course_code,
    detect_course_detail_header,
    detect_course_type,
    detect_semester,
    detect_year,
    overall_semester_to_year_sem,
    parse_date_range,
    year_sem_to_overall,
)


# ===========================================================================
# Metadata: year / semester
# ===========================================================================
class TestDetectYear:
    def test_roman_year(self):
        assert detect_year("B.TECH. IV YEAR") == 4
        assert detect_year("III Year") == 3
        assert detect_year("I YEAR") == 1

    def test_word_year(self):
        assert detect_year("First Year") == 1
        assert detect_year("FOURTH YEAR") == 4

    def test_digit_year(self):
        assert detect_year("3rd Year") == 3
        assert detect_year("2 Year") == 2

    def test_no_year(self):
        assert detect_year("Hello world") is None


class TestDetectSemester:
    def test_roman_semester(self):
        assert detect_semester("VII SEMESTER") == 7
        assert detect_semester("I Semester") == 1

    def test_b_tech_prefix(self):
        assert detect_semester("B.Tech. V Semester") == 5

    def test_digit_semester(self):
        assert detect_semester("5th Semester") == 5
        assert detect_semester("Semester 3") == 3

    def test_no_semester(self):
        assert detect_semester("Some random text") is None


class TestOverallSemesterMapping:
    def test_to_year_sem(self):
        assert overall_semester_to_year_sem(1) == (1, 1)
        assert overall_semester_to_year_sem(2) == (1, 2)
        assert overall_semester_to_year_sem(5) == (3, 1)
        assert overall_semester_to_year_sem(8) == (4, 2)

    def test_to_overall(self):
        assert year_sem_to_overall(1, 1) == 1
        assert year_sem_to_overall(2, 2) == 4
        assert year_sem_to_overall(4, 2) == 8


# ===========================================================================
# Metadata: course codes
# ===========================================================================
class TestCourseCode:
    def test_detect_r22(self):
        assert detect_course_code("22PC1IN401 Embedded GPU") == "22PC1IN401"

    def test_detect_r25(self):
        assert detect_course_code("(25BS1MT101) MATRICES") == "25BS1MT101"

    def test_detail_header(self):
        code, title = detect_course_detail_header("(22PC1AM202) DATA ENGINEERING")
        assert code == "22PC1AM202"
        assert title == "DATA ENGINEERING"

    def test_no_code(self):
        assert detect_course_code("No course here") is None
        assert detect_course_detail_header("Just text") is None


class TestCourseType:
    def test_theory(self):
        assert detect_course_type("22PC1IN401", "Embedded GPU") == "theory"

    def test_lab_from_code(self):
        assert detect_course_type("22PC2IN401", "Embedded GPU Lab") == "lab"

    def test_lab_from_title(self):
        assert detect_course_type("25BS2PH102", "Physics Laboratory") == "lab"

    def test_elective(self):
        assert detect_course_type("22PE1CS402", "Deep Learning") == "elective"

    def test_project(self):
        assert detect_course_type("22PW4IN401", "Major Project") == "project"


# ===========================================================================
# Metadata: academic year and dates
# ===========================================================================
class TestAcademicYear:
    def test_dash(self):
        assert detect_academic_year("Academic Year 2026-2027") == "2026-2027"

    def test_em_dash(self):
        assert detect_academic_year("2026 — 2027") == "2026-2027"

    def test_two_digit_end(self):
        assert detect_academic_year("2025-26") == "2025-2026"

    def test_none(self):
        assert detect_academic_year("no year here") is None


class TestCleanOcrDate:
    def test_removes_superscripts(self):
        assert "°" not in clean_ocr_date("15° August 2026")
        assert '"' not in clean_ocr_date('15" August 2026')

    def test_fixes_O_to_0(self):
        assert clean_ocr_date("O7 October 2026") == "7 October 2026"

    def test_removes_ordinals(self):
        assert clean_ocr_date("15th August 2026") == "15 August 2026"

    def test_strips_leading_zeros(self):
        assert clean_ocr_date("015 August 2026") == "15 August 2026"


class TestParseDateRange:
    def test_single_date(self):
        result = parse_date_range("10 August 2026")
        assert result == ("2026-08-10", "2026-08-10")

    def test_same_month_range(self):
        result = parse_date_range("15-18 September 2026")
        assert result == ("2026-09-15", "2026-09-18")

    def test_cross_month_range(self):
        result = parse_date_range("31 August 2026 - 02 September 2026")
        assert result == ("2026-08-31", "2026-09-02")

    def test_messy_ocr(self):
        result = parse_date_range('015-08" August 2026')
        assert result is not None
        assert result[0] == "2026-08-15"

    def test_no_date(self):
        assert parse_date_range("just some text") is None


class TestClassifyEventType:
    def test_commencement(self):
        assert classify_event_type("Commencement of Classes") == "commencement"

    def test_end_exam(self):
        assert classify_event_type("SEE: Semester End Examination") == "end_exam"
        assert classify_event_type("End Examinations") == "end_exam"

    def test_internal_exam(self):
        assert classify_event_type("CIE-I: Class Assessment") == "internal_exam"
        assert classify_event_type("Sessional Examination") == "internal_exam"

    def test_holiday(self):
        assert classify_event_type("Dasara Holiday") == "holiday"
        assert classify_event_type("Sankranti Vacation") == "holiday"

    def test_vacation(self):
        assert classify_event_type("Summer Vacation") == "vacation"

    def test_last_working_day(self):
        assert classify_event_type("End of Classes") == "last_working_day"
        assert classify_event_type("Last Working Day") == "last_working_day"

    def test_other(self):
        assert classify_event_type("Student Feedback") == "other"


# ===========================================================================
# Chunker helpers
# ===========================================================================
class TestSplitByUnits:
    def test_split(self):
        text = (
            "Intro text\n"
            "UNIT-I: First unit content\n"
            "Some topics\n"
            "UNIT-II: Second unit content\n"
            "More topics\n"
            "UNIT-III: Third unit content\n"
            "End topics"
        )
        pieces = _split_by_units(text, "22PC1IN401", "TEST COURSE")
        assert len(pieces) == 4  # intro + 3 units
        assert "TEST COURSE" in pieces[0]
        assert "UNIT-I" in pieces[1]
        assert "UNIT-II" in pieces[2]
        assert "UNIT-III" in pieces[3]

    def test_no_units(self):
        pieces = _split_by_units("Just some text", "22PC1IN401", "TEST")
        assert len(pieces) == 1


class TestExtractCourseList:
    def test_basic(self):
        page = "22PC1IN401\nEmbedded GPU\n22PC1AM402\nDevOps\n"
        pairs = _extract_course_list(page)
        assert len(pairs) == 2
        assert pairs[0] == ("22PC1IN401", "Embedded GPU")
        assert pairs[1] == ("22PC1AM402", "DevOps")

    def test_empty(self):
        assert _extract_course_list("No courses here") == []


class TestDetectYearFromFilename:
    def test_roman(self):
        assert _detect_year_from_filename("B.Tech-I-Year-Academic-Calendar-2026-2027.pdf") == 1
        assert _detect_year_from_filename("B.Tech-IV-Year-Academic-Calendar-2026-2027.pdf") == 4

    def test_no_year(self):
        assert _detect_year_from_filename("random.pdf") is None


# ===========================================================================
# Integration: chunk_syllabus on sample text
# ===========================================================================
class TestChunkSyllabus:
    def _make_doc(self, pages, filename="test_syllabus.pdf"):
        return DocumentContent(
            source_file=filename,
            source_path=None,  # type: ignore[arg-type]
            doc_type="syllabus",
            pages=[PageContent(page_number=i + 1, text=p, tables=[]) for i, p in enumerate(pages)],
            total_pages=len(pages),
        )

    def test_course_chunk(self):
        pages = [
            "B.TECH. I YEAR\nI SEMESTER\nR25\n"
            "Course Code\nTitle\n25BS1MT101\nMatrices\n",
            "B.Tech. I Semester\n(25BS1MT101) MATRICES AND CALCULUS\n"
            "COURSE OBJECTIVES:\nLearn math\nUNIT-I: Linear Algebra\nTopics A\n"
            "UNIT-II: Calculus\nTopics B\nTEXT BOOKS:\n1. Some Book",
        ]
        doc = self._make_doc(pages)
        chunks = chunk_syllabus(doc, pages)
        # Should have: 1 semester summary + 1 course (or 1 course_unit)
        course_chunks = [c for c in chunks if c.chunk_kind in ("course", "course_unit")]
        assert len(course_chunks) >= 1
        assert any(c.course_code == "25BS1MT101" for c in course_chunks)
        assert any(c.course_title == "MATRICES AND CALCULUS" for c in course_chunks)
        assert any(c.year_of_study == 1 for c in course_chunks)
        assert any(c.overall_semester == 1 for c in course_chunks)

    def test_semester_summary(self):
        pages = [
            "B.TECH. II YEAR\nIII SEMESTER\nR22\n"
            "Course Code\nTitle\n22PC1IN401\nEmbedded GPU\n22PC1AM402\nDevOps\n",
        ]
        doc = self._make_doc(pages)
        chunks = chunk_syllabus(doc, pages)
        summaries = [c for c in chunks if c.chunk_kind == "semester_summary"]
        assert len(summaries) >= 1
        assert "22PC1IN401" in summaries[0].text
        assert "Embedded GPU" in summaries[0].text
        assert summaries[0].year_of_study == 2

    def test_empty_pages(self):
        doc = self._make_doc(["", "", ""])
        chunks = chunk_syllabus(doc, ["", "", ""])
        assert chunks == []


# ===========================================================================
# Integration: chunk_calendar on sample text
# ===========================================================================
class TestChunkCalendar:
    def _make_doc(self, pages, filename="B.Tech-I-Year-Calendar.pdf"):
        return DocumentContent(
            source_file=filename,
            source_path=None,  # type: ignore[arg-type]
            doc_type="calendar",
            pages=[PageContent(page_number=1, text=pages[0], tables=[])],
            total_pages=1,
        )

    def test_event_with_date(self):
        text = (
            "ACADEMIC CALENDAR\nAcademic Year 2026 — 2027\n"
            "B.Tech. I Year\n"
            "I — SEMESTER\n"
            "Commencement of Classes\n10 August 2026\n"
            "CIE-I: Class Assessment\n15-18 September 2026\n"
        )
        doc = self._make_doc([text])
        chunks = chunk_calendar(doc, [text])
        events = [c for c in chunks if c.chunk_kind == "event"]
        assert len(events) >= 2
        assert events[0].academic_year == "2026-2027"
        assert events[0].year_of_study == 1
        # Commencement should have a date
        commencement = [c for c in events if c.event_type == "commencement"]
        assert len(commencement) >= 1
        assert commencement[0].start_date == "2026-08-10"

    def test_summary_chunk(self):
        text = "Commencement of Classes\n10 August 2026\n"
        doc = self._make_doc([text])
        chunks = chunk_calendar(doc, [text])
        summaries = [c for c in chunks if c.chunk_kind == "calendar_summary"]
        assert len(summaries) == 1
