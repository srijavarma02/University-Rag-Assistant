"""Tests for prompts, llm, and answer orchestration (Ollama mocked)."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

from app.answer import AnswerResponse, Source, _best_score, _to_sources, answer_question
from app.prompts import (
    NOT_FOUND_REPLY,
    OUT_OF_SCOPE_REPLY,
    build_context,
    build_messages,
)
from app.vectorstore.base import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result(text="sample", score=0.8, **meta_kwargs):
    meta = {
        "source_file": "test.pdf",
        "page_start": 5,
        "doc_type": "syllabus",
    }
    meta.update(meta_kwargs)
    return SearchResult(id="r1", text=text, score=score, metadata=meta)


# ===========================================================================
# Prompts
# ===========================================================================
class TestBuildContext:
    def test_empty(self):
        assert "no relevant context" in build_context([])

    def test_includes_citation(self):
        ctx = build_context([_result(text="Operating systems content")])
        assert "test.pdf" in ctx
        assert "p.5" in ctx
        assert "Operating systems content" in ctx

    def test_includes_event_tag(self):
        ctx = build_context([
            _result(text="exam", doc_type="calendar", event_type="end_exam", _temporal="upcoming")
        ])
        assert "end_exam" in ctx
        assert "upcoming" in ctx


class TestBuildMessages:
    def test_structure(self):
        msgs = build_messages("What are the units?", [_result()], today=date(2026, 9, 1))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_today_in_system(self):
        msgs = build_messages("q", [_result()], today=date(2026, 9, 1))
        assert "2026-09-01" in msgs[0]["content"]

    def test_not_found_phrase_in_system(self):
        msgs = build_messages("q", [_result()])
        assert NOT_FOUND_REPLY in msgs[0]["content"]
        assert OUT_OF_SCOPE_REPLY in msgs[0]["content"]

    def test_question_in_user(self):
        msgs = build_messages("What topics in Unit 3?", [_result()])
        assert "What topics in Unit 3?" in msgs[1]["content"]


# ===========================================================================
# Answer helpers
# ===========================================================================
class TestToSources:
    def test_conversion(self):
        results = [_result(text="A" * 300, course_code="22PC1IN401")]
        sources = _to_sources(results)
        assert len(sources) == 1
        assert sources[0].source_file == "test.pdf"
        assert sources[0].page == 5
        assert sources[0].course_code == "22PC1IN401"
        assert len(sources[0].snippet) <= 200


class TestBestScore:
    def test_uses_vector_score(self):
        r = _result(score=0.02)  # fused score is low
        r.metadata["_vector_score"] = 0.75
        assert _best_score([r]) == 0.75

    def test_fallback_to_score(self):
        r = _result(score=0.6)
        assert _best_score([r]) == 0.6

    def test_empty(self):
        assert _best_score([]) == 0.0


# ===========================================================================
# answer_question (retrieval + LLM both mocked)
# ===========================================================================
class TestAnswerQuestion:
    @patch("app.answer.chat")
    @patch("app.answer.retrieve")
    def test_good_answer(self, mock_retrieve, mock_chat):
        from app.retrieval import QueryFilters, RetrievalResult

        r = _result(text="Operating systems topics", score=0.05)
        r.metadata["_vector_score"] = 0.7
        mock_retrieve.return_value = RetrievalResult(
            results=[r],
            filters=QueryFilters(),
            routed_doc_type="syllabus",
            debug={},
        )
        mock_chat.return_value = "The topics are X and Y [test.pdf, p.5]."

        resp = answer_question("What topics?", store=None, score_threshold=0.3)
        assert isinstance(resp, AnswerResponse)
        assert "topics are X and Y" in resp.answer
        assert len(resp.sources) == 1
        mock_chat.assert_called_once()

    @patch("app.answer.chat")
    @patch("app.answer.retrieve")
    def test_below_threshold_skips_llm(self, mock_retrieve, mock_chat):
        from app.retrieval import QueryFilters, RetrievalResult

        r = _result(text="irrelevant", score=0.01)
        r.metadata["_vector_score"] = 0.1  # below threshold
        mock_retrieve.return_value = RetrievalResult(
            results=[r],
            filters=QueryFilters(),
            routed_doc_type=None,
            debug={},
        )

        resp = answer_question("random question", store=None, score_threshold=0.3)
        assert resp.answer == NOT_FOUND_REPLY
        assert resp.sources == []
        mock_chat.assert_not_called()

    @patch("app.answer.chat")
    @patch("app.answer.retrieve")
    def test_no_results(self, mock_retrieve, mock_chat):
        from app.retrieval import QueryFilters, RetrievalResult

        mock_retrieve.return_value = RetrievalResult(
            results=[],
            filters=QueryFilters(),
            routed_doc_type=None,
            debug={},
        )
        resp = answer_question("nothing", store=None)
        assert resp.answer == NOT_FOUND_REPLY
        mock_chat.assert_not_called()
