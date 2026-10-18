"""Tests for the FastAPI backend (dependencies mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.answer import AnswerResponse, Source


@pytest.fixture
def client():
    """A TestClient with a fresh app and reset store."""
    import app.api as api_module

    api_module._store = None  # reset shared store
    return TestClient(api_module.app)


# ===========================================================================
# /health
# ===========================================================================
class TestHealth:
    @patch("app.api.ollama.list")
    @patch("app.api.get_store")
    def test_healthy(self, mock_get_store, mock_list, client):
        mock_list.return_value = {"models": []}
        store = MagicMock()
        store.count.return_value = 427
        mock_get_store.return_value = store

        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["ollama"] == "ok"
        assert body["chunks_indexed"] == 427

    @patch("app.api.ollama.list")
    @patch("app.api.get_store")
    def test_ollama_down(self, mock_get_store, mock_list, client):
        mock_list.side_effect = ConnectionError("refused")
        store = MagicMock()
        store.count.return_value = 0
        mock_get_store.return_value = store

        resp = client.get("/health")
        assert resp.status_code == 200
        assert "unreachable" in resp.json()["ollama"]


# ===========================================================================
# /ask
# ===========================================================================
class TestAsk:
    @patch("app.api.answer_question")
    @patch("app.api.ollama.list")
    @patch("app.api.get_store")
    def test_ask_success(self, mock_get_store, mock_list, mock_answer, client):
        mock_list.return_value = {"models": []}
        mock_get_store.return_value = MagicMock()
        mock_answer.return_value = AnswerResponse(
            answer="The credits are 3 [test.pdf, p.5].",
            sources=[Source("test.pdf", 5, "syllabus", "snippet", "22PC1IN401", None)],
            debug={"best_score": 0.7},
        )

        resp = client.post("/ask", json={"question": "credits for 22PC1IN401?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "credits are 3" in body["answer"]
        assert len(body["sources"]) == 1
        assert body["sources"][0]["course_code"] == "22PC1IN401"

    @patch("app.api.ollama.list")
    @patch("app.api.get_store")
    def test_ask_ollama_down(self, mock_get_store, mock_list, client):
        mock_list.side_effect = ConnectionError("refused")
        mock_get_store.return_value = MagicMock()

        resp = client.post("/ask", json={"question": "anything"})
        assert resp.status_code == 503
        assert "Ollama" in resp.json()["detail"]

    def test_ask_empty_question(self, client):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422  # pydantic validation

    def test_ask_invalid_year(self, client):
        resp = client.post("/ask", json={"question": "q", "year": 7})
        assert resp.status_code == 422  # year must be 1–4

    @patch("app.api.answer_question")
    @patch("app.api.ollama.list")
    @patch("app.api.get_store")
    def test_ask_with_filters(self, mock_get_store, mock_list, mock_answer, client):
        mock_list.return_value = {"models": []}
        mock_get_store.return_value = MagicMock()
        mock_answer.return_value = AnswerResponse(answer="ok", sources=[], debug={})

        resp = client.post("/ask", json={"question": "subjects", "year": 3, "semester": 1})
        assert resp.status_code == 200
        # Verify filters were passed through
        _, kwargs = mock_answer.call_args
        assert kwargs["sidebar_year"] == 3
        assert kwargs["sidebar_semester"] == 1


# ===========================================================================
# /documents
# ===========================================================================
class TestDocuments:
    @patch("app.api.get_store")
    def test_list(self, mock_get_store, client):
        store = MagicMock()
        store.collection.get.return_value = {
            "metadatas": [
                {"source_file": "a.pdf", "doc_type": "syllabus"},
                {"source_file": "a.pdf", "doc_type": "syllabus"},
                {"source_file": "b.pdf", "doc_type": "calendar"},
            ]
        }
        mock_get_store.return_value = store

        resp = client.get("/documents")
        assert resp.status_code == 200
        body = resp.json()
        by_file = {d["source_file"]: d for d in body}
        assert by_file["a.pdf"]["chunk_count"] == 2
        assert by_file["b.pdf"]["chunk_count"] == 1
        assert by_file["b.pdf"]["doc_type"] == "calendar"


# ===========================================================================
# /ingest
# ===========================================================================
class TestIngest:
    @patch("app.api.run_ingestion")
    @patch("app.api.get_store")
    def test_ingest(self, mock_get_store, mock_run, client):
        mock_get_store.return_value = MagicMock()
        mock_run.return_value = {
            "total_chunks": 427,
            "by_doc_type": {"syllabus": 332, "calendar": 95},
            "by_source_file": {"a.pdf": 41},
            "by_year": {},
        }
        resp = client.post("/ingest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_chunks"] == 427
        assert body["by_doc_type"]["syllabus"] == 332


# ===========================================================================
# /
# ===========================================================================
class TestRoot:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "running" in resp.json()["message"]
