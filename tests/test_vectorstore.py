"""Unit tests for the ChromaStore (uses temp DB, Ollama mocked)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ingestion.chunkers import Chunk
from app.vectorstore.base import SearchResult
from app.vectorstore.chroma_store import ChromaStore, _chunk_to_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(
    text="sample text",
    source_file="test.pdf",
    doc_type="syllabus",
    chunk_kind="course",
    year=1,
    semester=1,
    code=None,
    title=None,
) -> Chunk:
    return Chunk(
        id=f"{source_file}_{abs(hash(text))}",
        text=text,
        doc_type=doc_type,
        chunk_kind=chunk_kind,
        year_of_study=year,
        semester_in_year=semester,
        overall_semester=semester,
        course_code=code,
        course_title=title,
        source_file=source_file,
        page_start=1,
        page_end=2,
    )


def _fake_embeddings(texts):
    """Deterministic fake embeddings (4-dim) for testing."""
    return [[float(len(t)), float(hash(t) % 100) / 100, 0.0, 1.0] for t in texts]


@pytest.fixture
def store():
    """Create a ChromaStore in a temp directory with mocked embeddings."""
    tmp = tempfile.mkdtemp()
    with patch("app.vectorstore.chroma_store.embed_texts", side_effect=_fake_embeddings):
        s = ChromaStore(path=tmp, collection_name="test")
        yield s
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# _chunk_to_metadata
# ---------------------------------------------------------------------------
class TestChunkToMetadata:
    def test_drops_none(self):
        c = _make_chunk()
        meta = _chunk_to_metadata(c)
        assert "course_code" not in meta  # was None
        assert meta["source_file"] == "test.pdf"
        assert meta["year_of_study"] == 1

    def test_keeps_non_none(self):
        c = _make_chunk(code="22PC1IN401", title="Test Course")
        meta = _chunk_to_metadata(c)
        assert meta["course_code"] == "22PC1IN401"
        assert meta["course_title"] == "Test Course"


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------
class TestChromaStore:
    def test_add_and_count(self, store):
        chunks = [_make_chunk(text=f"chunk {i}", year=i + 1) for i in range(3)]
        store.add(chunks)
        assert store.count() == 3

    def test_add_empty(self, store):
        store.add([])
        assert store.count() == 0

    def test_search(self, store):
        chunks = [
            _make_chunk(text="embedded systems unit 1", source_file="syl.pdf"),
            _make_chunk(text="data structures and algorithms", source_file="syl.pdf"),
            _make_chunk(text="commencement of classes august", source_file="cal.pdf"),
        ]
        store.add(chunks)
        # Search with an embedding (any 4-dim vector)
        results = store.search([10.0, 0.5, 0.0, 1.0], top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_search_with_filters(self, store):
        chunks = [
            _make_chunk(text="syllabus text", source_file="s.pdf", doc_type="syllabus", year=1),
            _make_chunk(text="calendar text", source_file="c.pdf", doc_type="calendar", year=1),
        ]
        store.add(chunks)
        results = store.search([5.0, 0.3, 0.0, 1.0], top_k=5, filters={"doc_type": "syllabus"})
        assert all(r.metadata.get("doc_type") == "syllabus" for r in results)

    def test_get_all_texts(self, store):
        chunks = [
            _make_chunk(text="alpha", source_file="a.pdf"),
            _make_chunk(text="beta", source_file="b.pdf"),
        ]
        store.add(chunks)
        texts = store.get_all_texts()
        assert "alpha" in texts
        assert "beta" in texts

    def test_get_all_texts_with_filters(self, store):
        chunks = [
            _make_chunk(text="syllabus content", doc_type="syllabus"),
            _make_chunk(text="calendar content", doc_type="calendar"),
        ]
        store.add(chunks)
        texts = store.get_all_texts(filters={"doc_type": "calendar"})
        assert "calendar content" in texts
        assert "syllabus content" not in texts

    def test_delete_by_source(self, store):
        chunks = [
            _make_chunk(text="chunk A", source_file="keep.pdf"),
            _make_chunk(text="chunk B", source_file="delete.pdf"),
            _make_chunk(text="chunk C", source_file="delete.pdf"),
        ]
        store.add(chunks)
        assert store.count() == 3
        store.delete_by_source("delete.pdf")
        assert store.count() == 1
        texts = store.get_all_texts()
        assert "chunk A" in texts
        assert "chunk B" not in texts

    def test_idempotent_re_add(self, store):
        """Re-adding chunks from the same file replaces old ones."""
        chunks_v1 = [_make_chunk(text="version 1", source_file="doc.pdf")]
        store.add(chunks_v1)
        assert store.count() == 1

        # Delete old and add new
        store.delete_by_source("doc.pdf")
        chunks_v2 = [_make_chunk(text="version 2", source_file="doc.pdf")]
        store.add(chunks_v2)
        assert store.count() == 1
        texts = store.get_all_texts()
        assert "version 2" in texts
        assert "version 1" not in texts
