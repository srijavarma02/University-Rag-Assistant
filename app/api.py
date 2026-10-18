"""FastAPI backend for the IoT Department Assistant.

Endpoints:
  POST /ask        — answer a question (optional year/semester filters)
  POST /ingest     — (re-)run the ingestion pipeline
  GET  /health     — service + Ollama health check
  GET  /documents  — list indexed files with chunk counts

Run with::

    uvicorn app.api:app --reload
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.answer import answer_question
from app.config import LLM_MODEL, OLLAMA_URL
from app.ingestion.pipeline import run_ingestion
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IoT Department Assistant API",
    description="RAG chatbot for the IoT syllabus and academic calendar.",
    version="1.0.0",
)

# CORS: allow the Streamlit frontend and local static servers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# A single shared store instance (loaded lazily).
_store: Optional[ChromaStore] = None


def get_store() -> ChromaStore:
    """Return the shared ChromaStore, creating it on first use."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The student's question.")
    year: Optional[int] = Field(None, ge=1, le=4, description="Year of study filter (1–4).")
    semester: Optional[int] = Field(None, ge=1, le=2, description="Semester-in-year filter (1 or 2).")


class SourceModel(BaseModel):
    source_file: str
    page: int
    doc_type: str
    snippet: str
    course_code: Optional[str] = None
    event_type: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    debug: dict


class HealthResponse(BaseModel):
    status: str
    ollama: str
    llm_model: str
    chunks_indexed: int


class DocumentInfo(BaseModel):
    source_file: str
    doc_type: str
    chunk_count: int


class IngestResponse(BaseModel):
    status: str
    total_chunks: int
    by_doc_type: dict
    by_source_file: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service health and whether Ollama is reachable."""
    ollama_status = "ok"
    try:
        ollama.list()
    except Exception as exc:  # noqa: BLE001
        ollama_status = f"unreachable: {exc}"

    try:
        chunks = get_store().count()
    except Exception:  # noqa: BLE001
        chunks = 0

    return HealthResponse(
        status="ok",
        ollama=ollama_status,
        llm_model=LLM_MODEL,
        chunks_indexed=chunks,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the indexed documents."""
    # Fail clearly if Ollama is down.
    try:
        ollama.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not reachable at {OLLAMA_URL}. "
                f"Start the Ollama app or run 'ollama serve'. ({exc})"
            ),
        )

    try:
        result = answer_question(
            request.question,
            store=get_store(),
            sidebar_year=request.year,
            sidebar_semester=request.semester,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error answering question")
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        answer=result.answer,
        sources=[SourceModel(**vars(s)) for s in result.sources],
        debug=result.debug,
    )


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    """List indexed files with their chunk counts."""
    store = get_store()
    try:
        data = store.collection.get(include=["metadatas"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

    counts: Counter[str] = Counter()
    doc_types: dict[str, str] = {}
    for meta in data.get("metadatas") or []:
        if not meta:
            continue
        fname = meta.get("source_file", "unknown")
        counts[fname] += 1
        doc_types[fname] = meta.get("doc_type", "unknown")

    return [
        DocumentInfo(source_file=f, doc_type=doc_types.get(f, "unknown"), chunk_count=c)
        for f, c in sorted(counts.items())
    ]


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """Re-run the ingestion pipeline over the documents folder."""
    try:
        summary = run_ingestion(store=get_store())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        status="ok",
        total_chunks=summary["total_chunks"],
        by_doc_type=summary["by_doc_type"],
        by_source_file=summary["by_source_file"],
    )


@app.get("/")
def root() -> dict:
    """Simple landing endpoint."""
    return {
        "message": "IoT Department Assistant API is running.",
        "endpoints": ["/ask", "/ingest", "/health", "/documents", "/docs"],
    }
