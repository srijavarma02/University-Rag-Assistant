"""ChromaDB implementation of the VectorStore interface.

Stores chunk embeddings and metadata in a persistent local ChromaDB
database.  Uses cosine similarity for vector search.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import chromadb

from app.config import CHROMA_PATH
from app.embeddings import embed_texts
from app.ingestion.chunkers import Chunk
from app.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

COLLECTION_NAME = "iot_rag"


def _chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Convert a Chunk's metadata fields to ChromaDB-compatible types.

    ChromaDB metadata values must be ``str``, ``int``, ``float``, or
    ``bool``.  *None* values are omitted so the field can be used in
    ``where`` filters only when it has a value.
    """
    fields: list[tuple[str, Any]] = [
        ("doc_type", chunk.doc_type),
        ("chunk_kind", chunk.chunk_kind),
        ("year_of_study", chunk.year_of_study),
        ("semester_in_year", chunk.semester_in_year),
        ("overall_semester", chunk.overall_semester),
        ("course_code", chunk.course_code),
        ("course_title", chunk.course_title),
        ("event_type", chunk.event_type),
        ("start_date", chunk.start_date),
        ("end_date", chunk.end_date),
        ("academic_year", chunk.academic_year),
        ("source_file", chunk.source_file),
        ("page_start", chunk.page_start),
        ("page_end", chunk.page_end),
    ]
    return {k: v for k, v in fields if v is not None}


class ChromaStore(VectorStore):
    """Persistent ChromaDB vector store."""

    def __init__(
        self,
        path: str = str(CHROMA_PATH),
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # add
    # ------------------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        metadatas = [_chunk_to_metadata(c) for c in chunks]

        # Guarantee unique IDs: two chunks can hash identically if their
        # first 200 chars match (e.g. repeated course-list headers).
        ids: list[str] = []
        seen: dict[str, int] = {}
        for c in chunks:
            base = c.id
            if base in seen:
                seen[base] += 1
                ids.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                ids.append(base)

        # Compute embeddings via Ollama
        logger.info("Embedding %d chunks …", len(chunks))
        embeddings = embed_texts(texts)

        # ChromaDB add (batches of 100 to stay within limits)
        batch = 100
        for start in range(0, len(ids), batch):
            end = start + batch
            self.collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info("Added %d chunks to ChromaDB", len(chunks))

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        search_results: list[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Cosine distance is in [0, 2]; convert to similarity score
            score = 1.0 - distance
            search_results.append(
                SearchResult(
                    id=chunk_id,
                    text=results["documents"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i] or {},
                )
            )
        return search_results

    # ------------------------------------------------------------------
    # get_all_texts
    # ------------------------------------------------------------------
    def get_all_texts(
        self, filters: Optional[dict[str, Any]] = None
    ) -> list[str]:
        kwargs: dict[str, Any] = {}
        if filters:
            kwargs["where"] = filters
        results = self.collection.get(**kwargs)
        if results and results["documents"]:
            return list(results["documents"])
        return []

    # ------------------------------------------------------------------
    # delete_by_source
    # ------------------------------------------------------------------
    def delete_by_source(self, source_file: str) -> None:
        try:
            self.collection.delete(where={"source_file": source_file})
            logger.info("Deleted old chunks from %s", source_file)
        except Exception as exc:  # noqa: BLE001
            logger.debug("delete_by_source('%s'): %s", source_file, exc)

    # ------------------------------------------------------------------
    # count
    # ------------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()
