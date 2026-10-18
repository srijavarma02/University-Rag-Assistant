"""Abstract VectorStore interface.

All vector store backends (ChromaDB, MongoDB Atlas, etc.) implement
this interface so the rest of the app doesn't depend on a specific
backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from app.ingestion.chunkers import Chunk


@dataclass
class SearchResult:
    """A single retrieval result."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Abstract interface for a vector store backend."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add a list of chunks (with embeddings) to the store."""
        ...

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for the most similar chunks to *embedding*.

        Returns at most *top_k* results, optionally filtered by
        metadata in *filters*.
        """
        ...

    @abstractmethod
    def get_all_texts(
        self, filters: Optional[dict[str, Any]] = None
    ) -> list[str]:
        """Return all chunk texts matching *filters* (for BM25 indexing)."""
        ...

    @abstractmethod
    def delete_by_source(self, source_file: str) -> None:
        """Delete every chunk that came from *source_file*."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks in the store."""
        ...
