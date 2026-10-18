"""Full ingestion pipeline: load → clean → chunk → embed → store.

Idempotent: re-ingesting a file replaces its old chunks.  Shows a
progress bar and prints a summary of chunks per doc_type and per year.

Run with::

    python -m app.ingestion.pipeline
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from app.config import DOCUMENTS_DIR
from app.ingestion.chunkers import chunk_document
from app.ingestion.cleaner import clean_document
from app.ingestion.loader import DocumentContent, load_all_documents
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def run_ingestion(
    documents_dir: Path = DOCUMENTS_DIR,
    store: Optional[ChromaStore] = None,
) -> dict:
    """Run the full ingestion pipeline.

    Parameters
    ----------
    documents_dir:
        Root directory containing the ``syllabus/`` and
        ``academic_calendar/`` sub-folders.
    store:
        An existing ChromaStore, or *None* to create one.

    Returns
    -------
    dict
        Summary with keys ``total_chunks``, ``by_doc_type``,
        ``by_year``, ``by_source_file``.
    """
    if store is None:
        store = ChromaStore()

    documents_dir = Path(documents_dir)
    print(f"\nLoading PDFs from {documents_dir} …")
    documents = load_all_documents(documents_dir)

    if not documents:
        print("No PDFs found.  Put PDFs under documents/syllabus/ and")
        print("documents/academic_calendar/, then re-run.")
        return {"total_chunks": 0, "by_doc_type": {}, "by_year": {}, "by_source_file": {}}

    by_doc_type: Counter[str] = Counter()
    by_year: Counter[Optional[int]] = Counter()
    by_source_file: Counter[str] = Counter()
    total_chunks = 0

    print(f"\nIngesting {len(documents)} document(s) …\n")

    for doc in tqdm(documents, desc="Ingesting", unit="file"):
        # 1. Clean
        cleaned_pages = clean_document([p.text for p in doc.pages])

        # 2. Chunk
        chunks = chunk_document(doc, cleaned_pages)
        if not chunks:
            logger.warning("No chunks produced from %s", doc.source_file)
            continue

        # 3. Delete old chunks for this file (idempotent)
        store.delete_by_source(doc.source_file)

        # 4. Embed + store
        store.add(chunks)

        # 5. Update summary counters
        total_chunks += len(chunks)
        for c in chunks:
            by_doc_type[c.doc_type] += 1
            by_year[c.year_of_study] += 1
            by_source_file[c.source_file] += 1

    # ---- Summary ----
    summary = {
        "total_chunks": total_chunks,
        "by_doc_type": dict(by_doc_type),
        "by_year": {k: v for k, v in sorted(by_year.items(), key=lambda x: (x[0] or 0))},
        "by_source_file": dict(by_source_file),
    }

    print(f"\n{'=' * 60}")
    print("INGESTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total chunks in store : {store.count()}")
    print(f"\n  By document type:")
    for dtype, count in sorted(by_doc_type.items()):
        print(f"    {dtype:<15} {count:>5}")
    print(f"\n  By year of study:")
    for year, count in sorted(summary["by_year"].items(), key=lambda x: x[0] or 0):
        label = f"Year {year}" if year else "Unspecified"
        print(f"    {label:<15} {count:>5}")
    print(f"\n  By source file:")
    for fname, count in sorted(by_source_file.items()):
        print(f"    {fname:<55} {count:>5}")
    print(f"{'=' * 60}\n")

    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_ingestion()
    return 0


if __name__ == "__main__":
    sys.exit(main())
