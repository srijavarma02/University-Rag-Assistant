"""Inspection script for Stage 3.

Loads all PDFs, chunks them, and prints 10 random chunks so the human
can verify the metadata and text quality by eye.

Run with::

    python -m app.ingestion.inspect_chunks
"""
from __future__ import annotations

import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import DOCUMENTS_DIR
from app.ingestion.chunkers import chunk_document
from app.ingestion.cleaner import clean_document
from app.ingestion.loader import load_all_documents

SEP = "=" * 80
THIN = "-" * 80


def main() -> int:
    print(SEP)
    print("STAGE 3 — CHUNK INSPECTION")
    print(SEP)

    documents = load_all_documents(DOCUMENTS_DIR)
    all_chunks: list = []
    for doc in documents:
        cleaned = clean_document([p.text for p in doc.pages])
        chunks = chunk_document(doc, cleaned)
        all_chunks.extend(chunks)

    by_kind: dict[str, int] = {}
    for c in all_chunks:
        by_kind[c.chunk_kind] = by_kind.get(c.chunk_kind, 0) + 1
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"By kind: {by_kind}\n")

    random.seed(42)
    sample = random.sample(all_chunks, min(10, len(all_chunks)))
    for i, c in enumerate(sample, 1):
        print(f"{THIN}")
        print(f"Chunk {i}/{len(sample)}  [{c.chunk_kind}]")
        print(f"  source_file : {c.source_file}")
        print(f"  year/sem    : {c.year_of_study} / {c.semester_in_year} (overall: {c.overall_semester})")
        print(f"  course_code : {c.course_code}")
        print(f"  course_title: {c.course_title}")
        print(f"  event_type  : {c.event_type}")
        print(f"  start_date  : {c.start_date}")
        print(f"  end_date    : {c.end_date}")
        print(f"  academic_yr : {c.academic_year}")
        print(f"  pages       : {c.page_start}-{c.page_end}")
        print(f"  text ({len(c.text)} chars):")
        # Print first 300 chars of text
        print(f"  {c.text[:300]}")
        print()

    print(THIN)
    print("INSPECTION COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
