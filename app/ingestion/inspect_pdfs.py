"""Inspection script for Stage 2.

Loads every PDF under ``documents/``, cleans the text, prints per-file
stats (pages, characters, tables, OCR pages), then prints sample pages so
the human can confirm the extraction quality.

Run with::

    python -m app.ingestion.inspect_pdfs
"""
from __future__ import annotations

import sys
import textwrap

# Force UTF-8 output so special Unicode characters from PDFs don't crash
# the Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import DOCUMENTS_DIR, SYLLABUS_DIR, CALENDAR_DIR
from app.ingestion.cleaner import clean_document
from app.ingestion.loader import DocumentContent, load_all_documents


SEPARATOR = "=" * 80
THIN_SEP = "-" * 80


def print_stats(doc: DocumentContent, cleaned_pages: list[str]) -> None:
    """Print a one-line summary for a document."""
    total_chars = sum(len(p) for p in cleaned_pages)
    n_tables = len(doc.table_pages)
    ocr_info = f", OCR pages: {doc.ocr_pages}" if doc.ocr_pages else ""
    print(
        f"  {doc.source_file:<55} "
        f"pages: {doc.total_pages:>3}  "
        f"chars: {total_chars:>7}  "
        f"tables: {n_tables:>2}"
        f"{ocr_info}"
    )


def print_sample(
    doc: DocumentContent,
    cleaned_pages: list[str],
    page_indices: list[int],
    label: str,
) -> None:
    """Print the cleaned text of specific pages with a label."""
    print(f"\n{SEPARATOR}")
    print(f"{label}  —  {doc.source_file}")
    print(SEPARATOR)
    for idx in page_indices:
        if 0 <= idx < len(cleaned_pages):
            print(f"\n--- Page {idx + 1} ---")
            text = cleaned_pages[idx].strip()
            if text:
                print(text)
            else:
                print("[empty page]")
    print(THIN_SEP)


def main() -> int:
    print(SEPARATOR)
    print("STAGE 2 — PDF EXTRACTION & CLEANING INSPECTION")
    print(SEPARATOR)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"\nLoading PDFs from {DOCUMENTS_DIR} ...\n")
    documents = load_all_documents(DOCUMENTS_DIR)

    if not documents:
        print("No PDFs found.  Put PDFs under documents/syllabus/ and")
        print("documents/academic_calendar/, then re-run.")
        return 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    print("\nPer-file stats:")
    print(THIN_SEP)
    cleaned_all: list[tuple[DocumentContent, list[str]]] = []
    for doc in documents:
        cleaned_pages = clean_document([p.text for p in doc.pages])
        cleaned_all.append((doc, cleaned_pages))
        print_stats(doc, cleaned_pages)
    print(THIN_SEP)

    # ------------------------------------------------------------------
    # Samples: first 2 pages of every PDF
    # ------------------------------------------------------------------
    for doc, cleaned_pages in cleaned_all:
        indices = list(range(min(2, len(cleaned_pages))))
        if indices:
            print_sample(doc, cleaned_pages, indices, "First pages")

    # ------------------------------------------------------------------
    # Samples: 2 pages from the middle of each syllabus PDF
    # ------------------------------------------------------------------
    syllabus_docs = [(d, p) for d, p in cleaned_all if d.doc_type == "syllabus"]
    for doc, cleaned_pages in syllabus_docs:
        n = len(cleaned_pages)
        if n >= 4:
            mid = n // 2
            indices = sorted({mid - 1, mid})
            # clamp to valid range
            indices = [i for i in indices if 0 <= i < n]
        elif n >= 2:
            indices = [n // 2]
        else:
            indices = [0]
        print_sample(doc, cleaned_pages, indices, "Middle pages (syllabus)")

    # ------------------------------------------------------------------
    # Samples: one full calendar page
    # ------------------------------------------------------------------
    calendar_docs = [(d, p) for d, p in cleaned_all if d.doc_type == "calendar"]
    for doc, cleaned_pages in calendar_docs[:1]:  # just the first calendar file
        n = len(cleaned_pages)
        idx = min(1, n - 1) if n else 0  # page 2 if available, else page 1
        print_sample(doc, cleaned_pages, [idx], "Full calendar page")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("INSPECTION COMPLETE")
    print(SEPARATOR)
    print(
        "\nPlease review the samples above.\n"
        "If the text looks clean and readable, reply 'continue'.\n"
        "If something looks wrong (missing text, garbled OCR, table\n"
        "issues), paste the problem here and I will adjust the parser.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
