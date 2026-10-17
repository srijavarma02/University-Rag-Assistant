"""PDF loader: extracts text and tables from PDFs with OCR fallback.

Uses PyMuPDF (``fitz``) for text and table extraction.  Pages with very
little selectable text (scanned pages) fall back to Tesseract OCR.

Typical usage::

    from app.ingestion.loader import load_all_documents
    docs = load_all_documents(DOCUMENTS_DIR)
    for doc in docs:
        print(doc.source_file, doc.total_pages, "pages")
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF (alias for backward-compatible API)

logger = logging.getLogger(__name__)

# Pages with fewer than this many characters of selectable text trigger OCR.
OCR_CHAR_THRESHOLD = 30

# DPI for OCR rendering (higher = better accuracy, slower).
OCR_DPI = 300


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PageContent:
    """Text and table data extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    tables: list[list[list[Optional[str]]]]  # [table][row][cell]
    used_ocr: bool = False
    char_count: int = 0


@dataclass
class DocumentContent:
    """All pages extracted from a single PDF file."""

    source_file: str  # file name only, e.g. "I-B.Tech-IOT-R25.pdf"
    source_path: Path
    doc_type: str  # "syllabus" or "calendar"
    pages: list[PageContent] = field(default_factory=list)
    total_pages: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    table_pages: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Table formatting helper
# ---------------------------------------------------------------------------
def format_tables(tables: list[list[list[Optional[str]]]]) -> str:
    """Render extracted tables as pipe-delimited text rows.

    Each table becomes a block of rows separated by ``|``.  Tables are
    separated by a blank line so downstream chunkers can split on them.
    """
    lines: list[str] = []
    for table in tables:
        for row in table:
            cells = [str(c).strip() if c else "" for c in row]
            lines.append(" | ".join(cells))
        lines.append("")  # blank line between tables
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------
def _extract_text_and_tables(page: fitz.Page) -> tuple[str, list]:
    """Extract plain text and any tables from a single PyMuPDF page."""
    text = page.get_text("text")

    tables: list = []
    try:
        table_finder = page.find_tables()
        for table in table_finder:
            extracted = table.extract()  # list[list[str|None]]
            if extracted:
                tables.append(extracted)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Table detection skipped on a page: %s", exc)

    return text, tables


def _tesseract_available() -> bool:
    """Check whether the Tesseract binary is available (cached after first call)."""
    global _TESSERACT_OK
    if _TESSERACT_OK is None:
        import shutil
        from app.config import TESSERACT_CMD

        on_path = shutil.which("tesseract") is not None
        configured = bool(TESSERACT_CMD) and Path(TESSERACT_CMD).is_file()
        _TESSERACT_OK = on_path or configured
        if not _TESSERACT_OK:
            logger.warning(
                "Tesseract is not installed or not on PATH. "
                "Scanned pages will have no text. "
                "Install Tesseract OCR to enable the OCR fallback."
            )
    return _TESSERACT_OK


_TESSERACT_OK: bool | None = None


def _ocr_page(page: fitz.Page) -> str:
    """Render a page to an image and run Tesseract OCR on it."""
    if not _tesseract_available():
        raise RuntimeError("Tesseract is not installed or not on PATH")

    import pytesseract
    from PIL import Image

    from app.config import TESSERACT_CMD

    # Point pytesseract at the binary (required on Windows if not on PATH)
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pixmap = page.get_pixmap(matrix=matrix)
    img_data = pixmap.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    return pytesseract.image_to_string(image)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _detect_doc_type(file_path: Path) -> str:
    """Determine document type from the parent folder name."""
    parent = file_path.parent.name.lower()
    if "calendar" in parent:
        return "calendar"
    if "syllabus" in parent:
        return "syllabus"
    return "unknown"


def load_pdf(file_path: Path) -> DocumentContent:
    """Load a single PDF and extract text, tables, with OCR fallback.

    Parameters
    ----------
    file_path:
        Path to a ``.pdf`` file.

    Returns
    -------
    DocumentContent
        All pages with text, tables, and OCR metadata.
    """
    file_path = Path(file_path)
    doc = DocumentContent(
        source_file=file_path.name,
        source_path=file_path,
        doc_type=_detect_doc_type(file_path),
    )

    pdf = fitz.open(str(file_path))
    doc.total_pages = pdf.page_count

    for i, page in enumerate(pdf):
        page_num = i + 1
        text, tables = _extract_text_and_tables(page)

        used_ocr = False
        if len(text.strip()) < OCR_CHAR_THRESHOLD:
            try:
                text = _ocr_page(page)
                used_ocr = True
                doc.ocr_pages.append(page_num)
                logger.info("OCR used on %s page %d", file_path.name, page_num)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OCR failed on %s page %d: %s", file_path.name, page_num, exc
                )

        if tables:
            doc.table_pages.append(page_num)

        # Merge table text below the body text
        table_text = format_tables(tables)
        if table_text.strip():
            if text.strip():
                text = text.rstrip() + "\n\n" + table_text
            else:
                text = table_text

        doc.pages.append(
            PageContent(
                page_number=page_num,
                text=text,
                tables=tables,
                used_ocr=used_ocr,
                char_count=len(text.strip()),
            )
        )

    pdf.close()
    return doc


def load_all_documents(documents_dir: Path) -> list[DocumentContent]:
    """Walk *documents_dir* recursively and load every ``.pdf`` found.

    Files are returned in sorted order for deterministic processing.
    """
    documents_dir = Path(documents_dir)
    documents: list[DocumentContent] = []

    pdf_paths = sorted(documents_dir.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found under %s", documents_dir)

    for pdf_path in pdf_paths:
        logger.info("Loading %s ...", pdf_path)
        doc = load_pdf(pdf_path)
        documents.append(doc)

    return documents
