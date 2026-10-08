"""
PDF text extraction service, using PyMuPDF (fitz) as the primary
method, with an OCR fallback (ocr_service.py) for pages where
PyMuPDF's text-layer extraction yields little or no usable text.

Phase 3 scope + OCR extension: extract text from stored legal PDFs,
page by page, handle empty/unreadable/corrupted PDFs gracefully, and
persist the extracted text to data/processed/ as a plain-text file.

OCR is a fallback only - PyMuPDF extraction always runs first. OCR is
triggered only when the ratio of "meaningful text" pages falls below
a configurable threshold (see Settings.ocr_page_char_threshold and
Settings.ocr_trigger_meaningful_ratio), and even then only runs on
the specific pages that lacked usable text - pages that already had
good text are left untouched.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.core.config import get_settings
from app.services import ocr_service

logger = logging.getLogger("juris.pdf_extraction_service")

PAGE_BREAK_MARKER = "\n\n----- PAGE BREAK -----\n\n"


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be opened or read at all."""


class ExtractionResult:
    """Holds the outcome of extracting text from a single PDF."""

    def __init__(
        self,
        *,
        page_count: int,
        character_count: int,
        full_text: str,
        is_empty: bool,
        ocr_used: bool = False,
    ):
        self.page_count = page_count
        self.character_count = character_count
        self.full_text = full_text
        self.is_empty = is_empty
        self.ocr_used = ocr_used


def extract_text_from_pdf(pdf_path: Path) -> ExtractionResult:
    """
    Extract text from every page of a PDF, using PyMuPDF first and
    falling back to OCR (per page, only where needed) if too few
    pages have meaningful extractable text.

    Raises PdfExtractionError if the file cannot be opened at all
    (corrupted, not actually a PDF despite its extension, etc).
    Returns an ExtractionResult with is_empty=True (not an exception)
    if, even after the OCR fallback, no text could be recovered.
    """
    if not pdf_path.exists():
        raise PdfExtractionError(f"File not found: {pdf_path}")

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        raise PdfExtractionError(f"Could not open PDF file: {exc}") from exc

    settings = get_settings()
    char_threshold = settings.ocr_page_char_threshold
    meaningful_ratio_threshold = settings.ocr_trigger_meaningful_ratio

    try:
        page_count = document.page_count
        logger.info("Opened PDF %s: %d page(s)", pdf_path.name, page_count)

        if page_count == 0:
            return ExtractionResult(
                page_count=0, character_count=0, full_text="", is_empty=True, ocr_used=False
            )

        # --- Pass 1: PyMuPDF text-layer extraction, page by page ---
        page_texts: list[str] = []
        for page_index in range(page_count):
            page = document.load_page(page_index)
            page_text = page.get_text("text").strip()
            page_texts.append(page_text)

        meaningful_pages = sum(1 for t in page_texts if len(t) >= char_threshold)
        meaningful_ratio = meaningful_pages / page_count
        pymupdf_char_count = sum(len(t) for t in page_texts)

        logger.info(
            "PyMuPDF extraction: %d total chars, %d/%d pages meaningful "
            "(ratio %.2f, trigger threshold %.2f)",
            pymupdf_char_count, meaningful_pages, page_count,
            meaningful_ratio, meaningful_ratio_threshold,
        )

        needs_ocr = meaningful_ratio < meaningful_ratio_threshold
        ocr_used = False

        # --- Pass 2 (conditional): OCR fallback, only for pages that lack usable text ---
        if needs_ocr:
            logger.info(
                "OCR fallback triggered for %s (meaningful-text ratio %.2f below %.2f)",
                pdf_path.name, meaningful_ratio, meaningful_ratio_threshold,
            )
            try:
                ocr_service._ensure_tesseract_configured()
            except ocr_service.OcrError as exc:
                logger.error(
                    "OCR unavailable, keeping PyMuPDF-only result for %s: %s", pdf_path.name, exc
                )
            else:
                for page_index in range(page_count):
                    if len(page_texts[page_index]) >= char_threshold:
                        continue  # this page already has usable text - don't touch it

                    logger.info("OCR: processing page %d/%d", page_index + 1, page_count)
                    try:
                        ocr_text = ocr_service.ocr_single_page(document, page_index)
                    except ocr_service.OcrPageError as exc:
                        logger.warning("Skipping page %d after OCR failure: %s", page_index + 1, exc)
                        continue

                    if ocr_text:
                        page_texts[page_index] = ocr_text
                        ocr_used = True

        full_text = PAGE_BREAK_MARKER.join(page_texts).strip()
        character_count = len(full_text)

        logger.info(
            "Final result for %s: %d characters, ocr_used=%s",
            pdf_path.name, character_count, ocr_used,
        )

        return ExtractionResult(
            page_count=page_count,
            character_count=character_count,
            full_text=full_text,
            is_empty=character_count == 0,
            ocr_used=ocr_used,
        )
    finally:
        document.close()


def save_extracted_text(*, text: str, output_path: Path) -> None:
    """Write extracted text to disk, creating parent folders if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Saved extracted text to %s (%d characters)", output_path, len(text))