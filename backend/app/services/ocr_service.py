"""
OCR fallback for scanned/image-only PDF pages.

Uses PyMuPDF to render pages to images (no intermediate files saved
to disk - everything stays in memory), Pillow for light preprocessing,
and pytesseract (Tesseract OCR engine, language 'eng') for text
recognition.

This is a FALLBACK ONLY. Normal PyMuPDF text-layer extraction in
pdf_extraction_service.py always runs first and is always preferred;
OCR is only invoked for pages where that extraction produced little
or no usable text, and only if TESSERACT_CMD points to a real,
existing Tesseract executable.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

from app.core.config import get_settings

logger = logging.getLogger("juris.ocr_service")

OCR_RENDER_DPI = 300
OCR_LANGUAGE = "eng"
OCR_BINARIZATION_THRESHOLD = 200

_tesseract_configured = False


class OcrError(Exception):
    """Raised when OCR cannot run at all (e.g. Tesseract binary not found/configured)."""


class OcrPageError(Exception):
    """Raised when OCR fails for a single page. Callers should catch this,
    log it, and continue with remaining pages rather than aborting."""


def _ensure_tesseract_configured() -> None:
    """
    Point pytesseract at the Tesseract binary from settings.tesseract_cmd,
    verifying it actually exists on disk first. Cached after the first
    successful check.
    """
    global _tesseract_configured
    if _tesseract_configured:
        return

    settings = get_settings()
    tesseract_cmd = settings.tesseract_cmd

    if not tesseract_cmd:
        raise OcrError(
            "TESSERACT_CMD is not set. Add it to backend/.env, e.g. "
            r"TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    tesseract_path = Path(tesseract_cmd)
    if not tesseract_path.exists():
        raise OcrError(
            f"Tesseract executable not found at '{tesseract_cmd}'. "
            f"Check the TESSERACT_CMD setting in .env."
        )

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    _tesseract_configured = True
    logger.info("Tesseract OCR engine configured at %s", tesseract_cmd)


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Light, conservative preprocessing: grayscale, auto-contrast, and a
    mild binarization threshold. Deliberately not aggressive - heavy
    thresholding can destroy thin legal-document fonts and small print.
    """
    grayscale = image.convert("L")
    contrast_enhanced = ImageOps.autocontrast(grayscale, cutoff=1)
    binarized = contrast_enhanced.point(
        lambda pixel: 255 if pixel > OCR_BINARIZATION_THRESHOLD else 0
    )
    return binarized


def ocr_single_page(document: fitz.Document, page_index: int) -> str:
    """
    Render a single PDF page to an in-memory image at OCR_RENDER_DPI
    and run Tesseract OCR on it.

    Raises OcrPageError on any failure for this specific page - the
    caller is expected to catch this, log it, and move on to the next
    page rather than aborting the whole document's extraction.
    """
    _ensure_tesseract_configured()

    try:
        page = document.load_page(page_index)
        zoom = OCR_RENDER_DPI / 72  # PyMuPDF's native resolution is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        preprocessed = _preprocess_image_for_ocr(image)

        text = pytesseract.image_to_string(preprocessed, lang=OCR_LANGUAGE)
        return text.strip()
    except OcrError:
        raise
    except Exception as exc:
        logger.warning("OCR failed for page %d: %s", page_index + 1, exc)
        raise OcrPageError(f"OCR failed for page {page_index + 1}: {exc}") from exc