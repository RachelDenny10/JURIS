"""
File-handling utilities for JURIS.

Provides safe filename generation, directory creation,
and PDF upload validation.
"""

import uuid
from pathlib import Path


ALLOWED_EXTENSIONS = {".pdf"}


def ensure_directory(path: Path) -> None:
    """Create a directory and its parents if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_safe_extension(original_filename: str) -> str:
    """Return the lowercase extension of a filename, including the dot."""
    return Path(original_filename).suffix.lower()


def is_allowed_extension(original_filename: str) -> bool:
    """Check whether a filename has an allowed extension."""
    return get_safe_extension(original_filename) in ALLOWED_EXTENSIONS


def looks_like_pdf(file_header: bytes) -> bool:
    """Check whether the file begins with the standard PDF signature."""
    return file_header.startswith(b"%PDF-")


def generate_safe_filename(original_filename: str) -> str:
    """
    Generate a random filename for disk storage.

    Only the validated PDF extension is preserved.
    """
    extension = get_safe_extension(original_filename)

    if extension not in ALLOWED_EXTENSIONS:
        extension = ""

    return f"{uuid.uuid4().hex}{extension}"