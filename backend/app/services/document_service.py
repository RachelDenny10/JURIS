"""
Document service: manages legal document metadata and PDF storage.

Phase 10: metadata now lives in a database table (PostgreSQL, or a
SQLite fallback) instead of documents.csv. The public functions and
return types are unchanged, so the rest of the backend is unaffected.

PDFs are still stored on disk under data/raw/<category_folder>/.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import BinaryIO, Optional

from sqlalchemy import select

from app.core.config import get_settings
from app.core.constants import CATEGORY_FOLDERS, DocumentCategory
from app.core.database import get_session, init_db
from app.models.document import DocumentRecord
from app.schemas.document import DocumentMetadata
from app.utils.file_utils import (
    ensure_directory,
    generate_safe_filename,
    is_allowed_extension,
    looks_like_pdf,
)

logger = logging.getLogger("juris.document_service")

UPDATABLE_FIELDS = {
    "filename",
    "original_filename",
    "category",
    "title",
    "source",
    "date",
    "storage_path",
    "text_path",
    "upload_timestamp",
    "processing_status",
    "file_size_bytes",
}


class DocumentValidationError(Exception):
    """Raised when an uploaded file fails validation."""


class DocumentNotFoundError(Exception):
    """Raised when a document_id does not exist in the database."""


def ensure_storage_ready() -> None:
    """Create the category folders and the database tables if missing."""
    settings = get_settings()
    ensure_directory(settings.data_metadata_path)
    for folder_name in CATEGORY_FOLDERS.values():
        ensure_directory(settings.data_raw_path / folder_name)
    init_db()


def _to_schema(row: DocumentRecord) -> DocumentMetadata:
    timestamp = row.upload_timestamp
    if timestamp.tzinfo is None:  # SQLite drops timezone information
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return DocumentMetadata(
        document_id=row.document_id,
        filename=row.filename,
        original_filename=row.original_filename,
        category=DocumentCategory(row.category),
        title=row.title,
        source=row.source or None,
        date=row.date or None,
        storage_path=row.storage_path,
        text_path=row.text_path or None,
        upload_timestamp=timestamp,
        processing_status=row.processing_status,
        file_size_bytes=int(row.file_size_bytes),
    )


def list_documents() -> list[DocumentMetadata]:
    """Return all document metadata records, newest first."""
    init_db()
    with get_session() as session:
        rows = session.scalars(
            select(DocumentRecord).order_by(DocumentRecord.upload_timestamp.desc())
        ).all()
        return [_to_schema(row) for row in rows]


def get_document(document_id: str) -> DocumentMetadata:
    """Return a single document's metadata, or raise DocumentNotFoundError."""
    init_db()
    with get_session() as session:
        row = session.get(DocumentRecord, document_id)
        if row is None:
            raise DocumentNotFoundError(f"No document found with id {document_id}")
        return _to_schema(row)


def save_uploaded_document(
    *,
    file_obj: BinaryIO,
    original_filename: str,
    category: DocumentCategory,
    title: str,
    source: Optional[str],
    date: Optional[str],
    max_size_bytes: int,
) -> DocumentMetadata:
    """
    Validate, store, and register a newly uploaded PDF under
    data/raw/<category_folder>/.

    Raises DocumentValidationError on bad input. The on-disk filename is
    always a random safe name, never the uploaded one. If the database
    insert fails, the stored PDF is removed again.
    """
    ensure_storage_ready()

    if not is_allowed_extension(original_filename):
        raise DocumentValidationError("Only PDF files are accepted.")

    settings = get_settings()
    target_dir = settings.data_raw_path / CATEGORY_FOLDERS[category.value]
    ensure_directory(target_dir)

    safe_filename = generate_safe_filename(original_filename)
    destination = target_dir / safe_filename

    bytes_written = 0
    header_checked = False
    error: Optional[str] = None

    with destination.open("wb") as out_file:
        while chunk := file_obj.read(1024 * 1024):
            if not header_checked:
                if not looks_like_pdf(chunk[:5]):
                    error = "File content does not look like a valid PDF."
                    break
                header_checked = True

            bytes_written += len(chunk)
            if bytes_written > max_size_bytes:
                error = f"File exceeds the {max_size_bytes // (1024 * 1024)} MB size limit."
                break
            out_file.write(chunk)

    if error is None and bytes_written == 0:
        error = "Uploaded file is empty."

    if error is not None:
        destination.unlink(missing_ok=True)
        raise DocumentValidationError(error)

    record = DocumentRecord(
        document_id=uuid.uuid4().hex,
        filename=safe_filename,
        original_filename=original_filename,
        category=category.value,
        title=title,
        source=source,
        date=date,
        storage_path=str(destination.relative_to(settings.project_root)),
        text_path=None,
        upload_timestamp=datetime.now(timezone.utc),
        processing_status="uploaded",
        file_size_bytes=bytes_written,
    )

    try:
        with get_session() as session:
            session.add(record)
            session.flush()
            result = _to_schema(record)
    except Exception:
        destination.unlink(missing_ok=True)  # do not leave an orphan PDF behind
        logger.exception("Database insert failed for upload %s", original_filename)
        raise

    logger.info("Stored document %s (%s) at %s", result.document_id, title, result.storage_path)
    return result


def update_document_fields(document_id: str, **fields) -> DocumentMetadata:
    """
    Update one or more fields on an existing document (for example
    text_path or processing_status). Unknown field names raise ValueError.
    """
    unknown = set(fields) - UPDATABLE_FIELDS
    if unknown:
        raise ValueError(f"Cannot update unknown document field(s): {sorted(unknown)}")

    init_db()
    with get_session() as session:
        row = session.get(DocumentRecord, document_id)
        if row is None:
            raise DocumentNotFoundError(f"No document found with id {document_id}")

        for name, value in fields.items():
            if isinstance(value, DocumentCategory):
                value = value.value
            setattr(row, name, value)

        session.flush()
        result = _to_schema(row)

    logger.info("Updated document %s fields: %s", document_id, list(fields.keys()))
    return result


def delete_document(document_id: str) -> None:
    """Remove a document's database row, its PDF, and its derived files."""
    settings = get_settings()
    init_db()

    with get_session() as session:
        row = session.get(DocumentRecord, document_id)
        if row is None:
            raise DocumentNotFoundError(f"No document found with id {document_id}")
        storage_path = row.storage_path
        session.delete(row)

    # Files are removed after the row is gone, so a failed delete never loses metadata.
    pdf_path = settings.project_root / storage_path
    if pdf_path.exists():
        pdf_path.unlink()

    processed_dir = settings.project_root / "data" / "processed"
    for name in (
        f"{document_id}.txt",
        f"{document_id}_preprocessed.json",
        f"{document_id}_entities.json",
    ):
        derived = processed_dir / name
        if derived.exists():
            derived.unlink()
            logger.info("Removed derived file %s", derived)

    logger.info("Deleted document %s", document_id)