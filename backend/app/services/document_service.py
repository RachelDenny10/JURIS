"""
Document service: manages legal document metadata and PDF storage.

Phase 2 scope: CSV-backed metadata (no database yet - Phase 10 adds
PostgreSQL), local file storage for PDFs under data/raw/<category>/,
and basic CRUD operations used by the documents API.

No text extraction happens here - text_path stays empty and
processing_status stays "uploaded" until Phase 3.
"""

import csv
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Optional

from app.core.config import get_settings
from app.core.constants import CATEGORY_FOLDERS, DocumentCategory
from app.schemas.document import DocumentMetadata
from app.utils.file_utils import (
    ensure_directory,
    generate_safe_filename,
    is_allowed_extension,
    looks_like_pdf,
)

logger = logging.getLogger("juris.document_service")

CSV_FIELDNAMES = [
    "document_id",
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
]


class DocumentValidationError(Exception):
    """Raised when an uploaded file fails validation."""


class DocumentNotFoundError(Exception):
    """Raised when a document_id does not exist in the metadata store."""


def _csv_path() -> Path:
    return get_settings().data_metadata_path / "documents.csv"


def ensure_storage_ready() -> None:
    """
    Create the metadata CSV (with header) and all category folders
    under data/raw/ if they do not already exist. Safe to call every
    startup - it never overwrites an existing CSV.
    """
    settings = get_settings()

    ensure_directory(settings.data_metadata_path)
    for folder_name in CATEGORY_FOLDERS.values():
        ensure_directory(settings.data_raw_path / folder_name)

    csv_path = _csv_path()
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        logger.info("Created new metadata file at %s", csv_path)


def _row_to_metadata(row: dict) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=row["document_id"],
        filename=row["filename"],
        original_filename=row["original_filename"],
        category=DocumentCategory(row["category"]),
        title=row["title"],
        source=row["source"] or None,
        date=row["date"] or None,
        storage_path=row["storage_path"],
        text_path=row["text_path"] or None,
        upload_timestamp=datetime.fromisoformat(row["upload_timestamp"]),
        processing_status=row["processing_status"],
        file_size_bytes=int(row["file_size_bytes"]),
    )


def _row_from_metadata(record: DocumentMetadata) -> dict:
    return {
        "document_id": record.document_id,
        "filename": record.filename,
        "original_filename": record.original_filename,
        "category": record.category.value,
        "title": record.title,
        "source": record.source or "",
        "date": record.date or "",
        "storage_path": record.storage_path,
        "text_path": record.text_path or "",
        "upload_timestamp": record.upload_timestamp.isoformat(),
        "processing_status": record.processing_status,
        "file_size_bytes": record.file_size_bytes,
    }


def list_documents() -> list[DocumentMetadata]:
    """Return all document metadata records, newest first."""
    ensure_storage_ready()
    with _csv_path().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [_row_to_metadata(row) for row in reader]
    records.sort(key=lambda d: d.upload_timestamp, reverse=True)
    return records


def get_document(document_id: str) -> DocumentMetadata:
    """Return a single document's metadata, or raise DocumentNotFoundError."""
    for record in list_documents():
        if record.document_id == document_id:
            return record
    raise DocumentNotFoundError(f"No document found with id {document_id}")


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

    Raises DocumentValidationError on bad input. Never trusts the
    original filename for the on-disk path - a random safe filename
    is generated instead.
    """
    ensure_storage_ready()

    if not is_allowed_extension(original_filename):
        raise DocumentValidationError("Only PDF files are accepted.")

    settings = get_settings()
    category_folder = CATEGORY_FOLDERS[category.value]
    target_dir = settings.data_raw_path / category_folder
    ensure_directory(target_dir)

    safe_filename = generate_safe_filename(original_filename)
    destination = target_dir / safe_filename

    bytes_written = 0
    header_checked = False
    with destination.open("wb") as out_file:
        while chunk := file_obj.read(1024 * 1024):
            if not header_checked:
                if not looks_like_pdf(chunk[:5]):
                    out_file.close()
                    destination.unlink(missing_ok=True)
                    raise DocumentValidationError("File content does not look like a valid PDF.")
                header_checked = True

            bytes_written += len(chunk)
            if bytes_written > max_size_bytes:
                out_file.close()
                destination.unlink(missing_ok=True)
                raise DocumentValidationError(
                    f"File exceeds the {max_size_bytes // (1024 * 1024)} MB size limit."
                )
            out_file.write(chunk)

    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise DocumentValidationError("Uploaded file is empty.")

    record = DocumentMetadata(
        document_id=uuid.uuid4().hex,
        filename=safe_filename,
        original_filename=original_filename,
        category=category,
        title=title,
        source=source,
        date=date,
        storage_path=str(destination.relative_to(settings.project_root)),
        text_path=None,
        upload_timestamp=datetime.now(timezone.utc),
        processing_status="uploaded",
        file_size_bytes=bytes_written,
    )

    with _csv_path().open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(_row_from_metadata(record))

    logger.info("Stored document %s (%s) at %s", record.document_id, title, record.storage_path)
    return record


def delete_document(document_id: str) -> None:
    """Remove a document's metadata row and its stored PDF file."""
    settings = get_settings()
    records = list_documents()
    target = next((r for r in records if r.document_id == document_id), None)

    if target is None:
        raise DocumentNotFoundError(f"No document found with id {document_id}")

    file_path = settings.project_root / target.storage_path
    if file_path.exists():
        file_path.unlink()
    # Also remove any derived processed artifacts (extracted text,
    # preprocessed JSON) so services like tfidf_service that scan
    # data/processed/ directly don't pick up orphaned documents.
    processed_dir = settings.project_root / "data" / "processed"
    for pattern in (f"{document_id}.txt", f"{document_id}_preprocessed.json"):
        orphan_path = processed_dir / pattern
        if orphan_path.exists():
            orphan_path.unlink()
            logger.info("Removed orphaned processed file %s", orphan_path)
            
    remaining = [r for r in records if r.document_id != document_id]
    with _csv_path().open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in remaining:
            writer.writerow(_row_from_metadata(record))

    logger.info("Deleted document %s", document_id)



def update_document_fields(document_id: str, **fields) -> DocumentMetadata:
    """
    Update one or more fields on an existing document's metadata row
    (e.g. text_path, processing_status) and rewrite the CSV.

    Only fields already defined on DocumentMetadata may be updated.
    """
    records = list_documents()
    target = None
    updated_records = []

    for record in records:
        if record.document_id == document_id:
            data = record.model_dump()
            data.update(fields)
            target = DocumentMetadata(**data)
            updated_records.append(target)
        else:
            updated_records.append(record)

    if target is None:
        raise DocumentNotFoundError(f"No document found with id {document_id}")

    with _csv_path().open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in updated_records:
            writer.writerow(_row_from_metadata(record))

    logger.info("Updated document %s fields: %s", document_id, list(fields.keys()))
    return target