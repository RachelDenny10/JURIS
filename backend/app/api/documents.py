"""
Document management API.

Phase 2 scope: upload, list, retrieve, and delete legal document
metadata and their stored PDF files. No text extraction here.
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.constants import DocumentCategory
from app.schemas.document import DocumentListResponse, DocumentMetadata
from app.services import document_service

logger = logging.getLogger("juris.api.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentMetadata, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    category: DocumentCategory = Form(...),
    title: str = Form(...),
    source: Optional[str] = Form(default=None),
    date: Optional[str] = Form(default=None),
):
    """Upload a legal document PDF and register its metadata."""
    settings = get_settings()
    try:
        return document_service.save_uploaded_document(
            file_obj=file.file,
            original_filename=file.filename or "unnamed.pdf",
            category=category,
            title=title,
            source=source,
            date=date,
            max_size_bytes=settings.max_upload_size_bytes,
        )
    except document_service.DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error while uploading document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process the uploaded file.",
        )


@router.get("", response_model=DocumentListResponse)
def list_documents():
    """List all stored documents, most recently uploaded first."""
    try:
        records = document_service.list_documents()
        return DocumentListResponse(total=len(records), documents=records)
    except Exception:
        logger.exception("Unexpected error while listing documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not list documents.",
        )


@router.get("/{document_id}", response_model=DocumentMetadata)
def get_document(document_id: str):
    """Retrieve metadata for a single document by its ID."""
    try:
        return document_service.get_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error while fetching document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve the document.",
        )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str):
    """Delete a document's metadata and its stored PDF file."""
    try:
        document_service.delete_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error while deleting document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete the document.",
        )