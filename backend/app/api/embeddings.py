"""
Embeddings API (Phase 11).

Chunks a document's extracted text, embeds the chunks with a Sentence
Transformer, and provides a within-document similarity test. The
cross-document FAISS index and semantic search endpoints come in Phase 12.
"""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import get_settings
from app.schemas.embeddings import (
    EmbeddingModelInfo,
    EmbedDocumentResponse,
    SimilarChunk,
    SimilaritySearchResponse,
)
from app.services import document_service, embedding_service

logger = logging.getLogger("juris.api.embeddings")

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("/model", response_model=EmbeddingModelInfo)
def embedding_model_info():
    """Return the configured embedding model and its vector dimension (loads the model)."""
    try:
        return EmbeddingModelInfo(**embedding_service.get_model_info())
    except embedding_service.EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/{document_id}", response_model=EmbedDocumentResponse)
def embed_document(document_id: str):
    """
    Chunk and embed a document's extracted text. Requires
    /analysis/extract/{document_id} to have been run first. The first
    call also downloads the embedding model.
    """
    settings = get_settings()

    try:
        record = document_service.get_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if not record.text_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No extracted text found for this document. Call /analysis/extract/{document_id} first.",
        )

    text_file = settings.project_root / record.text_path
    if not text_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recorded text file is missing on disk.",
        )

    try:
        result = embedding_service.embed_document(document_id, text_file.read_text(encoding="utf-8"))
    except embedding_service.EmbeddingError as exc:
        logger.error("Embedding failed for %s: %s", document_id, exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error embedding document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not embed the document.",
        )

    return EmbedDocumentResponse(document_id=document_id, **result)


@router.get("/{document_id}/similar", response_model=SimilaritySearchResponse)
def similar_chunks(
    document_id: str,
    query: str = Query(..., min_length=2, max_length=500),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Find the chunks of ONE document most similar in meaning to a query (a test of the embeddings)."""
    try:
        document_service.get_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        results = embedding_service.similarity_search_in_document(document_id, query, top_k=top_k)
    except embedding_service.EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error in similarity search for %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not run the similarity search.",
        )

    return SimilaritySearchResponse(
        document_id=document_id,
        query=query,
        results=[SimilarChunk(**r) for r in results],
    )