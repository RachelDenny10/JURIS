"""
Search API (Phase 12).

  POST /search/keyword        TF-IDF + cosine similarity (classical)
  POST /search/semantic       Sentence Transformers + FAISS (meaning-based)
  POST /search/index/build    (re)build the persistent FAISS index
  GET  /search/index/status   is the index built and up to date?
"""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.search import (
    IndexBuildResponse,
    IndexStatusResponse,
    KeywordSearchResponse,
    SearchRequest,
    SemanticSearchResponse,
)
from app.services import keyword_search_service, preprocessing_service, vector_search_service

logger = logging.getLogger("juris.api.search")

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/index/build", response_model=IndexBuildResponse)
def build_vector_index(
    force: bool = Query(
        default=False, description="Re-embed every document even if its embeddings look current."
    ),
):
    """
    Embed any document that needs it and (re)build the FAISS index.
    This can take several minutes the first time, because every
    document's chunks are embedded on CPU.
    """
    try:
        result = vector_search_service.build_index(force=force)
    except vector_search_service.IndexBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except vector_search_service.IndexNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except vector_search_service.VectorSearchError as exc:
        logger.error("Index build failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error building the vector index")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not build the vector index.",
        )
    return IndexBuildResponse(**result)


@router.get("/index/status", response_model=IndexStatusResponse)
def vector_index_status():
    """Report whether the FAISS index exists, what it covers, and whether it is out of date."""
    try:
        return IndexStatusResponse(**vector_search_service.get_index_status())
    except vector_search_service.VectorSearchError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error reading the vector index status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the index status.",
        )


@router.post("/semantic", response_model=SemanticSearchResponse)
def semantic_search(request: SearchRequest):
    """Meaning-based search over document chunks (Sentence Transformers + FAISS)."""
    try:
        result = vector_search_service.semantic_search(
            request.query,
            top_k=request.top_k,
            category=request.category,
            document_id=request.document_id,
        )
    except vector_search_service.IndexNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except vector_search_service.VectorSearchError as exc:
        logger.error("Semantic search failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error in semantic search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not run the semantic search.",
        )
    return SemanticSearchResponse(**result)


@router.post("/keyword", response_model=KeywordSearchResponse)
def keyword_search(request: SearchRequest):
    """Term-based document search (TF-IDF + cosine similarity)."""
    try:
        result = keyword_search_service.keyword_search(
            request.query,
            top_k=request.top_k,
            category=request.category,
            document_id=request.document_id,
        )
    except keyword_search_service.KeywordSearchError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except preprocessing_service.PreprocessingError as exc:
        logger.error("Keyword search preprocessing setup error: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error in keyword search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not run the keyword search.",
        )
    return KeywordSearchResponse(**result)