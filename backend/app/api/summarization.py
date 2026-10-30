"""
Summarization API (Phase 13).

  GET  /summarization/status           which LLM (if any) is configured
  POST /summarization/summarize        summarise a document (cached; force=true to redo)
  GET  /summarization/{document_id}    return the saved summary
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.summarization import LLMStatusResponse, SummarizeRequest, SummaryResponse
from app.services import document_service, llm_service, summarization_service

logger = logging.getLogger("juris.api.summarization")

router = APIRouter(prefix="/summarization", tags=["summarization"])


@router.get("/status", response_model=LLMStatusResponse)
def llm_status():
    """Report the configured LLM provider and model (never the API key)."""
    return LLMStatusResponse(**llm_service.provider_status())


@router.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummarizeRequest):
    """
    Summarise a document: short summary, key points, important provisions
    and main concepts, with the source passages and coverage.
    Requires the document's text to be extracted. Uses the configured LLM
    if available and falls back to an extractive summary otherwise.
    """
    try:
        result = summarization_service.summarize_document(request.document_id, force=request.force)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except summarization_service.SummarizationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error summarising document %s", request.document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not summarise the document.",
        )
    return SummaryResponse(**result)


@router.get("/{document_id}", response_model=SummaryResponse)
def get_summary(document_id: str):
    """Return the previously generated summary for a document."""
    try:
        result = summarization_service.get_cached_summary(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except summarization_service.SummarizationError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error reading the summary for %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the summary.",
        )
    return SummaryResponse(**result)