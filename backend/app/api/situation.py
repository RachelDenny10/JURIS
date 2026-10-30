"""API endpoint for evidence-grounded consumer situation analysis."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.situation import SituationRequest, SituationResponse
from app.services.situation_service import (
    SituationAnalysisError,
    analyze_situation,
)

logger = logging.getLogger("juris.api.situation")

router = APIRouter(prefix="/situation", tags=["situation"])


@router.post("/analyze", response_model=SituationResponse)
def analyze_consumer_situation(request: SituationRequest):
    """Assess a consumer situation using evidence from indexed legal documents."""
    try:
        result = analyze_situation(
            situation=request.situation,
            top_k=request.top_k,
        )
        return SituationResponse(**result)
    except SituationAnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Unexpected error analyzing consumer situation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not analyze the situation. Please try again.",
        )
