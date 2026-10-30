"""Schemas for evidence-grounded consumer situation analysis."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SituationRequest(BaseModel):
    situation: str = Field(..., min_length=10, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("situation")
    @classmethod
    def situation_not_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 10:
            raise ValueError("Describe your situation in at least 10 characters.")
        return value


class SituationSource(BaseModel):
    document_id: str
    title: str
    page_start: int
    page_end: int
    excerpt: str
    relevance_score: Optional[float] = None


class SituationResponse(BaseModel):
    status: str  # "supported", "insufficient_evidence", or "assessment_unavailable"
    assessment: Optional[str] = None
    supporting_points: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    sources: list[SituationSource] = Field(default_factory=list)
    warning: Optional[str] = None
    disclaimer: str = (
        "JURIS provides educational information, not legal advice. "
        "This assessment is limited to the uploaded documents and may not reflect "
        "later amendments or current legal requirements."
    )
