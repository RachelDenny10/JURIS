"""
Pydantic schemas for the summarization API (Phase 13).
"""

from typing import Optional

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=64)
    force: bool = Field(default=False, description="Regenerate even if a saved summary exists.")


class Provision(BaseModel):
    reference: str
    description: str


class SourcePassage(BaseModel):
    chunk_id: int
    page_start: int
    page_end: int
    excerpt: str


class Coverage(BaseModel):
    chunks_used: int
    chunks_total: int
    words_used: int
    words_total_approx: int


class SummaryResponse(BaseModel):
    document_id: str
    title: str
    category: str
    category_label: str
    mode: str  # "llm" or "extractive"
    provider: str
    llm_model: Optional[str] = None
    summary: str
    key_points: list[str]
    important_provisions: list[Provision]
    main_concepts: list[str]
    passages_used: list[SourcePassage]
    coverage: Coverage
    generated_at: str
    cached: bool = False
    warning: Optional[str] = None
    disclaimer: str


class LLMStatusResponse(BaseModel):
    provider: str
    llm_model: Optional[str] = None
    configured: bool
    detail: str