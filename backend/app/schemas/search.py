"""
Pydantic schemas for the search API (Phase 12).
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.core.constants import DocumentCategory


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    category: Optional[DocumentCategory] = None
    document_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Query must contain at least 2 non-space characters.")
        return value


class SemanticResult(BaseModel):
    rank: int
    score: float
    document_id: str
    title: str
    category: str
    category_label: str
    chunk_id: int
    page_start: int
    page_end: int
    text: str


class SemanticSearchResponse(BaseModel):
    mode: str = "semantic"
    query: str
    results: list[SemanticResult]
    chunks_searched: int
    index_built_at: str
    warning: Optional[str] = None


class MatchedTerm(BaseModel):
    term: str
    weight: float


class KeywordResult(BaseModel):
    rank: int
    score: float
    document_id: str
    title: str
    category: str
    category_label: str
    matched_terms: list[MatchedTerm]
    snippet: str


class KeywordSearchResponse(BaseModel):
    mode: str = "keyword"
    query: str
    query_terms: list[str]
    terms_not_in_vocabulary: list[str]
    results: list[KeywordResult]
    documents_searched: int
    warning: Optional[str] = None


class IndexedDocument(BaseModel):
    document_id: str
    title: str
    category: str
    chunk_count: int
    embedded_now: bool


class SkippedDocument(BaseModel):
    document_id: str
    title: str
    reason: str


class IndexBuildResponse(BaseModel):
    total_vectors: int
    dimension: int
    model_name: str
    seconds: float
    index_path: str
    documents_indexed: list[IndexedDocument]
    documents_skipped: list[SkippedDocument]


class IndexStatusDocument(BaseModel):
    document_id: str
    title: str
    category: str
    chunks: int


class IndexStatusResponse(BaseModel):
    built: bool
    built_at: Optional[str] = None
    model_name: Optional[str] = None
    dimension: Optional[int] = None
    total_vectors: int
    up_to_date: bool
    model_matches_config: Optional[bool] = None
    documents: list[IndexStatusDocument]
    missing_documents: list[str]
    removed_documents: list[str]