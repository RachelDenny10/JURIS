"""
Pydantic schemas for the embeddings API (Phase 11).
"""

from pydantic import BaseModel


class EmbeddingModelInfo(BaseModel):
    model_name: str
    dimension: int


class EmbedDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    dimension: int
    model_name: str
    average_chunk_words: float


class SimilarChunk(BaseModel):
    chunk_id: int
    score: float
    page_start: int
    page_end: int
    text: str


class SimilaritySearchResponse(BaseModel):
    document_id: str
    query: str
    results: list[SimilarChunk]