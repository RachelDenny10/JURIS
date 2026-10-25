"""
Pydantic schemas for the legal entity extraction API (Phase 9).
"""

from pydantic import BaseModel


class EntityCount(BaseModel):
    text: str
    count: int


class EntityExtractionResponse(BaseModel):
    """Entities found in a document, grouped by label with mention counts."""

    document_id: str
    total_mentions: int
    mention_counts: dict[str, int]
    unique_counts: dict[str, int]
    entities: dict[str, list[EntityCount]]
    entities_path: str