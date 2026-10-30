"""
Pydantic schemas for document metadata, used by the API layer and
the document service.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.core.constants import DocumentCategory


class DocumentMetadata(BaseModel):
    """Full metadata record for a single stored legal document."""

    document_id: str
    filename: str = Field(..., description="Safe, unique filename used on disk.")
    original_filename: str = Field(..., description="Filename as originally uploaded (display only).")
    category: DocumentCategory
    title: str
    source: Optional[str] = None
    date: Optional[str] = None
    storage_path: str = Field(..., description="Path to the stored PDF, relative to the project root.")
    text_path: Optional[str] = Field(default=None, description="Populated in Phase 3.")
    upload_timestamp: datetime
    processing_status: str = Field(
        default="uploaded", description="uploaded | text_extracted | preprocessed | classified"
    )
    file_size_bytes: int


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentMetadata]



class ExtractionResponse(BaseModel):
    """Result of running text extraction on a stored document."""

    document_id: str
    page_count: int
    character_count: int
    is_empty: bool
    ocr_used: bool = False
    text_path: str
    processing_status: str
    message: str



class PreprocessingResponse(BaseModel):
    """Result of running NLP preprocessing on a document's extracted text."""

    document_id: str
    token_count_raw: int
    token_count_no_stopwords: int
    token_count_lemmatized: int
    sample_tokens: list[str] = Field(
        default_factory=list, description="First 20 lemmatized tokens, for a quick sanity check."
    )
    preprocessed_path: str
    processing_status: str



class TfidfFitResponse(BaseModel):
    """Result of fitting the TF-IDF vectorizer across the corpus."""

    document_count: int
    vocabulary_size: int
    message: str


class TfidfKeyword(BaseModel):
    term: str
    score: float


class TfidfKeywordsResponse(BaseModel):
    document_id: str
    keywords: list[TfidfKeyword]


class TfidfStatusResponse(BaseModel):
    fitted: bool
    document_count: int
    document_ids: list[str]


class ChiSquareFeature(BaseModel):
    term: str
    chi2_score: float
    p_value: float


class ChiSquareResponse(BaseModel):
    results_per_category: dict[str, list[ChiSquareFeature]]
    category_counts: dict[str, int]
    document_count: int
    is_statistically_weak: bool
    warning: str | None = None

class TextRankKeyword(BaseModel):
    phrase: str
    score: float


class TextRankResponse(BaseModel):
    document_id: str
    keywords: list[TextRankKeyword]


class KeywordComparisonResponse(BaseModel):
    """Side-by-side comparison of TF-IDF and TextRank keywords for one document."""

    document_id: str
    tfidf_keywords: list[TfidfKeyword]
    textrank_keywords: list[TextRankKeyword]



class ClassificationTrainResponse(BaseModel):
    """Result of training and cross-validating the Logistic Regression classifier."""

    model: str = "Logistic Regression"
    document_count: int
    category_counts: dict[str, int]
    features_selected: int
    cv_folds: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix_labels: list[str]
    confusion_matrix: list[list[int]]
    is_small_corpus: bool
    warning: str | None = None


class ClassificationPredictionResponse(BaseModel):
    """Result of predicting a document's category."""

    document_id: str
    predicted_category: str
    confidence: float
    class_probabilities: dict[str, float]


class ClassificationStatusResponse(BaseModel):
    trained: bool
    categories: list[str]