"""
Analysis API.

Phase 3 scope: PDF text extraction only. Preprocessing, TF-IDF,
TextRank, and entity extraction are added in later phases (4, 5, 6, 9).
"""

import logging
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import get_settings
from app.schemas.document import (
    ChiSquareFeature,
    ChiSquareResponse,
    ExtractionResponse,
    PreprocessingResponse,
    TfidfFitResponse,
    TfidfKeyword,
    TfidfKeywordsResponse,
    TfidfStatusResponse,
)
from app.services import (
    chi_square_service,
    document_service,
    pdf_extraction_service,
    preprocessing_service,
    tfidf_service,
)

logger = logging.getLogger("juris.api.analysis")

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/extract/{document_id}", response_model=ExtractionResponse)
def extract_document_text(document_id: str):
    """
    Extract text from a previously uploaded PDF and persist it under
    data/processed/. Uses the PDF extraction service, including OCR
    fallback when configured.
    """
    settings = get_settings()

    try:
        record = document_service.get_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    pdf_path = settings.project_root / record.storage_path

    # IMPORTANT:
    # Define the output path BEFORE doing extraction.
    relative_text_path = Path("data") / "processed" / f"{document_id}.txt"
    absolute_text_path = settings.project_root / relative_text_path

    try:
        result = pdf_extraction_service.extract_text_from_pdf(pdf_path)

    except pdf_extraction_service.PdfExtractionError as exc:
        logger.warning(
            "Extraction failed for document %s: %s",
            document_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "Unexpected error extracting text for document %s",
            document_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not extract text from the document.",
        )

    # Save extracted/OCR text if available
    if not result.is_empty:

        pdf_extraction_service.save_extracted_text(
            text=result.full_text,
            output_path=absolute_text_path,
        )

        new_status = "text_extracted"

        if result.ocr_used:
            message = (
                f"Extracted {result.character_count} characters "
                f"from {result.page_count} page(s) using OCR fallback."
            )
        else:
            message = (
                f"Extracted {result.character_count} characters "
                f"from {result.page_count} page(s)."
            )

    else:

        new_status = "extraction_empty"

        if result.ocr_used:
            message = (
                "OCR was attempted but produced no usable text. "
                "The document may be unreadable, extremely low quality, "
                "or blank."
            )
        else:
            message = (
                "The PDF opened successfully but no extractable text "
                "was found."
            )

    updated = document_service.update_document_fields(
        document_id,
        text_path=str(relative_text_path) if not result.is_empty else None,
        processing_status=new_status,
    )

    return ExtractionResponse(
        document_id=document_id,
        page_count=result.page_count,
        character_count=result.character_count,
        is_empty=result.is_empty,
        ocr_used=result.ocr_used,
        text_path=updated.text_path or "",
        processing_status=updated.processing_status,
        message=message,
    )

@router.get("/text/{document_id}")
def get_extracted_text(document_id: str):
    """Return the raw extracted text for a document, if it has been extracted."""
    try:
        record = document_service.get_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if not record.text_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This document has not had text extracted yet. Call /analysis/extract/{document_id} first.",
        )

    settings = get_settings()
    text_file = settings.project_root / record.text_path
    if not text_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recorded text file is missing on disk.",
        )

    return {"document_id": document_id, "text": text_file.read_text(encoding="utf-8")}



@router.post("/preprocess/{document_id}", response_model=PreprocessingResponse)
def preprocess_document(document_id: str):
    """
    Run NLP preprocessing (clean -> tokenize -> stopword removal ->
    lemmatize) on a document's previously extracted text.

    Requires /analysis/extract/{document_id} to have been run first.
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

    raw_text = text_file.read_text(encoding="utf-8")

    try:
        result = preprocessing_service.preprocess_text(raw_text)
    except preprocessing_service.PreprocessingError as exc:
        logger.error("Preprocessing setup error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error preprocessing document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not preprocess the document text.",
        )

    relative_output_path = Path("data") / "processed" / f"{document_id}_preprocessed.json"
    absolute_output_path = settings.project_root / relative_output_path
    absolute_output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "document_id": document_id,
        "cleaned_text": result.cleaned_text,
        "tokens": result.tokens,
        "tokens_no_stopwords": result.tokens_no_stopwords,
        "lemmatized_tokens": result.lemmatized_tokens,
    }
    absolute_output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    updated = document_service.update_document_fields(
        document_id,
        processing_status="preprocessed",
    )

    return PreprocessingResponse(
        document_id=document_id,
        token_count_raw=len(result.tokens),
        token_count_no_stopwords=len(result.tokens_no_stopwords),
        token_count_lemmatized=len(result.lemmatized_tokens),
        sample_tokens=result.lemmatized_tokens[:20],
        preprocessed_path=str(relative_output_path),
        processing_status=updated.processing_status,
    )


@router.get("/preprocessed/{document_id}")
def get_preprocessed_data(document_id: str):
    """Return the full preprocessing output (tokens at each stage) for a document."""
    settings = get_settings()
    output_path = settings.project_root / "data" / "processed" / f"{document_id}_preprocessed.json"

    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No preprocessing output found. Call /analysis/preprocess/{document_id} first.",
        )

    return json.loads(output_path.read_text(encoding="utf-8"))



@router.post("/tfidf/fit", response_model=TfidfFitResponse)
def fit_tfidf():
    """
    Fit a TF-IDF vectorizer across all currently preprocessed
    documents. Must be re-run whenever new documents are preprocessed,
    so that they're included in the corpus-level TF-IDF scoring.
    """
    try:
        result = tfidf_service.fit_corpus_tfidf()
    except tfidf_service.TfidfError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error fitting TF-IDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fit the TF-IDF model.",
        )

    return TfidfFitResponse(
        document_count=result["document_count"],
        vocabulary_size=result["vocabulary_size"],
        message=(
            f"Fitted TF-IDF across {result['document_count']} document(s), "
            f"vocabulary size {result['vocabulary_size']}."
        ),
    )


@router.get("/tfidf/status", response_model=TfidfStatusResponse)
def tfidf_status():
    """Check whether a TF-IDF model is currently fitted, and over which documents."""
    status_info = tfidf_service.get_corpus_status()
    return TfidfStatusResponse(**status_info)


@router.get("/tfidf/{document_id}", response_model=TfidfKeywordsResponse)
def get_tfidf_keywords(document_id: str, top_n: int = Query(default=10, ge=1, le=50)):
    """Return the top-N TF-IDF keywords for a document, from the currently fitted model."""
    try:
        document_service.get_document(document_id)  # validate document exists
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        keywords = tfidf_service.get_top_keywords_for_document(document_id, top_n=top_n)
    except tfidf_service.TfidfError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error computing TF-IDF keywords for %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not compute TF-IDF keywords.",
        )

    return TfidfKeywordsResponse(
        document_id=document_id,
        keywords=[TfidfKeyword(**kw) for kw in keywords],
    )



@router.get("/chi-square", response_model=ChiSquareResponse)
def get_chi_square_features(top_n: int = Query(default=15, ge=1, le=100)):
    """
    Compute Chi-Square feature selection over the currently fitted
    TF-IDF corpus, returning the top-N most discriminative terms per
    category.

    Requires TF-IDF to have been fitted (POST /analysis/tfidf/fit)
    across documents spanning at least 2 categories.
    """
    try:
        result = chi_square_service.compute_chi_square_scores(top_n_per_category=top_n)
    except chi_square_service.ChiSquareError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error computing chi-square features")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not compute chi-square features.",
        )

    warning = None
    if result["is_statistically_weak"]:
        warning = (
            "This corpus is small (fewer than 10 documents, or fewer than 2 per category in places). "
            "Chi-square scores are computed correctly but are statistically unstable at this scale. "
            "Treat these results as illustrative until more labeled documents are added per category."
        )

    return ChiSquareResponse(
        results_per_category={
            category: [ChiSquareFeature(**f) for f in features]
            for category, features in result["results_per_category"].items()
        },
        category_counts=result["category_counts"],
        document_count=result["document_count"],
        is_statistically_weak=result["is_statistically_weak"],
        warning=warning,
    )