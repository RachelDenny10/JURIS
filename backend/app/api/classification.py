"""
Classification API.

Phase 8 scope: train/evaluate the Naive Bayes classifier (TF-IDF ->
Chi-Square -> Naive Bayes pipeline) and predict categories for
already-processed documents.
"""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.document import (
    ClassificationPredictionResponse,
    ClassificationStatusResponse,
    ClassificationTrainResponse,
)
from app.services import classification_service, document_service

logger = logging.getLogger("juris.api.classification")

router = APIRouter(prefix="/classification", tags=["classification"])


@router.post("/train", response_model=ClassificationTrainResponse)
def train_classifier(top_k_features: int = Query(default=200, ge=10, le=5000)):
    """
    Train and cross-validate the Naive Bayes classifier using the
    TF-IDF -> Chi-Square -> Naive Bayes pipeline (Section 7).

    Requires TF-IDF to already be fitted (POST /analysis/tfidf/fit)
    across at least 2 categories with at least 2 documents each.
    """
    try:
        result = classification_service.train_and_evaluate(top_k_features=top_k_features)
    except classification_service.ClassificationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error training classifier")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not train the classifier.",
        )

    warning = None
    if result["is_small_corpus"]:
        warning = (
            f"This corpus has only {result['document_count']} documents "
            f"(category counts: {result['category_counts']}), evaluated with "
            f"{result['cv_folds']}-fold cross-validation. These metrics demonstrate that the "
            f"TF-IDF -> Chi-Square -> Naive Bayes pipeline works correctly, but are not a "
            f"statistically robust accuracy claim. Add more labeled documents per category "
            f"for a more reliable evaluation."
        )

    return ClassificationTrainResponse(
        document_count=result["document_count"],
        category_counts=result["category_counts"],
        features_selected=result["features_selected"],
        cv_folds=result["cv_folds"],
        accuracy=result["accuracy"],
        precision_macro=result["precision_macro"],
        recall_macro=result["recall_macro"],
        f1_macro=result["f1_macro"],
        confusion_matrix_labels=result["confusion_matrix"]["labels"],
        confusion_matrix=result["confusion_matrix"]["matrix"],
        is_small_corpus=result["is_small_corpus"],
        warning=warning,
    )


@router.get("/status", response_model=ClassificationStatusResponse)
def classification_status():
    """Check whether a classifier is currently trained."""
    status_info = classification_service.get_training_status()
    return ClassificationStatusResponse(**status_info)


@router.post("/predict/{document_id}", response_model=ClassificationPredictionResponse)
def predict_document_category(document_id: str):
    """
    Predict the category of an already-uploaded, preprocessed document
    using the currently trained classifier.
    """
    try:
        document_service.get_document(document_id)  # validate document exists
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        result = classification_service.predict_document(document_id)
    except classification_service.ClassificationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error predicting category for %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not predict the document's category.",
        )

    return ClassificationPredictionResponse(document_id=document_id, **result)