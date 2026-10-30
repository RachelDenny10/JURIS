"""
Naïve Bayes document classification service.

Phase 8 scope: implements the full pipeline from project Section 7:

    TF-IDF -> Chi-Square feature selection -> Multinomial Naive Bayes

Uses the already-fitted corpus-level TF-IDF matrix (Phase 5), applies
SelectKBest(chi2) feature selection (Phase 7's statistical test,
applied here as a dimensionality-reduction step rather than for
per-category reporting), and trains a MultinomialNB classifier.

IMPORTANT - small corpus honesty (Section 27/37):
This corpus is small (low double digits of documents, 2-3 per
category). A conventional train/test split is not meaningful here,
since holding out even one document per class is a large fraction of
that class. Instead this service uses STRATIFIED K-FOLD CROSS-
VALIDATION, with k capped at the smallest class size, to produce an
honest (if still limited) evaluation. All reported metrics are
clearly labeled as being from a small corpus - not a robust accuracy
claim.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from app.core.config import get_settings
from app.services import document_service, tfidf_service

logger = logging.getLogger("juris.classification_service")

SELECTOR_FILENAME = "chi2_selector.joblib"
MODEL_FILENAME = "logistic_regression_model.joblib"
NAIVE_BAYES_MODEL_FILENAME = "naive_bayes_model.joblib"
LABELS_FILENAME = "classification_labels.joblib"

DEFAULT_TOP_K_FEATURES = 500
MIN_DOCUMENTS_TO_TRAIN = 4
MIN_CATEGORIES_TO_TRAIN = 2

class ClassificationError(Exception):
    """Raised when classification cannot proceed (e.g. not enough data, not trained yet)."""


def _model_dir() -> Path:
    settings = get_settings()
    path = settings.project_root / "models" / "classification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_corpus_matrix_and_labels():
    """
    Load the fitted TF-IDF matrix (Phase 5) and align it with each
    document's category label from the metadata store.
    """
    try:
        vectorizer, matrix, document_ids = tfidf_service._load_fitted_state()
    except tfidf_service.TfidfError as exc:
        raise ClassificationError(
            f"Classification requires a fitted TF-IDF model first. {exc}"
        ) from exc

    all_docs = {d.document_id: d for d in document_service.list_documents()}
    labels = []
    for doc_id in document_ids:
        record = all_docs.get(doc_id)
        if record is None:
            raise ClassificationError(
                f"Document {doc_id} from the fitted TF-IDF corpus no longer exists. "
                f"Re-run POST /analysis/tfidf/fit to refresh the corpus."
            )
        labels.append(record.category.value)

    return vectorizer, matrix, np.array(labels), document_ids



def train_and_evaluate(top_k_features: int = DEFAULT_TOP_K_FEATURES) -> dict:
    """Evaluate and train Logistic Regression, retaining Naive Bayes as fallback."""

    vectorizer, matrix, labels, document_ids = _get_corpus_matrix_and_labels()

    unique_categories = sorted(set(labels))
    category_counts = {
        cat: int((labels == cat).sum()) for cat in unique_categories
    }
    min_class_count = min(category_counts.values())
    document_count = len(labels)

    if document_count < MIN_DOCUMENTS_TO_TRAIN:
        raise ClassificationError(
            f"At least {MIN_DOCUMENTS_TO_TRAIN} documents are required."
        )

    if len(unique_categories) < MIN_CATEGORIES_TO_TRAIN:
        raise ClassificationError("At least two categories are required.")

    if min_class_count < 2:
        raise ClassificationError(
            f"Every category needs at least two documents. Counts: {category_counts}"
        )

    effective_k_features = min(top_k_features, matrix.shape[1])

    n_splits = min(min_class_count, 5)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    # Evaluate with feature selection fitted separately inside each fold.
    cv_pipeline = Pipeline([
        (
            "selector",
            SelectKBest(score_func=chi2, k=effective_k_features),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])

    cv_predictions = cross_val_predict(
        cv_pipeline,
        matrix,
        labels,
        cv=cv,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        cv_predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = float(np.mean(cv_predictions == labels))

    conf_matrix = confusion_matrix(
        labels,
        cv_predictions,
        labels=unique_categories,
    )

    # Fit final selector and primary classifier on the complete fitted corpus.
    selector = SelectKBest(
        score_func=chi2,
        k=effective_k_features,
    )
    selected_matrix = selector.fit_transform(matrix, labels)

    final_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    )
    final_model.fit(selected_matrix, labels)

    # Retain a separately trained Naive Bayes fallback.
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(selected_matrix, labels)

    model_dir = _model_dir()

    joblib.dump(selector, model_dir / SELECTOR_FILENAME)
    joblib.dump(final_model, model_dir / MODEL_FILENAME)
    joblib.dump(
        naive_bayes_model,
        model_dir / NAIVE_BAYES_MODEL_FILENAME,
    )
    joblib.dump(sorted(unique_categories), model_dir / LABELS_FILENAME)

    logger.info(
        "Trained Logistic Regression: %d documents, %d categories, "
        "%d features, %d-fold CV, accuracy=%.3f, macro-F1=%.3f",
        document_count,
        len(unique_categories),
        effective_k_features,
        n_splits,
        accuracy,
        f1,
    )

    is_small_corpus = document_count < 30 or min_class_count < 5

    return {
        "model": "Logistic Regression",
        "document_count": document_count,
        "category_counts": category_counts,
        "features_selected": effective_k_features,
        "cv_folds": n_splits,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "confusion_matrix": {
            "labels": unique_categories,
            "matrix": conf_matrix.tolist(),
        },
        "is_small_corpus": is_small_corpus,
    }


def _load_trained_artifacts():
    """Load the persisted selector, model, and label list. Raises ClassificationError if missing."""
    model_dir = _model_dir()
    selector_path = model_dir / SELECTOR_FILENAME
    model_path = model_dir / MODEL_FILENAME
    labels_path = model_dir / LABELS_FILENAME

    if not (selector_path.exists() and model_path.exists() and labels_path.exists()):
        raise ClassificationError(
            "No trained classifier found. Call POST /classification/train first."
        )

    selector = joblib.load(selector_path)
    model = joblib.load(model_path)
    labels = joblib.load(labels_path)
    return selector, model, labels


def predict_document(document_id: str) -> dict:
    """
    Predict the category of an already-uploaded, preprocessed document
    using the currently trained classifier.
    """
    selector, model, _ = _load_trained_artifacts()
    vectorizer, matrix, document_ids = tfidf_service._load_fitted_state()

    if document_id not in document_ids:
        raise ClassificationError(
            f"Document {document_id} is not part of the fitted TF-IDF corpus. "
            f"Preprocess it and re-run POST /analysis/tfidf/fit, then POST /classification/train."
        )

    row_index = document_ids.index(document_id)
    row = matrix[row_index]

    return _predict_from_tfidf_row(row, selector, model)


def _predict_from_tfidf_row(tfidf_row, selector, model) -> dict:
    selected_row = selector.transform(tfidf_row)
    predicted_category = model.predict(selected_row)[0]
    probabilities = model.predict_proba(selected_row)[0]

    class_probabilities = {
        cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probabilities)
    }
    confidence = round(float(max(probabilities)), 4)

    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
    }


def get_training_status() -> dict:
    """Report whether a classifier is currently trained."""
    try:
        _, _, labels = _load_trained_artifacts()
        return {"trained": True, "categories": list(labels)}
    except ClassificationError:
        return {"trained": False, "categories": []}