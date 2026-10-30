
"""Compare text classifiers using stratified cross-validation."""

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.services import document_service, tfidf_service

logger = logging.getLogger("juris.model_comparison")


def compare_models(top_k_features: int = 200) -> dict:
    """Compare three models on the same stratified folds."""

    _, _, document_ids = _get_corpus()
    records = {d.document_id: d for d in document_service.list_documents()}

    texts = []
    labels = []

    for doc_id in document_ids:
        record = records.get(doc_id)
        if record is None:
            raise ValueError(f"Document metadata not found: {doc_id}")

        text_path = getattr(record, "text_path", None)
        if not text_path:
            raise ValueError(f"Processed text path missing for: {doc_id}")

        from pathlib import Path

        path = Path(text_path)
        if not path.is_absolute():
            from app.core.config import get_settings
            path = get_settings().project_root / path

        if not path.exists():
            raise ValueError(f"Processed text file not found: {path}")

        texts.append(path.read_text(encoding="utf-8", errors="replace"))
        labels.append(record.category.value)

    labels = np.asarray(labels)
    category_counts = {
        str(category): int(np.sum(labels == category))
        for category in sorted(set(labels))
    }

    min_class_count = min(category_counts.values())
    if len(texts) < 4 or min_class_count < 2:
        raise ValueError(
            "Model comparison requires at least four documents and "
            "at least two documents in every category."
        )

    folds = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    model_specs = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = {}

    for name, classifier in model_specs.items():
        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    max_df=1.0,
                ),
            ),
            (
                "selector",
                SelectKBest(
                    score_func=chi2,
                    k=top_k_features,
                ),
            ),
            ("classifier", classifier),
        ])

        predictions = cross_val_predict(
            pipeline,
            texts,
            labels,
            cv=cv,
        )

        results[name] = {
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "macro_f1": round(
                float(f1_score(labels, predictions, average="macro", zero_division=0)),
                4,
            ),
            "classification_report": classification_report(
                labels,
                predictions,
                labels=sorted(set(labels)),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": {
                "labels": sorted(set(labels)),
                "matrix": confusion_matrix(
                    labels,
                    predictions,
                    labels=sorted(set(labels)),
                ).tolist(),
            },
        }

    return {
        "document_count": len(texts),
        "category_counts": category_counts,
        "cv_folds": folds,
        "features_requested": top_k_features,
        "results": results,
        "note": (
            "Exploratory comparison only. The corpus is small, so scores "
            "are unstable. TF-IDF and feature selection are fitted inside "
            "each fold to avoid evaluation leakage."
        ),
    }


def _get_corpus():
    """Get document IDs from the fitted TF-IDF corpus."""
    return tfidf_service._load_fitted_state()
