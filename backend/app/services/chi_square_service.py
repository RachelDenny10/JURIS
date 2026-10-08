"""
Chi-Square (chi2) feature selection service using scikit-learn.

Phase 7 scope: run chi2 over the already-fitted TF-IDF matrix
(Phase 5) against document category labels, to identify which terms
are most statistically discriminative for each category.

IMPORTANT: chi2 requires labeled data across at least 2 distinct
categories, and its results are only statistically meaningful with a
reasonable number of documents per category. With a very small corpus
(as is expected early in this project - see Section 8/27/37), scores
will be unstable. This service computes real scores regardless, but
callers should treat results from a tiny corpus as illustrative, not
conclusive.
"""

import logging

import numpy as np
from sklearn.feature_selection import chi2

from app.services import document_service, tfidf_service

logger = logging.getLogger("juris.chi_square_service")


class ChiSquareError(Exception):
    """Raised when chi-square cannot be computed (e.g. no TF-IDF fitted, <2 categories)."""


def _get_labels_for_fitted_documents(document_ids: list[str]) -> list[str]:
    """Look up the category label for each document_id in the fitted TF-IDF corpus."""
    all_docs = {d.document_id: d for d in document_service.list_documents()}
    labels = []
    for doc_id in document_ids:
        record = all_docs.get(doc_id)
        if record is None:
            raise ChiSquareError(f"Document {doc_id} from the fitted TF-IDF corpus no longer exists.")
        labels.append(record.category.value)
    return labels


def compute_chi_square_scores(top_n_per_category: int = 15) -> dict:
    """
    Compute chi2 scores between the fitted TF-IDF matrix and document
    category labels, returning the top-N most discriminative terms
    per category.
    """
    try:
        vectorizer, matrix, document_ids = tfidf_service._load_fitted_state()
    except tfidf_service.TfidfError as exc:
        raise ChiSquareError(
            f"Chi-Square requires a fitted TF-IDF model first. {exc}"
        ) from exc

    labels = _get_labels_for_fitted_documents(document_ids)
    unique_categories = sorted(set(labels))

    if len(unique_categories) < 2:
        raise ChiSquareError(
            f"Chi-Square feature selection needs at least 2 distinct categories in the "
            f"fitted corpus, found {len(unique_categories)} ({unique_categories}). "
            f"Upload, extract, and preprocess documents from more categories, then re-fit TF-IDF."
        )

    feature_names = vectorizer.get_feature_names_out()
    labels_array = np.array(labels)

    results_per_category: dict[str, list[dict]] = {}

    for category in unique_categories:
        # One-vs-rest binary target: 1 if this document belongs to `category`, else 0.
        binary_target = (labels_array == category).astype(int)

        if binary_target.sum() == 0 or binary_target.sum() == len(binary_target):
            # Should not happen given the unique_categories check above, but guard anyway.
            continue

        chi2_scores, p_values = chi2(matrix, binary_target)

        # Rank terms by chi2 score, descending.
        ranked_indices = np.argsort(chi2_scores)[::-1][:top_n_per_category]

        category_results = []
        for idx in ranked_indices:
            score = float(chi2_scores[idx])
            if np.isnan(score):
                continue
            category_results.append({
                "term": feature_names[idx],
                "chi2_score": round(score, 4),
                "p_value": round(float(p_values[idx]), 6),
            })

        results_per_category[category] = category_results

    document_count = len(document_ids)
    category_counts = {cat: labels.count(cat) for cat in unique_categories}
    min_docs_per_category = min(category_counts.values())

    is_statistically_weak = document_count < 10 or min_docs_per_category < 2

    logger.info(
        "Computed chi-square over %d documents across %d categories (min per category: %d)",
        document_count, len(unique_categories), min_docs_per_category,
    )

    return {
        "results_per_category": results_per_category,
        "category_counts": category_counts,
        "document_count": document_count,
        "is_statistically_weak": is_statistically_weak,
    }