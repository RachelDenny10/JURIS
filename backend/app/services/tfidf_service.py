"""
TF-IDF service using scikit-learn.

Phase 5 scope: fit a TF-IDF vectorizer across the corpus of
preprocessed documents, persist it to disk, and expose per-document
top-keyword extraction.

TF-IDF is inherently corpus-relative (the IDF part needs to see all
documents), so this service always operates on the full set of
documents that have been preprocessed so far - not a single document
in isolation. With very few documents, scores will look sparse or
uninformative; this is expected (see project Section 27/37 on not
overstating results from small corpora).
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import get_settings

logger = logging.getLogger("juris.tfidf_service")

VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
MATRIX_FILENAME = "tfidf_matrix.joblib"
DOC_INDEX_FILENAME = "tfidf_doc_index.json"


class TfidfError(Exception):
    """Raised when TF-IDF cannot be computed (e.g. no preprocessed documents yet)."""


def _model_dir() -> Path:
    settings = get_settings()
    path = settings.project_root / "models" / "classification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_all_preprocessed_documents() -> list[tuple[str, str]]:
    """
    Load every document's lemmatized tokens from data/processed/.

    Returns a list of (document_id, joined_lemmatized_text) tuples.
    Documents that haven't been preprocessed yet are skipped.
    """
    settings = get_settings()
    processed_dir = settings.data_metadata_path.parent / "processed"

    documents = []
    for json_file in sorted(processed_dir.glob("*_preprocessed.json")):
        document_id = json_file.stem.replace("_preprocessed", "")
        data = json.loads(json_file.read_text(encoding="utf-8"))
        lemmatized_tokens = data.get("lemmatized_tokens", [])
        if lemmatized_tokens:
            documents.append((document_id, " ".join(lemmatized_tokens)))

    return documents


def fit_corpus_tfidf() -> dict:
    """
    Fit a TF-IDF vectorizer across all currently preprocessed
    documents and persist the vectorizer, matrix, and document index
    to models/classification/.

    Raises TfidfError if fewer than 2 documents are available, since
    TF-IDF's IDF component is not meaningful with a single document.
    """
    documents = _load_all_preprocessed_documents()

    if len(documents) < 2:
        raise TfidfError(
            f"At least 2 preprocessed documents are needed to fit TF-IDF meaningfully "
            f"(found {len(documents)}). Upload, extract, and preprocess more documents first."
        )

    document_ids = [doc_id for doc_id, _ in documents]
    texts = [text for _, text in documents]

    vectorizer = TfidfVectorizer(
        lowercase=False,  # already lowercased during preprocessing
        token_pattern=r"(?u)\b\w+\b",
        max_df=0.95,  # ignore terms in >95% of docs (near-universal, low signal)
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    model_dir = _model_dir()
    joblib.dump(vectorizer, model_dir / VECTORIZER_FILENAME)
    joblib.dump(tfidf_matrix, model_dir / MATRIX_FILENAME)
    (model_dir / DOC_INDEX_FILENAME).write_text(json.dumps(document_ids), encoding="utf-8")

    logger.info(
        "Fitted TF-IDF over %d documents, vocabulary size %d",
        len(documents), len(vectorizer.vocabulary_),
    )

    return {
        "document_count": len(documents),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def _load_fitted_state():
    """Load the persisted vectorizer, matrix, and document index. Raises TfidfError if missing."""
    model_dir = _model_dir()
    vectorizer_path = model_dir / VECTORIZER_FILENAME
    matrix_path = model_dir / MATRIX_FILENAME
    index_path = model_dir / DOC_INDEX_FILENAME

    if not (vectorizer_path.exists() and matrix_path.exists() and index_path.exists()):
        raise TfidfError(
            "No fitted TF-IDF model found. Call POST /analysis/tfidf/fit first."
        )

    vectorizer = joblib.load(vectorizer_path)
    matrix = joblib.load(matrix_path)
    document_ids = json.loads(index_path.read_text(encoding="utf-8"))
    return vectorizer, matrix, document_ids


def get_top_keywords_for_document(document_id: str, top_n: int = 10) -> list[dict]:
    """
    Return the top-N TF-IDF keywords for a specific document, using
    the currently fitted corpus-level vectorizer.

    Raises TfidfError if no model is fitted, or if this document_id
    was not part of the fitted corpus (re-fit after preprocessing
    new documents to include them).
    """
    vectorizer, matrix, document_ids = _load_fitted_state()

    if document_id not in document_ids:
        raise TfidfError(
            f"Document {document_id} was not part of the fitted TF-IDF corpus. "
            f"Call POST /analysis/tfidf/fit again after preprocessing it."
        )

    row_index = document_ids.index(document_id)
    row = matrix[row_index].toarray().flatten()

    feature_names = vectorizer.get_feature_names_out()
    top_indices = np.argsort(row)[::-1][:top_n]

    results = []
    for idx in top_indices:
        score = float(row[idx])
        if score > 0:
            results.append({"term": feature_names[idx], "score": round(score, 4)})

    return results


def get_corpus_status() -> dict:
    """Return basic info about whether a TF-IDF model is currently fitted."""
    try:
        _, _, document_ids = _load_fitted_state()
        return {"fitted": True, "document_count": len(document_ids), "document_ids": document_ids}
    except TfidfError:
        return {"fitted": False, "document_count": 0, "document_ids": []}