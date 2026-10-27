"""
Keyword search (Phase 12): TF-IDF + cosine similarity, at document level.

This is the classical Text Mining search mode (project Section 19).
The query goes through the SAME preprocessing as the documents
(cleaning, tokenisation, stop-word removal, lemmatisation) and is then
transformed with the already-fitted TF-IDF vectorizer. TF-IDF rows are
L2-normalised, so the dot product between the query vector and a
document row is their cosine similarity.

Only terms in the fitted vocabulary can match. Query terms the corpus
has never seen are reported back, not silently dropped.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.core.constants import CATEGORY_LABELS, DocumentCategory
from app.services import document_service, preprocessing_service, tfidf_service

logger = logging.getLogger("juris.keyword_search_service")

PAGE_BREAK_TEXT = "----- PAGE BREAK -----"
SNIPPET_BEFORE = 60
SNIPPET_AFTER = 300
MAX_MATCHED_TERMS = 5


class KeywordSearchError(Exception):
    """Raised when keyword search cannot run (TF-IDF not fitted, no searchable terms)."""


def _best_snippet(text_file: Path, terms: list[str]) -> str:
    """Return the stretch of text where the matched terms are densest."""
    if not terms:
        return ""
    try:
        raw = text_file.read_text(encoding="utf-8")
    except OSError:
        return ""
    text = re.sub(r"\s+", " ", raw.replace(PAGE_BREAK_TEXT, " ")).strip()
    if not text:
        return ""

    pattern = re.compile("|".join(r"\b" + re.escape(t) for t in terms), re.IGNORECASE)
    positions = [m.start() for m in pattern.finditer(text)]
    if not positions:
        return ""

    # Two-pointer sweep: find the start position with the most matches in the next SNIPPET_AFTER chars.
    best_start, best_count, j = positions[0], 0, 0
    for i, start in enumerate(positions):
        j = max(j, i)
        while j < len(positions) and positions[j] - start <= SNIPPET_AFTER:
            j += 1
        if j - i > best_count:
            best_count, best_start = j - i, start

    begin = max(0, best_start - SNIPPET_BEFORE)
    end = min(len(text), best_start + SNIPPET_AFTER)
    return ("..." if begin > 0 else "") + text[begin:end].strip() + ("..." if end < len(text) else "")


def keyword_search(
    query: str,
    top_k: int = 5,
    category: Optional[DocumentCategory] = None,
    document_id: Optional[str] = None,
) -> dict:
    """Rank documents by TF-IDF cosine similarity to the query."""
    try:
        vectorizer, matrix, corpus_ids = tfidf_service._load_fitted_state()
    except tfidf_service.TfidfError as exc:
        raise KeywordSearchError(f"Keyword search needs a fitted TF-IDF model. {exc}") from exc

    processed = preprocessing_service.preprocess_text(query)
    lemmas = list(dict.fromkeys(processed.lemmatized_tokens))  # unique, order kept
    if not lemmas:
        raise KeywordSearchError(
            "The query has no searchable terms after stop-word removal. Try more specific words."
        )

    vocabulary = vectorizer.vocabulary_
    known = [t for t in lemmas if t in vocabulary]
    unknown = [t for t in lemmas if t not in vocabulary]

    settings = get_settings()
    docs = {d.document_id: d for d in document_service.list_documents()}
    corpus_set = set(corpus_ids)

    warnings: list[str] = []
    not_fitted = [
        d.title
        for d in docs.values()
        if d.processing_status == "preprocessed" and d.document_id not in corpus_set
    ]
    if not_fitted:
        warnings.append(
            f"{len(not_fitted)} preprocessed document(s) are not in the fitted TF-IDF corpus and "
            f"were not searched ({', '.join(not_fitted)}). Re-run POST /analysis/tfidf/fit."
        )
    if not known:
        warnings.append(
            "None of the query terms occur in the corpus vocabulary. Try different or broader words."
        )

    results: list[dict] = []
    if known:
        query_vector = vectorizer.transform([" ".join(lemmas)])
        scores = (matrix @ query_vector.T).toarray().ravel()

        for row in np.argsort(scores)[::-1]:
            score = float(scores[row])
            if score <= 0:
                break
            doc = docs.get(corpus_ids[int(row)])
            if doc is None:
                continue  # deleted after TF-IDF was fitted
            if category is not None and doc.category != category:
                continue
            if document_id is not None and doc.document_id != document_id:
                continue

            matched = []
            for term in known:
                weight = float(matrix[int(row), vocabulary[term]])
                if weight > 0:
                    matched.append({"term": term, "weight": round(weight, 4)})
            matched.sort(key=lambda m: m["weight"], reverse=True)
            matched = matched[:MAX_MATCHED_TERMS]

            snippet = ""
            if doc.text_path:
                snippet = _best_snippet(settings.project_root / doc.text_path, [m["term"] for m in matched])

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": round(score, 4),
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "category": doc.category.value,
                    "category_label": CATEGORY_LABELS[doc.category.value],
                    "matched_terms": matched,
                    "snippet": snippet,
                }
            )
            if len(results) >= top_k:
                break

    return {
        "query": query,
        "query_terms": lemmas,
        "terms_not_in_vocabulary": unknown,
        "results": results,
        "documents_searched": len([i for i in corpus_ids if i in docs]),
        "warning": " ".join(warnings) if warnings else None,
    }