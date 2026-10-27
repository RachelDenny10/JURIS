"""
Embedding service (Phase 11): chunking + Sentence Transformer embeddings.

Pipeline for one document:
    extracted text -> overlapping chunks (with page ranges) -> embeddings

Chunks are about chunk_target_words words with chunk_overlap_words words
of overlap, split on paragraph/sentence boundaries where possible.
Embeddings are L2-normalised, so cosine similarity is a dot product
(what FAISS will use in Phase 12).

Output per document (under models/embeddings/):
    <document_id>_chunks.json   chunk texts + page ranges + metadata
    <document_id>_vectors.npy   float32 matrix, one row per chunk

The model name is configurable (EMBEDDING_MODEL_NAME) and is never
hardcoded outside config.py.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger("juris.embedding_service")

PAGE_BREAK_TEXT = "----- PAGE BREAK -----"

_model_cache = None


class EmbeddingError(Exception):
    """Raised when embeddings cannot be produced (model missing, empty text, etc.)."""


@dataclass
class Chunk:
    chunk_id: int
    text: str
    word_count: int
    page_start: int
    page_end: int


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def _embeddings_dir() -> Path:
    path = get_settings().project_root / "models" / "embeddings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunks_path(document_id: str) -> Path:
    return _embeddings_dir() / f"{document_id}_chunks.json"


def vectors_path(document_id: str) -> Path:
    return _embeddings_dir() / f"{document_id}_vectors.npy"


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def _get_model():
    """Load (once) and return the Sentence Transformer model."""
    global _model_cache
    if _model_cache is None:
        model_name = get_settings().embedding_model_name
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s (first run downloads it)", model_name)
            _model_cache = SentenceTransformer(model_name)
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed in this environment."
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}': {exc}. "
                "The first run needs an internet connection to download it."
            ) from exc
    return _model_cache


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+")


def _split_into_units(page_text: str) -> list[str]:
    """Split a page into sentence-like units, falling back to word groups for long runs."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", page_text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


def chunk_text(raw_text: str, target_words: int, overlap_words: int) -> list[Chunk]:
    """
    Split extracted text into overlapping chunks, recording page ranges.

    Page numbers come from the PAGE BREAK markers inserted during
    extraction (1-based).
    """
    if target_words < 20:
        raise EmbeddingError("chunk_target_words must be at least 20.")
    if overlap_words < 0 or overlap_words >= target_words:
        raise EmbeddingError("chunk_overlap_words must be >= 0 and smaller than chunk_target_words.")

    pages = raw_text.split(PAGE_BREAK_TEXT)

    # (word, page_number) stream, built from sentence units so boundaries stay natural.
    unit_list: list[tuple[list[str], int]] = []
    for page_index, page_text in enumerate(pages, start=1):
        for unit in _split_into_units(page_text):
            words = unit.split()
            # Very long units (tables, run-on text) are cut into word groups.
            for start in range(0, len(words), target_words):
                unit_list.append((words[start:start + target_words], page_index))

    if not unit_list:
        return []

    chunks: list[Chunk] = []
    current: list[tuple[list[str], int]] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        words = [w for unit_words, _ in current for w in unit_words]
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                text=" ".join(words),
                word_count=len(words),
                page_start=current[0][1],
                page_end=current[-1][1],
            )
        )
        # Keep trailing units as overlap for the next chunk.
        carried: list[tuple[list[str], int]] = []
        carried_words = 0
        for unit in reversed(current):
            if carried_words >= overlap_words:
                break
            carried.insert(0, unit)
            carried_words += len(unit[0])
        current = carried
        current_words = carried_words

    for unit_words, page_number in unit_list:
        if current_words + len(unit_words) > target_words and current_words > overlap_words:
            flush()
        current.append((unit_words, page_number))
        current_words += len(unit_words)

    # Final chunk: skip it if it would only repeat the carried overlap.
    last_has_new_text = any(unit not in current[:0] for unit in current)
    if current and (not chunks or current_words > overlap_words) and last_has_new_text:
        flush()

    return chunks


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings; returns a float32 (n, dim) matrix of unit vectors."""
    if not texts:
        raise EmbeddingError("Cannot embed an empty list of texts.")
    model = _get_model()
    batch_size = get_settings().embedding_batch_size
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed one query string; returns a float32 vector of shape (dim,)."""
    if not query or not query.strip():
        raise EmbeddingError("Query text is empty.")
    return embed_texts([query.strip()])[0]


def get_model_info() -> dict:
    """Return the configured model name and its embedding dimension (loads the model)."""
    model = _get_model()
    return {
        "model_name": get_settings().embedding_model_name,
        "dimension": int(model.get_sentence_embedding_dimension()),
    }


# --------------------------------------------------------------------------
# Per-document pipeline
# --------------------------------------------------------------------------

def embed_document(document_id: str, raw_text: str) -> dict:
    """Chunk and embed one document, saving chunks and vectors to disk."""
    if not raw_text or not raw_text.strip():
        raise EmbeddingError("Document text is empty.")

    settings = get_settings()
    chunks = chunk_text(raw_text, settings.chunk_target_words, settings.chunk_overlap_words)
    if not chunks:
        raise EmbeddingError("No chunks could be produced from this document's text.")

    vectors = embed_texts([c.text for c in chunks])

    payload = {
        "document_id": document_id,
        "model_name": settings.embedding_model_name,
        "dimension": int(vectors.shape[1]),
        "chunk_target_words": settings.chunk_target_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "chunks": [asdict(c) for c in chunks],
    }
    chunks_path(document_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    np.save(vectors_path(document_id), vectors)

    logger.info(
        "Embedded document %s: %d chunks, dimension %d",
        document_id, len(chunks), vectors.shape[1],
    )
    return {
        "chunk_count": len(chunks),
        "dimension": int(vectors.shape[1]),
        "model_name": settings.embedding_model_name,
        "average_chunk_words": round(sum(c.word_count for c in chunks) / len(chunks), 1),
    }


def load_document_embeddings(document_id: str) -> tuple[dict, np.ndarray]:
    """Load previously saved chunks and vectors for a document."""
    c_path, v_path = chunks_path(document_id), vectors_path(document_id)
    if not (c_path.exists() and v_path.exists()):
        raise EmbeddingError(
            "No embeddings found for this document. Call POST /embeddings/{document_id} first."
        )
    return json.loads(c_path.read_text(encoding="utf-8")), np.load(v_path)


def similarity_search_in_document(document_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Brute-force cosine search within ONE document's chunks. This is a
    simple test of the embeddings; the cross-document FAISS index comes
    in Phase 12.
    """
    meta, vectors = load_document_embeddings(document_id)
    query_vector = embed_query(query)

    if meta.get("model_name") != get_settings().embedding_model_name:
        raise EmbeddingError(
            f"Stored embeddings were made with '{meta.get('model_name')}', but the configured "
            f"model is '{get_settings().embedding_model_name}'. Re-run POST /embeddings/{{document_id}}."
        )

    scores = vectors @ query_vector  # unit vectors: dot product == cosine similarity
    order = np.argsort(scores)[::-1][:top_k]
    chunk_list = meta["chunks"]
    return [
        {
            "chunk_id": chunk_list[i]["chunk_id"],
            "score": round(float(scores[i]), 4),
            "page_start": chunk_list[i]["page_start"],
            "page_end": chunk_list[i]["page_end"],
            "text": chunk_list[i]["text"],
        }
        for i in order
    ]