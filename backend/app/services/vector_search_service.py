"""
FAISS vector search service (Phase 12).

Builds one persistent FAISS index over the embedded chunks of every
document and answers semantic (meaning-based) queries against it.

Design:
  - IndexFlatIP: exact inner-product search. Embeddings are L2-normalised
    (Phase 11), so inner product == cosine similarity. A flat index is
    exact and fast at this corpus size (thousands of chunks).
  - Persisted to models/vector_store/ (index + JSON metadata). Written
    atomically (temp file, then replace) and read via faiss
    serialize/deserialize, so Windows path handling cannot trip FAISS's
    own file I/O.
  - Titles and categories are NOT stored in the index. They are looked
    up from the database at search time, so corrected metadata shows up
    immediately and chunks of deleted documents are filtered out.
  - The index is a snapshot: rebuild it after documents are added,
    re-extracted or deleted. get_index_status() reports when it is stale.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.core.constants import CATEGORY_LABELS, DocumentCategory
from app.services import document_service, embedding_service

logger = logging.getLogger("juris.vector_search_service")

INDEX_FILENAME = "juris_index.faiss"
META_FILENAME = "juris_index_meta.json"
SMALL_INDEX_LIMIT = 5000  # at or below this many vectors, search them all

_build_lock = threading.Lock()
_cache: dict = {"stamp": None, "index": None, "meta": None}
_faiss_module = None


class VectorSearchError(Exception):
    """Raised for setup/service problems (FAISS or the model unavailable, unreadable index)."""


class IndexNotReadyError(VectorSearchError):
    """Raised when the index does not exist yet or there is nothing to index."""


class IndexBusyError(VectorSearchError):
    """Raised when an index build is already running."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _import_faiss():
    global _faiss_module
    if _faiss_module is None:
        try:
            import faiss
        except ImportError as exc:
            raise VectorSearchError("faiss-cpu is not installed in this environment.") from exc
        # FAISS and PyTorch each bundle an OpenMP runtime; running both
        # multi-threaded in one process can crash on Windows. The index is
        # small, so a single FAISS thread costs nothing.
        faiss.omp_set_num_threads(1)
        _faiss_module = faiss
    return _faiss_module


def _store_dir() -> Path:
    path = get_settings().project_root / "models" / "vector_store"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path() -> Path:
    return _store_dir() / INDEX_FILENAME


def _meta_path() -> Path:
    return _store_dir() / META_FILENAME


def _atomic_write(path: Path, data: bytes) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_bytes(data)
    os.replace(temp_path, path)


def _embeddings_are_current(document_id: str, text_file: Path) -> bool:
    """True if saved embeddings exist, match the configured model/chunking, and are newer than the text."""
    settings = get_settings()
    chunk_file = embedding_service.chunks_path(document_id)
    vector_file = embedding_service.vectors_path(document_id)
    if not (chunk_file.exists() and vector_file.exists()):
        return False
    try:
        saved = json.loads(chunk_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        saved.get("model_name") == settings.embedding_model_name
        and saved.get("chunk_target_words") == settings.chunk_target_words
        and saved.get("chunk_overlap_words") == settings.chunk_overlap_words
        and text_file.stat().st_mtime <= vector_file.stat().st_mtime
    )


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def _verify_index(index, matrix: np.ndarray, sample_size: int = 25, k: int = 5) -> None:
    """
    Guard against silent FAISS errors (see the OpenMP note in main.py):
    search a sample of the indexed vectors and compare FAISS's top-k
    scores with an exact numpy computation. Scores are compared, not
    ids, so identical duplicate chunks (tied scores) cannot cause a
    false alarm. Raises VectorSearchError on any disagreement.
    """
    n = int(matrix.shape[0])
    rng = np.random.default_rng(0)
    rows = rng.choice(n, size=min(sample_size, n), replace=False)
    queries = np.ascontiguousarray(matrix[rows])
    k = min(k, n)

    faiss_scores, _ = index.search(queries, k)
    exact_scores = np.sort(queries @ matrix.T, axis=1)[:, ::-1][:, :k]

    if not np.allclose(faiss_scores, exact_scores, atol=1e-4):
        raise VectorSearchError(
            "FAISS results disagree with an exact numpy check, so the index was NOT saved. "
            "This can happen when two OpenMP runtimes share a process; see the note in main.py."
        )
    logger.info("Index self-check passed on %d sample queries", len(rows))

def build_index(force: bool = False) -> dict:
    """
    Embed any document whose embeddings are missing or stale (or all of
    them when force=True), then build and persist a FAISS index over
    every chunk.
    """
    faiss = _import_faiss()
    settings = get_settings()

    if not _build_lock.acquire(blocking=False):
        raise IndexBusyError("An index build is already running. Wait for it to finish.")

    started = time.monotonic()
    try:
        documents = sorted(
            document_service.list_documents(), key=lambda d: (d.category.value, d.title)
        )
        parts: list[np.ndarray] = []
        chunk_records: list[dict] = []
        indexed: list[dict] = []
        skipped: list[dict] = []
        dimension: Optional[int] = None

        def skip(doc, reason: str) -> None:
            skipped.append({"document_id": doc.document_id, "title": doc.title, "reason": reason})
            logger.warning("Index build: skipping %s (%s)", doc.title, reason)

        for doc in documents:
            if not doc.text_path:
                skip(doc, "No extracted text yet (run /analysis/extract first).")
                continue
            text_file = settings.project_root / doc.text_path
            if not text_file.exists():
                skip(doc, "Extracted text file is missing on disk.")
                continue
            text = text_file.read_text(encoding="utf-8")
            if not text.strip():
                skip(doc, "Extracted text is empty.")
                continue

            embedded_now = False
            if force or not _embeddings_are_current(doc.document_id, text_file):
                try:
                    embedding_service.embed_document(doc.document_id, text)
                except embedding_service.EmbeddingError as exc:
                    raise VectorSearchError(f"Could not embed '{doc.title}': {exc}") from exc
                embedded_now = True

            try:
                doc_meta, vectors = embedding_service.load_document_embeddings(doc.document_id)
            except embedding_service.EmbeddingError:
                skip(doc, "Saved embeddings could not be read; retry with force=true.")
                continue

            chunks = doc_meta["chunks"]
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[0] == 0:
                skip(doc, "Saved embeddings are inconsistent; retry with force=true.")
                continue
            if dimension is not None and vectors.shape[1] != dimension:
                skip(doc, "Embedding dimension differs from the other documents; retry with force=true.")
                continue
            dimension = int(vectors.shape[1])

            for chunk in chunks:
                chunk_records.append(
                    {
                        "document_id": doc.document_id,
                        "chunk_id": chunk["chunk_id"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "text": chunk["text"],
                    }
                )
            parts.append(vectors.astype(np.float32, copy=False))
            indexed.append(
                {
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "category": doc.category.value,
                    "chunk_count": len(chunks),
                    "embedded_now": embedded_now,
                }
            )
            logger.info("Index build: %s -> %d chunks (embedded now: %s)", doc.title, len(chunks), embedded_now)

        if not parts:
            raise IndexNotReadyError(
                "No documents with extracted text were found to index. Run extraction first."
            )

        matrix = np.ascontiguousarray(np.vstack(parts), dtype=np.float32)
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
        _verify_index(index, matrix)

        meta = {
            "model_name": settings.embedding_model_name,
            "dimension": int(matrix.shape[1]),
            "index_type": "IndexFlatIP",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "total_vectors": int(index.ntotal),
            "documents": {d["document_id"]: d["chunk_count"] for d in indexed},
            "chunks": chunk_records,
        }

        _atomic_write(_index_path(), faiss.serialize_index(index).tobytes())
        _atomic_write(_meta_path(), json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        _cache.update(stamp=None, index=None, meta=None)  # force a reload from disk

        seconds = round(time.monotonic() - started, 1)
        logger.info("Index built: %d vectors, %d documents, %.1fs", index.ntotal, len(indexed), seconds)

        return {
            "total_vectors": int(index.ntotal),
            "dimension": int(matrix.shape[1]),
            "model_name": settings.embedding_model_name,
            "seconds": seconds,
            "index_path": str(_index_path().relative_to(settings.project_root)),
            "documents_indexed": indexed,
            "documents_skipped": skipped,
        }
    finally:
        _build_lock.release()


# --------------------------------------------------------------------------
# Load / status
# --------------------------------------------------------------------------

def _load_index():
    """Load the persisted index and metadata (cached until the files change)."""
    faiss = _import_faiss()
    index_file, meta_file = _index_path(), _meta_path()
    if not (index_file.exists() and meta_file.exists()):
        raise IndexNotReadyError("No vector index found. Call POST /search/index/build first.")

    stamp = (index_file.stat().st_mtime_ns, meta_file.stat().st_mtime_ns)
    if _cache["stamp"] == stamp and _cache["index"] is not None:
        return _cache["index"], _cache["meta"]

    try:
        index = faiss.deserialize_index(np.frombuffer(index_file.read_bytes(), dtype=np.uint8))
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VectorSearchError(
            f"The saved vector index could not be read ({exc}). Rebuild it with POST /search/index/build."
        ) from exc

    if index.ntotal != meta.get("total_vectors") or index.ntotal != len(meta.get("chunks", [])):
        raise VectorSearchError(
            "The saved index and its metadata do not match. Rebuild it with POST /search/index/build."
        )

    _cache.update(stamp=stamp, index=index, meta=meta)
    return index, meta


def get_index_status() -> dict:
    """Describe the saved index and whether it matches the current database."""
    settings = get_settings()
    documents = document_service.list_documents()
    by_id = {d.document_id: d for d in documents}
    candidates = [
        d for d in documents if d.text_path and (settings.project_root / d.text_path).exists()
    ]

    if not (_index_path().exists() and _meta_path().exists()):
        return {
            "built": False,
            "built_at": None,
            "model_name": None,
            "dimension": None,
            "total_vectors": 0,
            "up_to_date": False,
            "model_matches_config": None,
            "documents": [],
            "missing_documents": [d.title for d in candidates],
            "removed_documents": [],
        }

    try:
        meta = json.loads(_meta_path().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VectorSearchError(
            f"Index metadata could not be read ({exc}). Rebuild with POST /search/index/build."
        ) from exc

    indexed_counts: dict = meta.get("documents", {})
    missing = [d.title for d in candidates if d.document_id not in indexed_counts]
    removed = [doc_id for doc_id in indexed_counts if doc_id not in by_id]
    model_ok = meta.get("model_name") == settings.embedding_model_name

    return {
        "built": True,
        "built_at": meta.get("built_at"),
        "model_name": meta.get("model_name"),
        "dimension": meta.get("dimension"),
        "total_vectors": meta.get("total_vectors", 0),
        "up_to_date": (not missing) and (not removed) and model_ok,
        "model_matches_config": model_ok,
        "documents": [
            {
                "document_id": doc_id,
                "title": by_id[doc_id].title if doc_id in by_id else "(deleted)",
                "category": by_id[doc_id].category.value if doc_id in by_id else "?",
                "chunks": count,
            }
            for doc_id, count in indexed_counts.items()
        ],
        "missing_documents": missing,
        "removed_documents": removed,
    }


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def semantic_search(
    query: str,
    top_k: int = 5,
    category: Optional[DocumentCategory] = None,
    document_id: Optional[str] = None,
) -> dict:
    """Return the chunks closest in meaning to the query, optionally filtered."""
    index, meta = _load_index()
    settings = get_settings()

    if meta.get("model_name") != settings.embedding_model_name:
        raise VectorSearchError(
            f"The index was built with '{meta.get('model_name')}' but the configured model is "
            f"'{settings.embedding_model_name}'. Rebuild it with POST /search/index/build?force=true."
        )

    try:
        query_vector = embedding_service.embed_query(query)
    except embedding_service.EmbeddingError as exc:
        raise VectorSearchError(str(exc)) from exc

    docs = {d.document_id: d for d in document_service.list_documents()}

    total = int(index.ntotal)
    filtered = category is not None or document_id is not None
    fetch = total if (filtered or total <= SMALL_INDEX_LIMIT) else min(total, top_k * 20)

    scores, ids = index.search(query_vector.reshape(1, -1).astype(np.float32), fetch)

    results: list[dict] = []
    for score, row in zip(scores[0], ids[0]):
        if row < 0:
            continue
        chunk = meta["chunks"][int(row)]
        doc = docs.get(chunk["document_id"])
        if doc is None:
            continue  # the document was deleted after the index was built
        if category is not None and doc.category != category:
            continue
        if document_id is not None and doc.document_id != document_id:
            continue
        results.append(
            {
                "rank": len(results) + 1,
                "score": round(float(score), 4),
                "document_id": doc.document_id,
                "title": doc.title,
                "category": doc.category.value,
                "category_label": CATEGORY_LABELS[doc.category.value],
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": chunk["text"],
            }
        )
        if len(results) >= top_k:
            break

    indexed_ids = set(meta.get("documents", {}))
    not_indexed = [
        d.title for d in docs.values() if d.text_path and d.document_id not in indexed_ids
    ]
    warning = None
    if not_indexed:
        warning = (
            f"{len(not_indexed)} document(s) with extracted text are not in the index yet "
            f"({', '.join(not_indexed)}). Rebuild it with POST /search/index/build."
        )

    return {
        "query": query,
        "results": results,
        "chunks_searched": total,
        "index_built_at": meta.get("built_at", ""),
        "warning": warning,
    }