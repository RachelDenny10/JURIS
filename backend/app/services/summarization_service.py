"""
Summarization service (Phase 13).

Pipeline for one document:
  saved chunk embeddings (Phase 11)
    -> centroid + MMR passage selection (word budget)
    -> LLM summary (if configured)  OR  extractive summary (always available)
    -> grounding check on provisions, coverage report, cache

Extractive mode needs no LLM: it quotes the most representative passages.
LLM mode only sees the selected passages and is told to use nothing else.
Any LLM problem (no key, network, quota, malformed JSON) falls back to
extractive mode with a warning. Provisions named by the LLM are kept
only if their number appears in the passages that were sent.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.core.constants import CATEGORY_LABELS
from app.services import (
    document_service,
    embedding_service,
    entity_service,
    llm_service,
    tfidf_service,
    vector_search_service,
)

logger = logging.getLogger("juris.summarization_service")

DISCLAIMER = (
    "JURIS is an educational information system. This summary is machine-generated "
    "and is not legal advice. Check the cited source passages."
)

SYSTEM_PROMPT = (
    "You are a careful legal-information assistant for JURIS, an educational project on Indian "
    "consumer law. Summarise ONLY from the passages provided. Do not use outside knowledge and do "
    "not invent sections, numbers, penalties or deadlines. If the passages do not support a point, "
    "leave it out. You are not giving legal advice. Reply with a single JSON object and nothing else."
)


class SummarizationError(Exception):
    """Raised when a document cannot be summarised (no text, empty text, embedding problems)."""


class SummaryFormatError(Exception):
    """Raised when the LLM reply is not usable JSON in the expected shape."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+")
_REF_TOKEN = re.compile(r"\d+[A-Za-z]?(?:\([0-9A-Za-z]{1,4}\))*")


def _cache_path(document_id: str) -> Path:
    return get_settings().project_root / "data" / "processed" / f"{document_id}_summary.json"


def _pages(chunk: dict) -> str:
    start, end = chunk["page_start"], chunk["page_end"]
    return str(start) if start == end else f"{start}-{end}"


def _lead_text(text: str, max_chars: int) -> str:
    """The first sentence or two of a passage, cut to roughly max_chars."""
    out = ""
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if out and len(out) + len(sentence) + 1 > max_chars:
            break
        out = f"{out} {sentence}".strip()
        if len(out) >= 60:
            break
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "..."
    return out


def _load_chunks(doc, text: str, text_file: Path) -> tuple[list[dict], np.ndarray]:
    """Load saved chunks and vectors, embedding the document first if needed."""
    try:
        if not vector_search_service._embeddings_are_current(doc.document_id, text_file):
            embedding_service.embed_document(doc.document_id, text)
        meta, vectors = embedding_service.load_document_embeddings(doc.document_id)
    except embedding_service.EmbeddingError as exc:
        raise SummarizationError(f"Could not prepare passages for summarisation: {exc}") from exc

    chunks = meta["chunks"]
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not chunks:
        raise SummarizationError("Saved embeddings are inconsistent. Re-run POST /embeddings/{document_id}.")
    return chunks, vectors


def _select_passages(
    vectors: np.ndarray, chunks: list[dict], word_budget: int, mmr_lambda: float = 0.7
) -> list[int]:
    """
    Pick representative, non-redundant chunks: closeness to the document
    centroid (relevance) minus similarity to chunks already picked (MMR).
    Returns chunk indices in selection order (most central first).
    """
    n = vectors.shape[0]
    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    relevance = vectors @ centroid

    max_similarity = np.zeros(n, dtype=np.float32)
    taken = np.zeros(n, dtype=bool)
    order: list[int] = []
    words = 0

    while len(order) < n:
        score = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_similarity
        score[taken] = -np.inf
        best = int(np.argmax(score))
        chunk_words = int(chunks[best]["word_count"])
        if order and words + chunk_words > word_budget:
            break
        order.append(best)
        taken[best] = True
        words += chunk_words
        max_similarity = np.maximum(max_similarity, vectors @ vectors[best])
    return order


def _most_cited_provisions(text: str, top_n: int = 8) -> list[dict]:
    """Most frequently cited Section/Rule/Regulation references (rule-based, no spaCy)."""
    try:
        cleaned = entity_service.clean_text_for_entities(text)
        mentions = entity_service._rule_based_entities(cleaned)
    except Exception:
        logger.exception("Could not count provision references")
        return []
    counts = Counter(m.text for m in mentions if m.label in {"SECTION", "RULE", "REGULATION"})
    return [
        {"reference": ref, "description": f"Cited {n} time(s) in the document."}
        for ref, n in counts.most_common(top_n)
    ]


def _provision_is_supported(reference: str, context: str) -> bool:
    """True if every number in the reference (e.g. 35, 2(7)) appears in the context passages."""
    tokens = _REF_TOKEN.findall(reference)
    if not tokens:
        return False
    normalized = re.sub(r"\s+\(", "(", context.lower())
    return all(
        re.search(r"(?<![0-9a-z])" + re.escape(token.lower()) + r"(?![0-9a-z])", normalized)
        for token in tokens
    )


# --------------------------------------------------------------------------
# Extractive summary (no LLM)
# --------------------------------------------------------------------------

def _extractive_summary(doc, text: str, chunks: list[dict], rank_order: list[int]) -> dict:
    ranked = [chunks[i] for i in rank_order]
    top = sorted(ranked[:3], key=lambda c: c["chunk_id"])

    summary = " ".join(_lead_text(c["text"], 350) for c in top)
    key_points = [f"(p.{_pages(c)}) {_lead_text(c['text'], 250)}" for c in ranked[3:9]]

    warnings: list[str] = []
    try:
        top_terms = tfidf_service.get_top_keywords_for_document(doc.document_id, top_n=10)
        concepts = [t["term"] for t in top_terms]
    except tfidf_service.TfidfError:
        concepts = []
        warnings.append("Main concepts are empty because this document is not in the fitted TF-IDF corpus.")

    return {
        "mode": "extractive",
        "summary": summary,
        "key_points": key_points,
        "important_provisions": _most_cited_provisions(text),
        "main_concepts": concepts,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# LLM summary
# --------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise SummaryFormatError("The LLM reply did not contain a JSON object.")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise SummaryFormatError("The LLM reply was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise SummaryFormatError("The LLM reply JSON was not an object.")
    return data


def _string_list(value, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:max_chars] for item in value if str(item).strip()][:limit]


def _llm_summary(doc, chunks: list[dict], selected: list[int]) -> dict:
    context = "\n\n".join(
        f"[Passage {n} | pages {_pages(chunks[i])}]\n{chunks[i]['text']}"
        for n, i in enumerate(selected, start=1)
    )
    category_label = CATEGORY_LABELS[doc.category.value]
    user_prompt = (
        f"Document: {doc.title}\nCategory: {category_label}\n\n"
        f"Passages (selected from the document, in order):\n\n{context}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"summary": "3-5 sentence overview of what these passages cover",\n'
        ' "key_points": ["5-8 short bullet strings"],\n'
        ' "important_provisions": [{"reference": "e.g. Section 35(1)", "description": "one sentence"}],\n'
        ' "main_concepts": ["5-10 short legal concepts"]}\n'
        "Only list a provision if its number appears in the passages above."
    )

    data = _parse_llm_json(llm_service.generate(SYSTEM_PROMPT, user_prompt))

    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise SummaryFormatError("The LLM reply had no summary text.")

    provisions: list[dict] = []
    dropped = 0
    raw_provisions = data.get("important_provisions", [])
    for item in raw_provisions if isinstance(raw_provisions, list) else []:
        if not isinstance(item, dict):
            continue
        reference = str(item.get("reference", "")).strip()
        description = str(item.get("description", "")).strip()
        if not reference:
            continue
        if _provision_is_supported(reference, context):
            provisions.append({"reference": reference[:120], "description": description[:400]})
        else:
            dropped += 1

    warnings = []
    if dropped:
        warnings.append(
            f"{dropped} provision(s) named by the LLM were dropped because their numbers do not "
            f"appear in the passages that were sent."
        )

    return {
        "mode": "llm",
        "summary": summary[:2000],
        "key_points": _string_list(data.get("key_points"), 10, 400),
        "important_provisions": provisions[:10],
        "main_concepts": _string_list(data.get("main_concepts"), 12, 80),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def summarize_document(document_id: str, force: bool = False) -> dict:
    """
    Summarise one document. Returns a cached result unless force=True.
    Raises document_service.DocumentNotFoundError or SummarizationError.
    """
    settings = get_settings()
    doc = document_service.get_document(document_id)  # may raise DocumentNotFoundError

    cache_file = _cache_path(document_id)
    if not force and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["cached"] = True
            return cached
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable summary cache for %s", document_id)

    if not doc.text_path:
        raise SummarizationError("No extracted text found. Call /analysis/extract/{document_id} first.")
    text_file = settings.project_root / doc.text_path
    if not text_file.exists():
        raise SummarizationError("Recorded text file is missing on disk.")
    text = text_file.read_text(encoding="utf-8")
    if not text.strip():
        raise SummarizationError("The extracted text is empty.")

    chunks, vectors = _load_chunks(doc, text, text_file)
    rank_order = _select_passages(vectors, chunks, settings.summary_context_words)
    selected_in_order = sorted(rank_order)

    status = llm_service.provider_status()
    warnings: list[str] = []
    cacheable = True
    result = None

    if status["provider"] != "none":
        try:
            result = _llm_summary(doc, chunks, selected_in_order)
            warnings.extend(result.pop("warnings"))
        except llm_service.LLMNotConfiguredError as exc:
            warnings.append(f"LLM not used: {exc} Showing an extractive summary instead.")
            cacheable = False
        except llm_service.LLMError as exc:
            logger.warning("LLM summary failed for %s: %s", document_id, exc)
            warnings.append(f"LLM request failed: {exc} Showing an extractive summary instead.")
            cacheable = False
        except SummaryFormatError as exc:
            logger.warning("LLM reply unusable for %s: %s", document_id, exc)
            warnings.append(f"LLM reply could not be used: {exc} Showing an extractive summary instead.")
            cacheable = False

    if result is None:
        result = _extractive_summary(doc, text, chunks, rank_order)
        warnings.extend(result.pop("warnings"))
        if status["provider"] == "none":
            warnings.insert(
                0,
                "Extractive summary (no LLM configured): it quotes the most representative "
                "passages rather than rewriting them.",
            )

    used_provider = status["provider"] if result["mode"] == "llm" else "none"
    used_model = status["llm_model"] if result["mode"] == "llm" else None

    response = {
        "document_id": doc.document_id,
        "title": doc.title,
        "category": doc.category.value,
        "category_label": CATEGORY_LABELS[doc.category.value],
        "mode": result["mode"],
        "provider": used_provider,
        "llm_model": used_model,
        "summary": result["summary"],
        "key_points": result["key_points"],
        "important_provisions": result["important_provisions"],
        "main_concepts": result["main_concepts"],
        "passages_used": [
            {
                "chunk_id": chunks[i]["chunk_id"],
                "page_start": chunks[i]["page_start"],
                "page_end": chunks[i]["page_end"],
                "excerpt": chunks[i]["text"][:200],
            }
            for i in selected_in_order
        ],
        "coverage": {
            "chunks_used": len(selected_in_order),
            "chunks_total": len(chunks),
            "words_used": int(sum(chunks[i]["word_count"] for i in selected_in_order)),
            "words_total_approx": int(sum(c["word_count"] for c in chunks)),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cached": False,
        "warning": " ".join(warnings) if warnings else None,
        "disclaimer": DISCLAIMER,
    }

    if cacheable:
        try:
            cache_file.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.warning("Could not write the summary cache for %s", document_id)

    logger.info(
        "Summarised %s: mode=%s, %d/%d chunks",
        document_id, response["mode"], len(selected_in_order), len(chunks),
    )
    return response


def get_cached_summary(document_id: str) -> dict:
    """Return a previously generated summary without regenerating it."""
    document_service.get_document(document_id)  # may raise DocumentNotFoundError
    cache_file = _cache_path(document_id)
    if not cache_file.exists():
        raise SummarizationError("No summary found. Call POST /summarization/summarize first.")
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SummarizationError("The saved summary is unreadable. Regenerate it with force=true.") from exc
    cached["cached"] = True
    return cached