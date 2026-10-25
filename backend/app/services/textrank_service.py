"""
TextRank keyword extraction, implemented directly with spaCy (POS
tagging) and networkx (graph construction + PageRank).

Unlike TF-IDF, TextRank operates on a SINGLE document - it needs no
corpus, since it ranks words by how centrally they co-occur with
other important words within that one document's text.

Algorithm:
  1. POS-tag the text with spaCy; keep only NOUN, PROPN, ADJ tokens
     as candidate keywords (function words carry little topical
     meaning as keywords).
  2. Build a co-occurrence graph: an edge connects two candidate
     words if they appear within WINDOW_SIZE tokens of each other
     anywhere in the candidate sequence.
  3. Run PageRank over this graph to score each candidate word.
  4. Take the top-ranked words. Adjacent top-ranked words in the
     original text are merged into multi-word phrases.
"""

import logging

import networkx as nx
import spacy

logger = logging.getLogger("juris.textrank_service")

CANDIDATE_POS = {"NOUN", "PROPN", "ADJ"}
WINDOW_SIZE = 4

# Legal drafting words that spaCy tags as adjectives but carry no topical meaning.
# Hand-picked from inspecting TextRank output on one Act; extend as needed.
EXCLUDED_WORDS = {
    "such", "other", "said", "same", "certain", "aforesaid", "concerned",
    "respective", "following", "further", "various", "relevant", "many", "more",
}

_spacy_model_cache = None


class TextRankError(Exception):
    """Raised when TextRank cannot be computed (e.g. model not installed, empty text)."""


def _get_spacy_model():
    global _spacy_model_cache
    if _spacy_model_cache is None:
        try:
            _spacy_model_cache = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise TextRankError(
                "spaCy model 'en_core_web_sm' not found. Run scripts/download_nlp_models.py first."
            ) from exc
    return _spacy_model_cache


def _extract_candidates(doc) -> list[tuple[str, int]]:
    """
    Return a list of (lemma, original_token_index) for tokens whose
    POS tag is in CANDIDATE_POS. The original_token_index lets us
    later check whether two candidates were adjacent in the real text.
    """
    candidates = []
    for i, token in enumerate(doc):
                if (
            token.pos_ in CANDIDATE_POS
            and token.is_alpha
            and token.lemma_.lower() not in EXCLUDED_WORDS
        ):
                    candidates.append((token.lemma_.lower(), i))
    return candidates


def _build_cooccurrence_graph(candidates: list[tuple[str, int]]) -> nx.Graph:
    """Build an undirected graph where candidate words within WINDOW_SIZE tokens are connected."""
    graph = nx.Graph()
    words = [c[0] for c in candidates]
    graph.add_nodes_from(set(words))

    for i in range(len(candidates)):
        for j in range(i + 1, min(i + WINDOW_SIZE, len(candidates))):
            word_a, word_b = candidates[i][0], candidates[j][0]
            if word_a != word_b:
                if graph.has_edge(word_a, word_b):
                    graph[word_a][word_b]["weight"] += 1
                else:
                    graph.add_edge(word_a, word_b, weight=1)

    return graph


def _merge_adjacent_keywords(candidates: list[tuple[str, int]], top_words: set[str]) -> list[str]:
    """
    Merge adjacent candidate tokens that are both in top_words into
    multi-word phrases (e.g. "consumer" + "complaint" -> "consumer complaint"),
    preserving the order they first appear in the document.
    """
    phrases: list[str] = []
    current_phrase: list[str] = []
    last_index = None

    for word, index in candidates:
        if word in top_words:
            if last_index is not None and index == last_index + 1:
                current_phrase.append(word)
            else:
                if current_phrase:
                    phrases.append(" ".join(current_phrase))
                current_phrase = [word]
            last_index = index
        else:
            if current_phrase:
                phrases.append(" ".join(current_phrase))
                current_phrase = []
            last_index = None

    if current_phrase:
        phrases.append(" ".join(current_phrase))

    seen = set()
    ordered_unique = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            ordered_unique.append(phrase)

    return ordered_unique


def extract_textrank_keywords(cleaned_text: str, top_n: int = 10) -> list[dict]:
    """
    Run TextRank on a single document's cleaned text and return the
    top-N keywords/phrases with their PageRank scores.
    """
    if not cleaned_text.strip():
        raise TextRankError("Cannot run TextRank on empty text.")

    nlp = _get_spacy_model()
    doc = nlp(cleaned_text)

    candidates = _extract_candidates(doc)
    if len(candidates) < 2:
        raise TextRankError("Not enough candidate keywords (nouns/adjectives) found in this text.")

    graph = _build_cooccurrence_graph(candidates)
    if graph.number_of_edges() == 0:
        raise TextRankError("No co-occurrence relationships found; text may be too short for TextRank.")

    scores = nx.pagerank(graph, weight="weight")

    ranked_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_single_words = {word for word, _ in ranked_words[: top_n * 2]}

    phrases = _merge_adjacent_keywords(candidates, top_single_words)

    phrase_scores = []
    for phrase in phrases:
        words_in_phrase = phrase.split(" ")
        avg_score = sum(scores.get(w, 0.0) for w in words_in_phrase) / len(words_in_phrase)
        phrase_scores.append({"phrase": phrase, "score": round(float(avg_score), 5)})

    phrase_scores.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "TextRank: %d candidates, %d graph nodes, %d phrases extracted",
        len(candidates), graph.number_of_nodes(), len(phrase_scores),
    )

    return phrase_scores[:top_n]
