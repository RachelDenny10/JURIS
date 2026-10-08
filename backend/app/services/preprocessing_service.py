"""
NLP preprocessing service: cleaning, tokenization, stopword removal,
and lemmatization for legal text.

Uses NLTK for tokenization/stopwords and spaCy for lemmatization,
per project Section 5.

IMPORTANT: legal text requires a modified stopword list. Words like
"not", "shall", "must", "under" can be legally load-bearing (e.g.
"shall not be liable" vs "shall be liable"), so we exclude a curated
set of legally meaningful terms from the standard NLTK stopword list
rather than blindly removing every common word. See LEGAL_SAFE_WORDS
below.
"""

import logging
import re

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger("juris.preprocessing_service")

# Words that NLTK's default English stopword list would remove, but
# which can carry legal significance and should be KEPT.
LEGAL_SAFE_WORDS = {
    "not", "no", "nor",
    "shall", "must", "may", "should", "will", "can",
    "against", "before", "after", "under", "over", "between",
    "than", "same", "own",
    "few", "more", "most", "other", "such",
    "only", "own",
    "any", "all", "both",
    "if", "because", "until", "while",
}

_nltk_stopwords_cache: set[str] | None = None
_spacy_model_cache = None


class PreprocessingError(Exception):
    """Raised when preprocessing cannot proceed (e.g. models not installed)."""


def _get_legal_stopwords() -> set[str]:
    """Return the effective stopword set: NLTK defaults minus legal-safe words."""
    global _nltk_stopwords_cache
    if _nltk_stopwords_cache is None:
        try:
            base_stopwords = set(stopwords.words("english"))
        except LookupError as exc:
            raise PreprocessingError(
                "NLTK stopwords corpus not found. Run scripts/download_nlp_models.py first."
            ) from exc
        _nltk_stopwords_cache = base_stopwords - LEGAL_SAFE_WORDS
    return _nltk_stopwords_cache


def _get_spacy_model():
    """Load and cache the spaCy English model."""
    global _spacy_model_cache
    if _spacy_model_cache is None:
        try:
            _spacy_model_cache = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise PreprocessingError(
                "spaCy model 'en_core_web_sm' not found. Run scripts/download_nlp_models.py first."
            ) from exc
    return _spacy_model_cache


def clean_text(raw_text: str) -> str:
    """
    Basic text cleaning: normalize whitespace, remove page-break
    markers inserted during extraction, strip stray control characters.

    Deliberately does NOT lowercase or strip punctuation here - that
    happens at the tokenization stage, since some downstream steps
    (like entity extraction in Phase 9) benefit from original casing.
    """
    text = raw_text.replace("----- PAGE BREAK -----", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def tokenize(cleaned_text: str) -> list[str]:
    """Tokenize cleaned text into words using NLTK, lowercased, punctuation removed."""
    try:
        raw_tokens = word_tokenize(cleaned_text)
    except LookupError as exc:
        raise PreprocessingError(
            "NLTK 'punkt' tokenizer not found. Run scripts/download_nlp_models.py first."
        ) from exc

    tokens = [t.lower() for t in raw_tokens if re.match(r"^[a-zA-Z]+$", t)]
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords, preserving legally meaningful terms (see LEGAL_SAFE_WORDS)."""
    legal_stopwords = _get_legal_stopwords()
    return [t for t in tokens if t not in legal_stopwords]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize a list of tokens using spaCy."""
    nlp = _get_spacy_model()
    # Process as a single space-joined doc for efficiency, since we
    # already have clean lowercase tokens (no need for spaCy's own
    # tokenizer or sentence segmentation here).
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if token.lemma_.strip()]


class PreprocessingResult:
    def __init__(
        self,
        *,
        cleaned_text: str,
        tokens: list[str],
        tokens_no_stopwords: list[str],
        lemmatized_tokens: list[str],
    ):
        self.cleaned_text = cleaned_text
        self.tokens = tokens
        self.tokens_no_stopwords = tokens_no_stopwords
        self.lemmatized_tokens = lemmatized_tokens


def preprocess_text(raw_text: str) -> PreprocessingResult:
    """Run the full preprocessing pipeline: clean -> tokenize -> destop -> lemmatize."""
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmatized = lemmatize(tokens_no_stop)

    logger.info(
        "Preprocessed text: %d raw tokens -> %d after stopword removal -> %d lemmatized",
        len(tokens), len(tokens_no_stop), len(lemmatized),
    )

    return PreprocessingResult(
        cleaned_text=cleaned,
        tokens=tokens,
        tokens_no_stopwords=tokens_no_stop,
        lemmatized_tokens=lemmatized,
    )