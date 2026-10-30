
"""
Bilingual NLP preprocessing service for English and Hindi legal documents.

Pipeline:
Unicode normalization -> cleaning -> tokenization ->
language-aware stopword removal -> English lemmatization.
"""

import logging
import re
import unicodedata

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger("juris.preprocessing_service")

LEGAL_SAFE_WORDS = {
    "not", "no", "nor",
    "shall", "must", "may", "should", "will", "can",
    "against", "before", "after", "under", "over", "between",
    "than", "same", "own", "few", "more", "most", "other",
    "such", "only", "any", "all", "both", "if", "because",
    "until", "while",
}

_nltk_stopwords_cache = None
_spacy_model_cache = None


class PreprocessingError(Exception):
    """Raised when preprocessing cannot proceed."""


def _get_legal_stopwords():
    global _nltk_stopwords_cache

    if _nltk_stopwords_cache is None:
        try:
            base = set(stopwords.words("english"))
        except LookupError as exc:
            raise PreprocessingError(
                "NLTK stopwords corpus not found."
            ) from exc

        _nltk_stopwords_cache = base - LEGAL_SAFE_WORDS

    return _nltk_stopwords_cache


def _get_spacy_model():
    global _spacy_model_cache

    if _spacy_model_cache is None:
        try:
            _spacy_model_cache = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise PreprocessingError(
                "spaCy model 'en_core_web_sm' not found."
            ) from exc

    return _spacy_model_cache


def clean_text(raw_text: str) -> str:
    """Normalize Unicode and clean extracted text without deleting Hindi."""

    text = unicodedata.normalize("NFC", raw_text)
    text = text.replace("----- PAGE BREAK -----", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(cleaned_text: str) -> list[str]:
    """
    Retain English alphabetic tokens and Hindi Devanagari tokens.
    """

    try:
        raw_tokens = word_tokenize(cleaned_text)
    except LookupError as exc:
        raise PreprocessingError(
            "NLTK tokenizer resources not found."
        ) from exc

    tokens = []

    for token in raw_tokens:
        token = token.lower()

        # English alphabetic words
        if re.fullmatch(r"[a-z]+", token):
            tokens.append(token)

        # Hindi Devanagari words, including vowel marks and nukta
        elif re.fullmatch(r"[\u0900-\u097F]+", token):
            tokens.append(token)

    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove English stopwords only; preserve Hindi tokens."""

    english_stopwords = _get_legal_stopwords()

    return [
        token for token in tokens
        if not re.fullmatch(r"[a-z]+", token)
        or token not in english_stopwords
    ]



def lemmatize(tokens: list[str]) -> list[str]:
    """
    Lemmatize English tokens while preserving Hindi tokens
    and the original sequence of all tokens.
    """
    nlp = _get_spacy_model()

    english_positions = [
        i for i, token in enumerate(tokens)
        if re.fullmatch(r"[a-z]+", token)
    ]

    if not english_positions:
        return tokens.copy()

    english_tokens = [tokens[i] for i in english_positions]
    doc = nlp(" ".join(english_tokens))

    result = tokens.copy()

    for position, token in zip(english_positions, doc):
        if token.lemma_.strip():
            result[position] = token.lemma_

    return result


class PreprocessingResult:
    def __init__(
        self,
        *,
        cleaned_text,
        tokens,
        tokens_no_stopwords,
        lemmatized_tokens,
    ):
        self.cleaned_text = cleaned_text
        self.tokens = tokens
        self.tokens_no_stopwords = tokens_no_stopwords
        self.lemmatized_tokens = lemmatized_tokens


def preprocess_text(raw_text: str) -> PreprocessingResult:

    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmatized = lemmatize(tokens_no_stop)

    logger.info(
        "Preprocessed bilingual text: %d tokens -> %d after stopwords -> %d final",
        len(tokens),
        len(tokens_no_stop),
        len(lemmatized),
    )

    return PreprocessingResult(
        cleaned_text=cleaned,
        tokens=tokens,
        tokens_no_stopwords=tokens_no_stop,
        lemmatized_tokens=lemmatized,
    )
