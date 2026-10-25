"""
Legal entity extraction service.

Phase 9 scope: find legal entities in a document's extracted text using
two layers:

  1. Rule-based patterns (regular expressions) for legal citations that
     generic NER does not recognise: Acts, Sections, Rules, Regulations,
     named Rules/Regulations/Guidelines, Courts, Commissions, authorities,
     case numbers, case names, and dates.
  2. spaCy NER (en_core_web_sm) for general entities: organisations,
     persons, locations, and textual dates.

Where the two layers overlap, rule-based matches always win, and among
rule-based matches the longest span wins.

Offsets (start/end) are relative to the CLEANED text (page-break markers
removed, whitespace collapsed), which is the same cleaning applied by the
preprocessing service.

No accuracy claim is made: there is no labelled evaluation set. The
patterns are precise on well-formed citations; spaCy output on legal
text can be noisy.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import spacy

logger = logging.getLogger("juris.entity_service")

PAGE_BREAK_TEXT = "----- PAGE BREAK -----"
SPACY_CHUNK_SIZE = 200_000

_nlp_cache = None


class EntityExtractionError(Exception):
    """Raised when the input text cannot be processed (e.g. it is empty)."""


class EntityModelError(EntityExtractionError):
    """Raised when the spaCy model cannot be loaded."""


@dataclass
class Entity:
    text: str      # normalised text of the entity
    label: str     # e.g. SECTION, ACT, COURT
    start: int     # character offset in the cleaned text
    end: int
    source: str    # "rule" or "spacy"


# --------------------------------------------------------------------------
# Rule-based patterns
# --------------------------------------------------------------------------

_YEAR = r"(?:1[89]\d{2}|20\d{2})"
_CAP_WORD = r"(?:[A-Z][A-Za-z&'\-]*|\([A-Z][A-Za-z&'\- ]*\))"

# A section/rule/regulation number such as 12, 12A, 2(7), 35(1)(a).
# 1-3 digits, so years like 2019 are never mistaken for a section number.
_NUM = r"\d{1,3}(?!\d)[A-Z]?(?:\([0-9A-Za-z]{1,4}\))*"
_NUM_LIST = _NUM + r"(?:\s*(?:,|and|or|to)\s*" + _NUM + r")*"

SECTION_PATTERN = re.compile(r"\b[Ss]ections?\s+" + _NUM_LIST)
RULE_PATTERN = re.compile(r"\b[Rr]ules?\s+" + _NUM_LIST)
REGULATION_PATTERN = re.compile(r"\b[Rr]egulations?\s+" + _NUM_LIST)

# Named instruments: "Consumer Protection Act, 2019",
# "Consumer Protection (E-Commerce) Rules, 2020", "... Regulations, 2020".
INSTRUMENT_PATTERN = re.compile(
    r"\b" + _CAP_WORD
    + r"(?:\s+(?:" + _CAP_WORD + r"|of|and|for|the)){0,10}?"
    + r"\s+(?P<kind>Act|Rules|Regulations)\s*,?\s*" + _YEAR + r"(?!\d)"
    + r"(?:\s*\(\s*\d{1,3}\s+of\s+" + _YEAR + r"\s*\))?"
)
INSTRUMENT_LABELS = {
    "Act": "ACT",
    "Rules": "RULES_TITLE",
    "Regulations": "REGULATIONS_TITLE",
}

# "Guidelines for Prevention and Regulation of Dark Patterns, 2023"
GUIDELINES_PATTERN = re.compile(
    r"\bGuidelines\s+for\s+[A-Z][A-Za-z&'\-]*"
    r"(?:\s+(?:[A-Z][A-Za-z&'\-]*|of|and|for|the|in|on)){0,14}?"
    r"\s*,\s*" + _YEAR
)

COURT_PATTERNS = [
    re.compile(r"\bSupreme Court of India\b"),
    re.compile(r"\bSupreme Court\b"),
    re.compile(r"\bHigh Court of [A-Z][a-z]+(?:\s+(?:and\s+)?[A-Z][a-z]+){0,2}"),
    re.compile(r"\b(?!The\b)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+High Court\b"),
    re.compile(r"\bDistrict (?:and Sessions )?Court\b"),
]

COMMISSION_PATTERNS = [
    re.compile(r"\b(?:National|State|District)\s+Consumer\s+Disputes\s+Redressal\s+Commissions?\b"),
    re.compile(r"\b(?:National|State|District)\s+Commissions?\b"),
    re.compile(r"\bConsumer\s+Disputes\s+Redressal\s+Commissions?\b"),
    re.compile(r"\bConsumer\s+Commissions?\b"),
    re.compile(r"\bCompetition\s+Commission\s+of\s+India\b"),
    re.compile(r"\bNCDRC\b"),
]

CODE_PATTERN = re.compile(
    r"\b(?:Indian Penal Code|Code of Civil Procedure|Code of Criminal Procedure)"
    r"(?:\s*,?\s*(?:18|19|20)\d{2}(?!\d))?"
)

AUTHORITY_PHRASES = [
    "Central Council",
    "State Council",
    "District Council",
    "Central Consumer Protection Authority",
    "Central Consumer Protection Council",
    "State Consumer Protection Council",
    "District Consumer Protection Council",
    "Consumer Protection Council",
    "Central Authority",
    "Central Government",
    "State Government",
    "Government of India",
    "Director General",
    "Director-General",
    "Bureau of Indian Standards",
    "Reserve Bank of India",
    "Securities and Exchange Board of India",
    "Food Safety and Standards Authority of India",
    "Department of Consumer Affairs",
    "Ministry of Consumer Affairs, Food and Public Distribution",
    "Advertising Standards Council of India",
    "CCPA",
]
AUTHORITY_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(p) for p in sorted(set(AUTHORITY_PHRASES), key=len, reverse=True))
    + r")\b"
)

# "Consumer Complaint No. 636 of 2015", "CC No. 636 of 2015"
_CASE_NO_TAIL = r"\s+No[s]?\.?\s*\d+(?:\s*[/-]\s*\d+)?\s+of\s+(?:19|20)\d{2}"
CASE_NUMBER_PATTERNS = [
    re.compile(
        r"\b(?:(?:Consumer|Civil|Criminal|Revision|First|Second|Special\s+Leave|Writ|"
        r"Review|Original|Execution|Miscellaneous|Transfer|Contempt|Arbitration|"
        r"Interlocutory)\s+)?(?:Appeal|Petition|Complaint|Application|Suit|Case)"
        r"(?:\s*\([A-Za-z.]+\))?" + _CASE_NO_TAIL
    ),
    re.compile(r"\b(?:CC|FA|RP|SLP|WP|CA|OP|EA|MA|IA)(?:\s*\([A-Za-z.]+\))?" + _CASE_NO_TAIL),
]

# "Paras Jain v. Amazon Seller Services"
_PARTY_ABBR = r"(?:Pvt\.|Ltd\.|Co\.|Inc\.|Corp\.|Ors\.|Anr\.|Bros\.)"
_PARTY_WORD = r"(?:" + _PARTY_ABBR + r"|[A-Z][A-Za-z0-9&'\-]*|of|and|for|the)"
CASE_NAME_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9&'\-]*(?:\s+" + _PARTY_WORD + r"){0,6}?"
    r"\s+(?:v\.|vs\.?|versus)\s+"
    r"(?:" + _PARTY_ABBR + r"|[A-Z][A-Za-z0-9&'\-]*)(?:\s+" + _PARTY_WORD + r"){0,6}"
)

_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
)
DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?" + _MONTHS + r",?\s+(?:19|20)\d{2}\b"),
    re.compile(r"\b" + _MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b"),
    re.compile(r"\b" + _MONTHS + r",?\s+(?:19|20)\d{2}\b"),
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2}\b"),
]

# Words that often start a sentence in front of a title or case name and
# should not be part of the entity ("Under the Consumer Protection Act, 2019").
_LEADING_NOISE = {
    "the", "under", "in", "of", "by", "as", "see", "for", "and", "or", "to",
    "from", "with", "subject", "notwithstanding", "provided", "whereas",
    "where", "if", "this", "that", "any", "an", "a", "such", "said",
    "existing", "other", "all", "every", "each", "on", "at", "per",
    "pursuant", "after", "before", "within", "these", "those", "which",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def clean_text_for_entities(raw_text: str) -> str:
    """Remove page-break markers and collapse whitespace."""
    text = raw_text.replace(PAGE_BREAK_TEXT, " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_reference(value: str) -> str:
    """'section 12 (1)' -> 'Section 12(1)'."""
    value = _normalize_whitespace(value)
    value = re.sub(r"\s+\(", "(", value)
    return value[0].upper() + value[1:]


def _trim_leading_noise(value: str, start: int) -> tuple[str, int]:
    """Strip sentence-starter words from the front of a match, adjusting the start offset."""
    while True:
        match = re.match(r"([A-Za-z']+)\s+", value)
        if match and match.group(1).lower() in _LEADING_NOISE:
            value = value[match.end():]
            start += match.end()
        else:
            return value, start


def _add_matches(
    found: list,
    text: str,
    patterns: list,
    label: str,
    normalizer: Callable[[str], str] = _normalize_whitespace,
) -> None:
    for pattern in patterns:
        for match in pattern.finditer(text):
            found.append(Entity(normalizer(match.group()), label, match.start(), match.end(), "rule"))


def _strip_leading_caps_heading(value: str, start: int) -> tuple[str, int]:
    # Drop leading ALL-CAPS heading words (for example STATEMENT OF OBJECTS AND REASONS before a title).
    match = re.match(r"(?:[A-Z]{2,}[,.:;]?\s+)+", value)
    if match and match.end() < len(value):
        return value[match.end():], start + match.end()
    return value, start


def _rule_based_entities(text: str) -> list[Entity]:
    found: list[Entity] = []

    _add_matches(found, text, [SECTION_PATTERN], "SECTION", _normalize_reference)
    _add_matches(found, text, [RULE_PATTERN], "RULE", _normalize_reference)
    _add_matches(found, text, [REGULATION_PATTERN], "REGULATION", _normalize_reference)
    _add_matches(found, text, [GUIDELINES_PATTERN], "GUIDELINES_TITLE")
    _add_matches(found, text, COURT_PATTERNS, "COURT")
    _add_matches(found, text, COMMISSION_PATTERNS, "COMMISSION")
    _add_matches(found, text, [AUTHORITY_PATTERN], "AUTHORITY")
    _add_matches(found, text, [CODE_PATTERN], "CODE")
    _add_matches(found, text, CASE_NUMBER_PATTERNS, "CASE_NUMBER")
    _add_matches(found, text, DATE_PATTERNS, "DATE")

    for match in INSTRUMENT_PATTERN.finditer(text):
        trimmed, start = _strip_leading_caps_heading(match.group(), match.start())
        trimmed, start = _trim_leading_noise(trimmed, start)
        if re.match(r"^(?:Act|Rules|Regulations)\b", trimmed):
            continue  # nothing left but the word "Act"/"Rules"
        canonical = re.sub(r"\s*\(\s*\d{1,3}\s+of\s+\d{4}\s*\)\s*$", "", trimmed)
        canonical = _normalize_whitespace(canonical)
        canonical = re.sub(r"\s*,?\s*(\d{4})$", r", \1", canonical)
        label = INSTRUMENT_LABELS[match.group("kind")]
        found.append(Entity(canonical, label, start, match.end(), "rule"))

    for match in CASE_NAME_PATTERN.finditer(text):
        trimmed, start = _trim_leading_noise(match.group(), match.start())
        trimmed = _normalize_whitespace(trimmed).rstrip(" ,;:")
        trimmed = re.sub(r"(?:\s+(?:of|and|for|the))+$", "", trimmed)
        if re.match(r"^[A-Z]", trimmed) and re.search(r"\s(?:v\.|vs\.?|versus)\s", trimmed):
            found.append(Entity(trimmed, "CASE_NAME", start, match.end(), "rule"))

    return found


# --------------------------------------------------------------------------
# spaCy layer
# --------------------------------------------------------------------------

SPACY_LABEL_MAP = {
    "ORG": "ORGANIZATION",
    # PERSON intentionally not mapped: spaCy PERSON on statutes is mostly noise.
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
}

_SPACY_NOISE = {
    "act", "the act", "rules", "regulations", "commission", "court",
    "section", "chapter", "schedule", "state", "central",
    "government", "central government", "state government",
    # headings, footers and generic words that spaCy mislabels in statutes
    "indiacode", "gazette", "notifn", "notification", "vide notifn", "sec",
    "bill", "manner", "vacancies", "offences", "regulator", "house", "houses",
    "chairperson", "commissioners", "benches", "first schedule",
    "judicial magistrate", "gazette of india, extraordinary",
}
_YEAR_SEARCH = re.compile(r"(?:19|20)\d{2}")


def _get_nlp():
    """Load and cache the spaCy model (parser/lemmatizer are not needed for NER)."""
    global _nlp_cache
    if _nlp_cache is None:
        try:
            _nlp_cache = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except OSError as exc:
            raise EntityModelError(
                "spaCy model 'en_core_web_sm' not found. Run scripts/download_nlp_models.py first."
            ) from exc
    return _nlp_cache


def _chunk_text(text: str, size: int = SPACY_CHUNK_SIZE) -> list[tuple[int, str]]:
    """Split long text into (offset, chunk) pieces, preferring sentence boundaries."""
    chunks: list[tuple[int, str]] = []
    position = 0
    length = len(text)
    while position < length:
        end = min(position + size, length)
        if end < length:
            split = text.rfind(". ", position, end)
            if split > position:
                end = split + 2
        chunks.append((position, text[position:end]))
        position = end
    return chunks


def _keep_spacy_entity(value: str, label: str) -> bool:
    if len(value) < 3 or len(value) > 100:
        return False
    if not re.search(r"[A-Za-z]", value):
        return False
    if re.search("[\u2014\u2013\u00e2]", value):
        return False  # heading fragments with dashes or encoding debris
    if value.lower() in _SPACY_NOISE:
        return False
    if label == "DATE" and not _YEAR_SEARCH.search(value):
        return False
    return True


def _spacy_entities(text: str) -> list[Entity]:
    nlp = _get_nlp()
    found: list[Entity] = []
    for offset, chunk in _chunk_text(text):
        doc = nlp(chunk)
        for ent in doc.ents:
            label = SPACY_LABEL_MAP.get(ent.label_)
            if label is None:
                continue
            value = _normalize_whitespace(ent.text)
            start = offset + ent.start_char
            if value.lower().startswith("the "):
                value = value[4:]
                start += 4
            if not _keep_spacy_entity(value, label):
                continue
            found.append(Entity(value, label, start, offset + ent.end_char, "spacy"))
    return found


# --------------------------------------------------------------------------
# Combining and summarising
# --------------------------------------------------------------------------

def _resolve_overlaps(candidates: list[Entity], text_length: int) -> list[Entity]:
    """
    Keep a non-overlapping set of entities. Rule-based matches beat spaCy
    matches; within the same source, longer spans beat shorter ones.
    """
    ordered = sorted(
        candidates,
        key=lambda e: (0 if e.source == "rule" else 1, -(e.end - e.start), e.start),
    )
    occupied = bytearray(text_length)
    accepted: list[Entity] = []
    for entity in ordered:
        if entity.end <= entity.start:
            continue
        if occupied.find(1, entity.start, entity.end) != -1:
            continue
        occupied[entity.start:entity.end] = b"\x01" * (entity.end - entity.start)
        accepted.append(entity)
    accepted.sort(key=lambda e: e.start)
    return accepted


def extract_entities(raw_text: str) -> list[Entity]:
    """Run rule-based + spaCy extraction on a document's extracted text."""
    text = clean_text_for_entities(raw_text)
    if not text:
        raise EntityExtractionError("Cannot extract entities from empty text.")

    rule_entities = _rule_based_entities(text)
    spacy_entities = _spacy_entities(text)
    resolved = _resolve_overlaps(rule_entities + spacy_entities, len(text))

    logger.info(
        "Entity extraction: %d rule-based + %d spaCy candidates -> %d kept (text length %d)",
        len(rule_entities), len(spacy_entities), len(resolved), len(text),
    )
    return resolved


def summarize_entities(entities: list[Entity], top_n_per_label: int = 25) -> dict:
    """Group entities by label with mention counts, most frequent first."""
    by_label: dict[str, Counter] = {}
    for entity in entities:
        by_label.setdefault(entity.label, Counter())[entity.text] += 1

    return {
        "entities": {
            label: [{"text": text, "count": count} for text, count in counter.most_common(top_n_per_label)]
            for label, counter in sorted(by_label.items())
        },
        "mention_counts": {label: sum(c.values()) for label, c in sorted(by_label.items())},
        "unique_counts": {label: len(c) for label, c in sorted(by_label.items())},
    }