"""
One-off patch for app/services/entity_service.py (Phase 9 refinements).

Each edit is applied only if its target text is found the expected number
of times. Already-applied edits are detected and skipped, so it is safe to
run twice. Run from the backend folder:
    python scripts/patch_entities.py
"""

from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "app" / "services" / "entity_service.py"
BACKUP = Path(__file__).resolve().parents[2] / "data" / "processed" / "entity_service.py.bak"

# (name, old text, new text, expected match count, marker proving it is applied)
EDITS = [
    (
        "case names: lowercase 'v.' only",
        r'''v\.|V\.|vs\.?|versus''',
        r'''v\.|vs\.?|versus''',
        2,
        r'''(?:v\.|vs\.?|versus)''',
    ),
    (
        "spaCy: drop PERSON",
        r'''    "PERSON": "PERSON",
''',
        r'''    # PERSON intentionally not mapped: spaCy PERSON on statutes is mostly noise.
''',
        1,
        "PERSON intentionally not mapped",
    ),
    (
        "spaCy: larger noise list",
        r'''_SPACY_NOISE = {
    "act", "the act", "rules", "regulations", "commission", "court",
    "section", "chapter", "schedule", "state", "central",
    "government", "central government", "state government",
}
''',
        r'''_SPACY_NOISE = {
    "act", "the act", "rules", "regulations", "commission", "court",
    "section", "chapter", "schedule", "state", "central",
    "government", "central government", "state government",
    # headings, footers and generic words that spaCy mislabels in statutes
    "indiacode", "gazette", "notifn", "notification", "vide notifn", "sec",
    "bill", "manner", "vacancies", "offences", "regulator", "house", "houses",
    "chairperson", "commissioners", "benches", "first schedule",
    "judicial magistrate", "gazette of india, extraordinary",
}
''',
        1,
        '"indiacode"',
    ),
    (
        "spaCy: reject heading fragments with dashes",
        r'''    if not re.search(r"[A-Za-z]", value):
        return False
    if value.lower() in _SPACY_NOISE:
        return False
''',
        r'''    if not re.search(r"[A-Za-z]", value):
        return False
    if re.search("[\u2014\u2013\u00e2]", value):
        return False  # heading fragments with dashes or encoding debris
    if value.lower() in _SPACY_NOISE:
        return False
''',
        1,
        "u2014",
    ),
    (
        "spaCy: strip leading 'the'",
        r'''            value = _normalize_whitespace(ent.text)
            if not _keep_spacy_entity(value, label):
                continue
            found.append(
                Entity(value, label, offset + ent.start_char, offset + ent.end_char, "spacy")
            )
''',
        r'''            value = _normalize_whitespace(ent.text)
            start = offset + ent.start_char
            if value.lower().startswith("the "):
                value = value[4:]
                start += 4
            if not _keep_spacy_entity(value, label):
                continue
            found.append(Entity(value, label, start, offset + ent.end_char, "spacy"))
''',
        1,
        "value = value[4:]",
    ),
    (
        "helper: strip ALL-CAPS heading words",
        r'''def _rule_based_entities(text: str) -> list[Entity]:
''',
        r'''def _strip_leading_caps_heading(value: str, start: int) -> tuple[str, int]:
    # Drop leading ALL-CAPS heading words (for example STATEMENT OF OBJECTS AND REASONS before a title).
    match = re.match(r"(?:[A-Z]{2,}[,.:;]?\s+)+", value)
    if match and match.end() < len(value):
        return value[match.end():], start + match.end()
    return value, start


def _rule_based_entities(text: str) -> list[Entity]:
''',
        1,
        "def _strip_leading_caps_heading",
    ),
    (
        "titles: use the heading stripper",
        r'''    for match in INSTRUMENT_PATTERN.finditer(text):
        trimmed, start = _trim_leading_noise(match.group(), match.start())
''',
        r'''    for match in INSTRUMENT_PATTERN.finditer(text):
        trimmed, start = _strip_leading_caps_heading(match.group(), match.start())
        trimmed, start = _trim_leading_noise(trimmed, start)
''',
        1,
        "_strip_leading_caps_heading(match.group()",
    ),
    (
        "codes + council phrases",
        r'''AUTHORITY_PHRASES = [
''',
        r'''CODE_PATTERN = re.compile(
    r"\b(?:Indian Penal Code|Code of Civil Procedure|Code of Criminal Procedure)"
    r"(?:\s*,?\s*(?:18|19|20)\d{2}(?!\d))?"
)

AUTHORITY_PHRASES = [
    "Central Council",
    "State Council",
    "District Council",
''',
        1,
        "CODE_PATTERN = re.compile",
    ),
    (
        "codes: register the pattern",
        r'''    _add_matches(found, text, [AUTHORITY_PATTERN], "AUTHORITY")
''',
        r'''    _add_matches(found, text, [AUTHORITY_PATTERN], "AUTHORITY")
    _add_matches(found, text, [CODE_PATTERN], "CODE")
''',
        1,
        '[CODE_PATTERN], "CODE"',
    ),
]


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.write_text(text, encoding="utf-8")

    applied = skipped = missing = 0
    for name, old, new, expected, marker in EDITS:
        if marker in text:
            print(f"[skip]    {name} (already applied)")
            skipped += 1
        elif text.count(old) == expected:
            text = text.replace(old, new)
            print(f"[applied] {name}")
            applied += 1
        else:
            print(f"[MISSING] {name}: found {text.count(old)} match(es), expected {expected}")
            missing += 1

    if applied:
        TARGET.write_text(text, encoding="utf-8")
    print(f"\nDone: {applied} applied, {skipped} skipped, {missing} missing.")
    print(f"Backup saved to {BACKUP}")


if __name__ == "__main__":
    main()