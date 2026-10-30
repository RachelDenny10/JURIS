"""
Shared constants for JURIS: the six legal document categories used
throughout the dataset, database, ML pipeline, API, and frontend.
"""

from enum import Enum


class DocumentCategory(str, Enum):
    """The six legal document categories used across JURIS."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


CATEGORY_LABELS: dict[str, str] = {
    DocumentCategory.A.value: "Primary Consumer-Related Legislation",
    DocumentCategory.B.value: "Consumer Protection Rules",
    DocumentCategory.C.value: "Consumer Protection Regulations",
    DocumentCategory.D.value: "Special Consumer Rules",
    DocumentCategory.E.value: "Consumer Protection Guidelines",
    DocumentCategory.F.value: "Consumer Law Case Decisions",
}

CATEGORY_FOLDERS: dict[str, str] = {
    DocumentCategory.A.value: "consumer_protection_act",
    DocumentCategory.B.value: "consumer_protection_rules",
    DocumentCategory.C.value: "consumer_protection_regulations",
    DocumentCategory.D.value: "special_consumer_rules",
    DocumentCategory.E.value: "consumer_guidelines",
    DocumentCategory.F.value: "case_decisions",
}