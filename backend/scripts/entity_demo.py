"""
Offline sanity test for the entity extraction service (no server needed).

The passage below is DEMO TEXT written to exercise the patterns. It is
not an authentic legal document.

Run from the backend folder:
    python scripts\\entity_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import entity_service  # noqa: E402

SAMPLE_TEXT = (
    "In exercise of the powers conferred by section 101 of the Consumer Protection Act, 2019 "
    "(35 of 2019), the Central Government hereby makes the following rules, namely: the "
    "Consumer Protection (E-Commerce) Rules, 2020. A complaint under Section 2(7) read with "
    "Section 35(1)(a) shall be filed before the District Commission. An appeal lies to the "
    "State Commission under section 41 and to the National Consumer Disputes Redressal "
    "Commission under Section 51. See Rule 4 and Regulation 7 of the Consumer Protection "
    "(Consumer Commission Procedure) Regulations, 2020. In Consumer Complaint No. 636 of 2015, "
    "decided on 22nd September, 2021, the Supreme Court of India and the Delhi High Court "
    "considered Paras Jain v. Amazon Seller Services. The Central Consumer Protection Authority "
    "issued the Guidelines for Prevention and Regulation of Dark Patterns, 2023 on 30-11-2023."
)


def main() -> None:
    entities = entity_service.extract_entities(SAMPLE_TEXT)
    print(f"{'LABEL':<20}{'SOURCE':<8}TEXT")
    print("-" * 70)
    for entity in entities:
        print(f"{entity.label:<20}{entity.source:<8}{entity.text}")
    print("-" * 70)
    print(f"Total entities: {len(entities)}")


if __name__ == "__main__":
    main()