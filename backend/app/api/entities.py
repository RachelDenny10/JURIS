"""
Legal entity extraction API.

Phase 9 scope: extract legal entities from a document's extracted text
(rule-based legal patterns + spaCy NER), save the full result, and
return a grouped summary.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import get_settings
from app.schemas.entities import EntityCount, EntityExtractionResponse
from app.services import document_service, entity_service

logger = logging.getLogger("juris.api.entities")

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _entities_paths(document_id: str) -> tuple[Path, Path]:
    """Return (relative_path, absolute_path) of the saved entities file."""
    settings = get_settings()
    relative = Path("data") / "processed" / f"{document_id}_entities.json"
    return relative, settings.project_root / relative


def _build_response(
    document_id: str,
    entities: list[entity_service.Entity],
    top_n: int,
    relative_path: Path,
) -> EntityExtractionResponse:
    summary = entity_service.summarize_entities(entities, top_n_per_label=top_n)
    return EntityExtractionResponse(
        document_id=document_id,
        total_mentions=len(entities),
        mention_counts=summary["mention_counts"],
        unique_counts=summary["unique_counts"],
        entities={
            label: [EntityCount(**item) for item in items]
            for label, items in summary["entities"].items()
        },
        entities_path=str(relative_path),
    )


@router.post("/entities/{document_id}", response_model=EntityExtractionResponse)
def extract_document_entities(document_id: str, top_n: int = Query(default=25, ge=1, le=200)):
    """
    Extract legal entities (Acts, Sections, Rules, Regulations, Courts,
    Commissions, authorities, case numbers/names, dates, organisations,
    persons, locations) from a document's extracted text.

    Requires /analysis/extract/{document_id} to have been run first.
    Long documents can take a while because spaCy processes the full text.
    """
    settings = get_settings()

    try:
        record = document_service.get_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if not record.text_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No extracted text found for this document. Call /analysis/extract/{document_id} first.",
        )

    text_file = settings.project_root / record.text_path
    if not text_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recorded text file is missing on disk.",
        )

    raw_text = text_file.read_text(encoding="utf-8")

    try:
        entities = entity_service.extract_entities(raw_text)
    except entity_service.EntityModelError as exc:
        logger.error("Entity model setup error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except entity_service.EntityExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error extracting entities for document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not extract entities from the document.",
        )

    relative_path, absolute_path = _entities_paths(document_id)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(
        json.dumps(
            {"document_id": document_id, "entities": [asdict(e) for e in entities]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return _build_response(document_id, entities, top_n, relative_path)


@router.get("/entities/{document_id}", response_model=EntityExtractionResponse)
def get_document_entities(document_id: str, top_n: int = Query(default=25, ge=1, le=200)):
    """Return the previously extracted entities for a document (no re-processing)."""
    try:
        document_service.get_document(document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    relative_path, absolute_path = _entities_paths(document_id)
    if not absolute_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No entity extraction output found. Call POST /analysis/entities/{document_id} first.",
        )

    try:
        data = json.loads(absolute_path.read_text(encoding="utf-8"))
        entities = [entity_service.Entity(**item) for item in data["entities"]]
    except Exception:
        logger.exception("Could not read saved entities for document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Saved entity file is unreadable. Re-run POST /analysis/entities/{document_id}.",
        )

    return _build_response(document_id, entities, top_n, relative_path)