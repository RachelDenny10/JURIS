"""
One-off migration: copy the rows of data/metadata/documents.csv into the
documents table. Safe to run more than once: rows whose document_id
already exists in the database are skipped.

The CSV is never modified or deleted; keep it as a backup.

Run from the backend folder:
    python scripts\\migrate_csv_to_db.py
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.constants import DocumentCategory  # noqa: E402
from app.core.database import get_database_url, get_session, init_db  # noqa: E402
from app.models.document import DocumentRecord  # noqa: E402


def main() -> None:
    settings = get_settings()
    csv_path = settings.data_metadata_path / "documents.csv"

    if not csv_path.exists():
        print(f"No CSV found at {csv_path}. Nothing to migrate.")
        return

    init_db()
    print(f"Database: {get_database_url().split('@')[-1]}")

    migrated = skipped = failed = missing_files = 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"Rows in CSV: {len(rows)}")

    for row in rows:
        document_id = (row.get("document_id") or "").strip()
        try:
            if not document_id:
                raise ValueError("row has no document_id")
            DocumentCategory(row["category"])  # validates the category letter

            timestamp = datetime.fromisoformat(row["upload_timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            with get_session() as session:
                if session.get(DocumentRecord, document_id) is not None:
                    skipped += 1
                    continue
                session.add(
                    DocumentRecord(
                        document_id=document_id,
                        filename=row["filename"],
                        original_filename=row["original_filename"],
                        category=row["category"],
                        title=row["title"],
                        source=row.get("source") or None,
                        date=row.get("date") or None,
                        storage_path=row["storage_path"],
                        text_path=row.get("text_path") or None,
                        upload_timestamp=timestamp,
                        processing_status=row.get("processing_status") or "uploaded",
                        file_size_bytes=int(row["file_size_bytes"]),
                    )
                )
            migrated += 1

            if not (settings.project_root / row["storage_path"]).exists():
                missing_files += 1
                print(f"  warning: PDF file missing on disk for {document_id}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED {document_id or '<no id>'}: {exc}")

    print(f"\nMigrated: {migrated}   Skipped (already present): {skipped}   Failed: {failed}")
    if missing_files:
        print(f"Rows whose PDF file is missing on disk: {missing_files}")


if __name__ == "__main__":
    main()