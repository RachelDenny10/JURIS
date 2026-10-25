"""
ORM model for the documents table (Phase 10).

One row per stored legal document. PDFs themselves stay on disk;
this table holds metadata and the paths to those files.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(1), index=True)
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    date: Mapped[str | None] = mapped_column(String(50), nullable=True)
    storage_path: Mapped[str] = mapped_column(Text)
    text_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    processing_status: Mapped[str] = mapped_column(String(32), index=True, default="uploaded")
    file_size_bytes: Mapped[int] = mapped_column(BigInteger)