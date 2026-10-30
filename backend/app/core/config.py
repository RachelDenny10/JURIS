"""
Application configuration.

Centralizes all environment-driven settings and filesystem paths so
that no other module reads os.environ directly or hardcodes paths.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# JURIS project root, resolved from this file's location:
# backend/app/core/config.py -> core -> app -> backend -> JURIS
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed application settings, loaded from environment variables / .env."""

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str = ""
    openai_api_key: str = ""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    storage_dir: str = "storage/documents"
    faiss_index_dir: str = "models/vector_store"

    max_upload_size_mb: int = 20

    # --- OCR fallback (Phase 3 extension) ---
    tesseract_cmd: str = ""
    # Minimum characters on a page for PyMuPDF's extraction to count
    # as "meaningful text" for that page.
    ocr_page_char_threshold: int = 20
    # If fewer than this fraction of pages are "meaningful", OCR is
    # triggered for the non-meaningful pages.
    ocr_trigger_meaningful_ratio: float = 0.5
           # --- Embeddings / chunking (Phase 11) ---
    # embedding_model_name is defined above (EMBEDDING_MODEL_NAME in .env)
    chunk_target_words: int = 200
    chunk_overlap_words: int = 40
    embedding_batch_size: int = 32

    # --- LLM (Phase 13) ---
    # openai_api_key is defined above (OPENAI_API_KEY in .env)
    llm_provider: str = "none"  # none | gemini | openai | anthropic
    llm_model: str = ""  # empty = provider default (see llm_service.DEFAULT_MODELS)
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    llm_timeout_seconds: int = 90
    llm_max_output_tokens: int = 1200
    summary_context_words: int = 3000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        return BASE_DIR

    @property
    def storage_path(self) -> Path:
        """Absolute path to where uploaded PDFs are stored."""
        return BASE_DIR / self.storage_dir

    @property
    def data_raw_path(self) -> Path:
        return BASE_DIR / "data" / "raw"

    @property
    def data_metadata_path(self) -> Path:
        return BASE_DIR / "data" / "metadata"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (avoids re-parsing .env repeatedly)."""
    return Settings()