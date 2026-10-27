"""
JURIS backend entrypoint.

Registers all API routers (documents, analysis, classification,
entities) and initialises storage on startup.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analysis import router as analysis_router
from app.api.classification import router as classification_router
from app.api.documents import router as documents_router
from app.api.entities import router as entities_router
from app.core.config import get_settings
from app.services.document_service import ensure_storage_ready

from app.api.embeddings import router as embeddings_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("juris")

settings = get_settings()

app = FastAPI(
    title="JURIS API",
    description="AI-Powered Legal Knowledge Management and Text Mining System",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(analysis_router)
app.include_router(classification_router)
app.include_router(entities_router)
app.include_router(embeddings_router)


@app.on_event("startup")
def on_startup():
    """Ensure metadata storage and category folders exist before serving requests."""
    ensure_storage_ready()
    logger.info("JURIS backend startup complete")


@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": "JURIS backend",
        "environment": settings.app_env,
    }


@app.get("/")
def root():
    return {"message": "JURIS backend is running. See /health or /docs."}