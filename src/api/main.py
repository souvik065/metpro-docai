"""MacPro AI — FastAPI application entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.utils.database import init_db
from src.utils.helpers import get_logger, ensure_dir
from src.api.routes import ingest, query, health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("MacPro AI starting up…")
    ensure_dir(settings.data_dir)
    ensure_dir(settings.input_dir)
    ensure_dir(settings.output_dir)
    await init_db()
    logger.info("MacPro AI ready.")
    yield
    logger.info("MacPro AI shutting down.")


app = FastAPI(
    title="MacPro AI — Medical RAG API",
    description=(
        "Multimodal medical document ingestion, indexing, and retrieval. "
        "Supports PDFs, images, scans, and DICOM files."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
