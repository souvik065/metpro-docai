"""
Integration tests for the ingestion pipeline.
Uses in-memory SQLite + mocked vector store and embedders,
so no external services are needed.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fixtures.generators import write_fixtures


# ── Async fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixtures")
    write_fixtures(d)
    return d


@pytest.fixture
def mock_embedders():
    """Return mocked text and image embedders that return zero vectors."""
    te = MagicMock()
    te.embed.return_value = [0.0] * 384
    te.embed_batch.side_effect = lambda texts: [[0.0] * 384 for _ in texts]

    ie = MagicMock()
    ie.embed_image_bytes.return_value = [0.0] * 512
    ie.embed_image_path.return_value = [0.0] * 512
    ie.embed_text_for_image_search.return_value = [0.0] * 512
    return te, ie


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.upsert_text.return_value = None
    vs.upsert_image.return_value = None
    vs.count.return_value = 0
    return vs


@pytest.fixture
def mock_ocr():
    from src.ocr.engine import OCRResult
    ocr = MagicMock()
    ocr.run.return_value = OCRResult(
        full_text="OCR extracted text from image.",
        regions=[],
        engine_used="mock",
    )
    return ocr


# ── Helpers ───────────────────────────────────────────────────────────────────

async def make_pipeline(session, mock_embedders, mock_vector_store, mock_ocr):
    """Create an IngestionPipeline with all heavy deps mocked."""
    te, ie = mock_embedders
    with patch("src.ingestion.pipeline.get_text_embedder", return_value=te), \
         patch("src.ingestion.pipeline.get_image_embedder", return_value=ie), \
         patch("src.ingestion.pipeline.get_vector_store", return_value=mock_vector_store), \
         patch("src.ingestion.pipeline.OCREngine", return_value=mock_ocr):
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(db_session=session)
        pipeline.ocr = mock_ocr
        pipeline.text_embedder = te
        pipeline.image_embedder = ie
        pipeline.vector_store = mock_vector_store
        return pipeline


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_pdf(fixture_dir, mock_embedders, mock_vector_store, mock_ocr):
    from sqlmodel import SQLModel
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from src.models.schema import Document, Page, Asset

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        pipeline = await make_pipeline(session, mock_embedders, mock_vector_store, mock_ocr)
        pdf_path = fixture_dir / "sample_report.pdf"
        doc = await pipeline.ingest_file(str(pdf_path))

    assert doc is not None
    assert doc.filename == "sample_report.pdf"
    assert doc.file_type.value == "pdf"
    assert doc.status.value == "done"


@pytest.mark.asyncio
async def test_ingest_png(fixture_dir, mock_embedders, mock_vector_store, mock_ocr):
    from sqlmodel import SQLModel
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from src.models.schema import Document

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        pipeline = await make_pipeline(session, mock_embedders, mock_vector_store, mock_ocr)
        png_path = fixture_dir / "chest_xray.png"
        doc = await pipeline.ingest_file(str(png_path))

    assert doc is not None
    assert doc.file_type.value == "image"
    assert doc.status.value == "done"


@pytest.mark.asyncio
async def test_ingest_missing_file(mock_embedders, mock_vector_store, mock_ocr):
    from sqlmodel import SQLModel
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        pipeline = await make_pipeline(session, mock_embedders, mock_vector_store, mock_ocr)
        result = await pipeline.ingest_file("/nonexistent/path/file.pdf")

    assert result is None


@pytest.mark.asyncio
async def test_ingest_folder_counts_docs(fixture_dir, mock_embedders, mock_vector_store, mock_ocr):
    from sqlmodel import SQLModel
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        pipeline = await make_pipeline(session, mock_embedders, mock_vector_store, mock_ocr)
        docs = await pipeline.ingest_folder(str(fixture_dir), recursive=False)

    # At least the PDF and PNG fixture files should be ingested
    assert len(docs) >= 2
