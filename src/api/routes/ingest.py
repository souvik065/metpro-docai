"""
MacPro AI — Ingestion endpoints.

POST /ingest/folder   — ingest a folder path (background task)
POST /ingest/file     — upload and ingest a single file
GET  /ingest/status/{job_id} — check job status (simplified)
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from sqlmodel.ext.asyncio.session import AsyncSession

from config.settings import settings
from src.ingestion.pipeline import IngestionPipeline
from src.models.schema import IngestRequest, IngestResponse
from src.utils.database import get_session
from src.utils.helpers import ensure_dir, get_logger

logger = get_logger(__name__)
router = APIRouter()

# Simple in-memory job tracker (swap for Redis in production)
_jobs: dict[str, str] = {}


async def _run_folder_ingest(folder: str, recursive: bool, job_id: str):
    _jobs[job_id] = "processing"
    try:
        from src.utils.database import get_session_factory
        factory = get_session_factory()
        async with factory() as session:
            pipeline = IngestionPipeline(db_session=session)
            docs = await pipeline.ingest_folder(folder, recursive=recursive)
        _jobs[job_id] = f"done:{len(docs)}"
        logger.info(f"Job {job_id} completed: {len(docs)} documents ingested.")
    except Exception as e:
        _jobs[job_id] = f"failed:{e}"
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)


@router.post("/folder", response_model=IngestResponse, summary="Ingest a folder of medical files")
async def ingest_folder(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    folder = Path(request.folder_path)
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")

    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".webp"}
    pattern = "**/*" if request.recursive else "*"
    files = [p for p in folder.glob(pattern) if p.suffix.lower() in supported]

    job_id = str(uuid.uuid4())
    _jobs[job_id] = "queued"
    background_tasks.add_task(
        _run_folder_ingest, str(folder), request.recursive, job_id
    )
    return IngestResponse(job_id=job_id, message="Ingestion started.", files_queued=len(files))


@router.post("/file", summary="Upload and ingest a single file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    upload_dir = ensure_dir(settings.input_dir / "uploads")
    dest = upload_dir / file.filename
    with open(str(dest), "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = "queued"

    async def _run(path: str, jid: str):
        _jobs[jid] = "processing"
        try:
            from src.utils.database import get_session_factory
            factory = get_session_factory()
            async with factory() as session:
                pipeline = IngestionPipeline(db_session=session)
                doc = await pipeline.ingest_file(path)
            _jobs[jid] = f"done:{doc.id if doc else 'none'}"
        except Exception as e:
            _jobs[jid] = f"failed:{e}"

    background_tasks.add_task(_run, str(dest), job_id)
    return {"job_id": job_id, "filename": file.filename, "message": "File queued for ingestion."}


@router.get("/status/{job_id}", summary="Check ingestion job status")
async def job_status(job_id: str):
    status = _jobs.get(job_id, "not_found")
    return {"job_id": job_id, "status": status}
