"""MacPro AI — Health check endpoints."""
from fastapi import APIRouter
from src.indexing.vector_store import get_vector_store
from src.models.schema import IndexStats

router = APIRouter()


@router.get("/", summary="Health check")
async def health():
    return {"status": "ok", "service": "MacPro AI"}


@router.get("/stats", response_model=IndexStats, summary="Index statistics")
async def stats():
    vs = get_vector_store()
    vector_count = vs.count()
    return IndexStats(documents=0, pages=0, assets=0, vectors=vector_count)
