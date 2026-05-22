"""
MacPro AI — Query endpoint.

POST /query   — natural-language query → answer + sources
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from config.settings import settings
from src.models.schema import QueryResponse
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.synthesizer import LLMSynthesizer
from src.utils.helpers import get_logger

logger = get_logger(__name__)
router = APIRouter()

_retrieval_pipeline: Optional[RetrievalPipeline] = None
_synthesizer: Optional[LLMSynthesizer] = None


def get_retrieval_pipeline() -> RetrievalPipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline


def get_synthesizer() -> LLMSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = LLMSynthesizer()
    return _synthesizer


class QueryRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None
    modality: Optional[str] = None
    top_k: Optional[int] = None


@router.post("/", response_model=QueryResponse, summary="Query medical documents")
async def query_documents(request: QueryRequest):
    """
    Ask a natural-language question about indexed medical documents.

    Returns:
    - **answer**: synthesized answer from LLM
    - **sources**: list of evidence with document/page/asset references
    """
    start = time.time()
    pipeline = get_retrieval_pipeline()
    synthesizer = get_synthesizer()

    # Optional: inject explicit filters from request
    result = pipeline.retrieve(request.query)

    # Override top_k for this request if specified
    if request.top_k:
        result.sources = result.sources[:request.top_k]

    # LLM synthesis
    answer = synthesizer.synthesize(request.query, result.sources)

    elapsed_ms = (time.time() - start) * 1000
    return QueryResponse(
        answer=answer,
        sources=result.sources,
        query=request.query,
        processing_time_ms=round(elapsed_ms, 1),
    )


@router.get("/search", summary="Quick semantic search (no LLM synthesis)")
async def semantic_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
):
    """
    Pure retrieval without LLM synthesis — useful for debugging or
    building custom frontends.
    """
    pipeline = get_retrieval_pipeline()
    result = pipeline.retrieve(q)
    sources = result.sources[:top_k]
    return {
        "query": q,
        "filters_extracted": result.query_filters,
        "results": [s.model_dump() for s in sources],
    }
