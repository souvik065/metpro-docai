"""
MacPro AI — Retrieval Pipeline.

Steps:
1. Parse query for metadata filters (patient ID, modality, date)
2. Embed query with sentence-transformers → search text vectors
3. Embed query with CLIP text encoder → search image vectors
4. Merge, deduplicate, and re-rank results
5. Return structured SourceReference objects

The LLM synthesis step is in the query handler (api/routes/query.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from src.embeddings.models import get_image_embedder, get_text_embedder
from src.indexing.vector_store import get_vector_store
from src.models.schema import AssetType, SourceReference
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Simple date pattern for filter extraction: YYYY-MM-DD or YYYYMMDD
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{8})\b")
# Modality keywords
_MODALITY_MAP = {
    "x-ray": "CR", "xray": "CR", "chest x-ray": "CR",
    "ct": "CT", "mri": "MR", "mr": "MR", "ultrasound": "US",
    "mammogram": "MG", "pet": "PT",
}


@dataclass
class RetrievalResult:
    sources: list[SourceReference]
    query_filters: dict


class RetrievalPipeline:

    def __init__(self):
        self.text_embedder = get_text_embedder()
        self.image_embedder = get_image_embedder()
        self.vector_store = get_vector_store()

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Main entry point: given a natural-language query,
        return ranked SourceReferences from all modalities.
        """
        logger.info(f"Retrieving for query: {query!r}")

        # ── Extract metadata filters from query ───────────────────────────
        filters = self._extract_filters(query)
        logger.debug(f"  Extracted filters: {filters}")

        # ── Text search ───────────────────────────────────────────────────
        text_vec = self.text_embedder.embed(query)
        text_hits = self.vector_store.search_text(
            query_vector=text_vec,
            top_k=settings.top_k_text,
            filters=filters if filters else None,
        )

        # ── Image search (CLIP text → image space) ────────────────────────
        image_hits: list[dict] = []
        try:
            clip_vec = self.image_embedder.embed_text_for_image_search(query)
            if clip_vec:
                image_hits = self.vector_store.search_image(
                    query_vector=clip_vec,
                    top_k=settings.top_k_image,
                    filters=filters if filters else None,
                )
        except Exception as e:
            logger.warning(f"  Image search failed: {e}")

        # ── Merge and deduplicate ─────────────────────────────────────────
        sources = self._merge_results(text_hits, image_hits)
        logger.info(f"  → {len(sources)} sources retrieved")

        return RetrievalResult(sources=sources, query_filters=filters)

    # ── Private helpers ───────────────────────────────────────────────────

    def _extract_filters(self, query: str) -> dict:
        """
        Heuristic filter extraction.
        Replace with an NER model or function-calling LLM for production.
        """
        filters: dict = {}
        q_lower = query.lower()

        # Modality
        for keyword, code in _MODALITY_MAP.items():
            if keyword in q_lower:
                filters["modality"] = code
                break

        # Date
        date_match = _DATE_RE.search(query)
        if date_match:
            filters["study_date"] = date_match.group(0).replace("-", "")

        # Patient ID heuristic: "patient P12345" or "patient id 12345"
        patient_match = re.search(r"patient(?:\s+id)?\s+([A-Za-z0-9\-]+)", q_lower)
        if patient_match:
            filters["patient_id"] = patient_match.group(1)

        return filters

    def _merge_results(
        self, text_hits: list[dict], image_hits: list[dict]
    ) -> list[SourceReference]:
        seen_asset_ids: set[str] = set()
        sources: list[SourceReference] = []

        def add_hit(hit: dict, score: float):
            p = hit.get("payload", {})
            asset_id = p.get("asset_id", str(hit.get("id", "")))
            if asset_id in seen_asset_ids:
                return
            seen_asset_ids.add(asset_id)
            sources.append(
                SourceReference(
                    type=AssetType(p.get("asset_type", "text")),
                    document_id=p.get("document_id", ""),
                    filename=p.get("filename", ""),
                    page=p.get("page_number"),
                    asset_id=asset_id,
                    snippet=p.get("snippet") or p.get("ocr_text") or "",
                    path_or_uri=p.get("path_or_uri"),
                    score=round(score, 4),
                )
            )

        # Text results weighted by alpha
        for hit in text_hits:
            add_hit(hit, hit["score"] * settings.hybrid_alpha)

        # Image results weighted by (1 - alpha)
        for hit in image_hits:
            add_hit(hit, hit["score"] * (1.0 - settings.hybrid_alpha))

        # Sort by score descending
        sources.sort(key=lambda s: s.score or 0, reverse=True)
        return sources
