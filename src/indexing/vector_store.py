"""
MacPro AI — Qdrant Vector Store.

Uses Qdrant's LOCAL (file-based) mode for zero-setup development.
Set QDRANT_HOST in .env to switch to a remote Qdrant server.

Named vectors:
  "text"  — sentence-transformers embeddings (384-dim)
  "image" — CLIP image/text embeddings (512-dim)

This lets us store both modalities in a single collection and
run separate ANN searches per modality, then merge results.
"""
from __future__ import annotations

from typing import Any, Optional

from qdrant_client.models import NamedVector

from config.settings import settings
from src.utils.helpers import get_logger, ensure_dir

logger = get_logger(__name__)

# Named vector keys
TEXT_VECTOR = "text"
IMAGE_VECTOR = "image"


class VectorStore:
    """
    Thin wrapper around Qdrant client.
    All pipeline code imports this class — swap Qdrant for Weaviate/Pinecone
    here without touching ingestion or retrieval code.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams, VectorsConfig

                if settings.qdrant_host:
                    logger.info(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
                    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
                else:
                    path = str(ensure_dir(settings.qdrant_path))
                    logger.info(f"Using local Qdrant at {path}")
                    client = QdrantClient(path=path)

                # Create collection if missing
                existing = [c.name for c in client.get_collections().collections]
                if settings.qdrant_collection not in existing:
                    logger.info(f"Creating Qdrant collection '{settings.qdrant_collection}'")
                    client.create_collection(
                        collection_name=settings.qdrant_collection,
                        vectors_config={
                            TEXT_VECTOR: VectorParams(
                                size=settings.text_embed_dim,
                                distance=Distance.COSINE,
                            ),
                            IMAGE_VECTOR: VectorParams(
                                size=settings.image_embed_dim,
                                distance=Distance.COSINE,
                            ),
                        },
                    )
                self._client = client
            except ImportError:
                raise RuntimeError("qdrant-client not installed. Run: pip install qdrant-client")
        return self._client

    # ── Upsert ────────────────────────────────────────────────────────────

    def upsert_text(
        self,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Store a text embedding point."""
        from qdrant_client.models import PointStruct
        client = self._get_client()
        # Named vector dict — only the "text" vector for text assets
        # Provide a zero image vector as placeholder so the schema is satisfied
        vectors = {
            TEXT_VECTOR: vector,
            IMAGE_VECTOR: [0.0] * settings.image_embed_dim,
        }
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[PointStruct(id=self._to_uint64(point_id), vector=vectors, payload=payload)],
        )

    def upsert_image(
        self,
        point_id: str,
        image_vector: list[float],
        payload: dict[str, Any],
        text_vector: Optional[list[float]] = None,
    ) -> None:
        """Store an image embedding point (optionally with a text vector too)."""
        from qdrant_client.models import PointStruct
        client = self._get_client()
        vectors = {
            TEXT_VECTOR: text_vector if text_vector else [0.0] * settings.text_embed_dim,
            IMAGE_VECTOR: image_vector,
        }
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[PointStruct(id=self._to_uint64(point_id), vector=vectors, payload=payload)],
        )

    # ── Search ────────────────────────────────────────────────────────────

    def search_text(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """ANN search on the 'text' named vector."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = self._get_client()
        qdrant_filter = self._build_filter(filters)
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            using=TEXT_VECTOR,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )


        return [
        {
            "score": p.score,
            "payload": p.payload,
            "id": p.id
        }
        for p in results.points
    ]

    def search_image(
        self,
        query_vector: list[float],
        top_k: int = 3,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """ANN search on the 'image' named vector."""
        client = self._get_client()
        qdrant_filter = self._build_filter(filters)
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            using=IMAGE_VECTOR,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
        {
            "score": p.score,
            "payload": p.payload,
            "id": p.id
        }
        for p in results.points
    ]

    def count(self) -> int:
        client = self._get_client()
        return client.count(collection_name=settings.qdrant_collection).count

    # ── Helpers ───────────────────────────────────────────────────────────

    def _to_uint64(self, uid: str) -> int:
        """
        Qdrant requires integer or UUID point IDs.
        Convert string UUID to int via hashing.
        """
        import hashlib
        h = hashlib.sha256(uid.encode()).hexdigest()
        return int(h[:16], 16)

    def _build_filter(self, filters: Optional[dict]):
        if not filters:
            return None
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None


# ── Singleton ─────────────────────────────────────────────────────────────────
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
