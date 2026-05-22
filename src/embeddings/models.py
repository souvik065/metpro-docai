"""
MacPro AI — Embedding Models.

TextEmbedder:   sentence-transformers (all-MiniLM-L6-v2 by default)
                Fast, 384-dim, runs on CPU without API keys.

ImageEmbedder:  CLIP (openai/clip-vit-base-patch32)
                512-dim visual embeddings for X-rays, figures, scanned pages.
                CLIP also provides text→image retrieval: encode a query
                with CLIP's text encoder and find visually similar images.

Both are lazy-loaded on first use.
Swap models by changing config/settings.py without touching this file.
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np

from config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    """Encodes text strings into dense vectors."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading text embedding model: {settings.text_embed_model}")
                self._model = SentenceTransformer(settings.text_embed_model)
                logger.info("Text embedder ready.")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class ImageEmbedder:
    """
    Encodes images AND text using CLIP.
    Both outputs live in the same 512-dim space, enabling
    text-query → image retrieval without fine-tuning.
    """

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                logger.info(f"Loading CLIP model: {settings.clip_model}")
                self._model = CLIPModel.from_pretrained(settings.clip_model)
                self._processor = CLIPProcessor.from_pretrained(settings.clip_model)
                self._model.eval()
                logger.info("CLIP image embedder ready.")
            except ImportError:
                raise RuntimeError(
                    "transformers not installed. "
                    "Run: pip install transformers"
                )
        return self._model, self._processor

    def embed_image_bytes(self, image_bytes: bytes) -> Optional[list[float]]:
        try:
            from PIL import Image
            import torch
            model, processor = self._load()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                # Newer transformers versions may return a ModelOutput object
                # instead of a raw tensor — handle both cases.
                if not isinstance(outputs, torch.Tensor):
                    feats = getattr(outputs, "image_embeds", None)
                    if feats is None:
                        feats = getattr(outputs, "pooler_output", outputs)
                else:
                    feats = outputs
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats[0].tolist()
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return None


    def embed_image_path(self, image_path: str) -> Optional[list[float]]:
        with open(image_path, "rb") as f:
            return self.embed_image_bytes(f.read())

    def embed_text_for_image_search(self, text: str) -> Optional[list[float]]:
        """
        Encode a text query with CLIP's text encoder.
        The result lives in the same space as image embeddings,
        enabling cross-modal retrieval.
        """
        try:
            import torch
            model, processor = self._load()
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                # CLIPModel.get_text_features returns the projected text embeddings
                outputs = model.get_text_features(**inputs)
                # Some versions might return a wrapper, though usually it's a tensor
                if not isinstance(outputs, torch.Tensor):
                    # Fallback if it's a model output object
                    feats = getattr(outputs, "text_embeds", None)
                    if feats is None:
                        feats = getattr(outputs, "pooler_output", outputs)
                else:
                    feats = outputs
                
                # Normalize
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats[0].tolist()
        except Exception as e:
            logger.error(f"CLIP text embedding failed: {e}")
            return None


# ── Singleton instances (imported by other modules) ───────────────────────────
_text_embedder: Optional[TextEmbedder] = None
_image_embedder: Optional[ImageEmbedder] = None


def get_text_embedder() -> TextEmbedder:
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = TextEmbedder()
    return _text_embedder


def get_image_embedder() -> ImageEmbedder:
    global _image_embedder
    if _image_embedder is None:
        _image_embedder = ImageEmbedder()
    return _image_embedder
