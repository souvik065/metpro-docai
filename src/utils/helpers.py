"""MacPro AI — Logging and shared utilities."""
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger


def new_id() -> str:
    return str(uuid.uuid4())


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_file_type(path: str | Path) -> str:
    """Return a broad file type string based on extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".dcm" or suffix == "":
        # DICOM files sometimes have no extension
        return "dicom"
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        return "image"
    return "unknown"


def safe_str(value: Any, max_len: int = 2000) -> str:
    """Safely convert any value to a truncated string."""
    try:
        s = str(value)
        return s[:max_len] if len(s) > max_len else s
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Naive word-boundary chunker.
    For production, swap with a tokenizer-aware splitter
    (e.g. LangChain's RecursiveCharacterTextSplitter or LlamaIndex's SentenceSplitter).
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
