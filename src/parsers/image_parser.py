"""
MacPro AI — Standalone Image Parser.

Handles PNG, JPEG, TIFF, BMP files that arrive as individual assets
(e.g. scanned pages, exported figures).

Returns a minimal ParsedPage(page_number=1) so the ingestion pipeline
can treat image files uniformly alongside PDFs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.parsers.pdf_parser import ExtractedImage, ParsedPage
from src.utils.helpers import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class ImageParser:
    """Parse a standalone image file into a single ParsedPage."""

    def parse(self, img_path: str | Path) -> Optional[ParsedPage]:
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow not installed. Run: pip install pillow")

        img_path = Path(img_path)
        if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported image extension: {img_path.suffix}")
            return None

        logger.info(f"Parsing image: {img_path.name}")
        try:
            img = Image.open(str(img_path))
            width, height = img.size
            # Read raw bytes for embedding
            with open(str(img_path), "rb") as f:
                img_bytes = f.read()

            ext = img_path.suffix.lstrip(".").lower()
            extracted_img = ExtractedImage(
                index=0,
                bbox=[0.0, 0.0, float(width), float(height)],
                data=img_bytes,
                ext=ext,
                xref=0,
            )
            return ParsedPage(
                page_number=1,
                raw_text="",   # OCR will fill this in
                width=float(width),
                height=float(height),
                images=[extracted_img],
            )
        except Exception as e:
            logger.error(f"Failed to parse image {img_path}: {e}")
            return None
