"""
MacPro AI — PDF Parser.

Primary engine: PyMuPDF (fitz) — fastest, handles encrypted/scanned PDFs,
extracts text, images, and hyperlinks per page.

Table extraction: pdfplumber — more accurate column detection than PyMuPDF
for structured tables; falls back gracefully if not installed.

Design: Each parser returns a list of ParsedPage objects which the ingestion
pipeline stitches into Document + Page + Asset DB records.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.helpers import get_logger, ensure_dir

logger = get_logger(__name__)


@dataclass
class ExtractedImage:
    index: int          # image index on the page
    bbox: list[float]   # [x0, y0, x1, y1]
    data: bytes         # raw image bytes
    ext: str            # "png" | "jpeg" etc.
    xref: int           # PyMuPDF internal reference


@dataclass
class ExtractedTable:
    index: int
    bbox: list[float]
    rows: list[list[str]]   # 2-D list of cell strings
    headers: list[str]


@dataclass
class ExtractedURL:
    url: str
    bbox: list[float]
    page_text_context: str  # ±100 chars around the link


@dataclass
class ParsedPage:
    page_number: int        # 1-based
    raw_text: str
    width: float
    height: float
    images: list[ExtractedImage] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    urls: list[ExtractedURL] = field(default_factory=list)


class PDFParser:
    """
    Parses a PDF into pages, extracting:
    - text (with layout preservation)
    - embedded images
    - hyperlinks/URLs
    - tables (via pdfplumber)
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def parse(self, pdf_path: str | Path) -> list[ParsedPage]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF not installed. Run: pip install pymupdf"
            )

        pdf_path = Path(pdf_path)
        logger.info(f"Parsing PDF: {pdf_path.name}")
        pages: list[ParsedPage] = []

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to open {pdf_path}: {e}")
            return pages

        # Try pdfplumber for tables (optional dependency)
        plumber_pages = self._load_plumber_pages(pdf_path)

        for page_idx in range(len(doc)):
            fitz_page = doc[page_idx]
            page_num = page_idx + 1

            # ── Text ──────────────────────────────────────────────────────
            raw_text = fitz_page.get_text("text")

            # ── Images ────────────────────────────────────────────────────
            images = self._extract_images(fitz_page, doc, page_num)

            # ── URLs / hyperlinks ─────────────────────────────────────────
            urls = self._extract_urls(fitz_page, raw_text)

            # ── Tables ────────────────────────────────────────────────────
            tables: list[ExtractedTable] = []
            if plumber_pages and page_idx < len(plumber_pages):
                tables = self._extract_tables_plumber(plumber_pages[page_idx], page_idx)

            rect = fitz_page.rect
            pages.append(
                ParsedPage(
                    page_number=page_num,
                    raw_text=raw_text,
                    width=rect.width,
                    height=rect.height,
                    images=images,
                    tables=tables,
                    urls=urls,
                )
            )

        doc.close()
        logger.info(f"  → {len(pages)} pages parsed from {pdf_path.name}")
        return pages

    # ── Private helpers ───────────────────────────────────────────────────

    def _extract_images(self, page, doc, page_num: int) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        try:
            img_list = page.get_images(full=True)
            logger.info(f"  Page {page_num}: found {len(img_list)} images")
            for img_idx, img_info in enumerate(img_list):
                logger.debug(f"    Image {img_idx}: xref={img_info[0]} size={img_info[2]}x{img_info[3]} bpc={img_info[4]} colorspace={img_info[5]}")
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    ext = base_image["ext"]
                    # Get bounding box of image on page
                    img_rects = page.get_image_rects(xref)
                    bbox = list(img_rects[0]) if img_rects else [0, 0, 0, 0]
                    images.append(
                        ExtractedImage(
                            index=img_idx,
                            bbox=bbox,
                            data=img_bytes,
                            ext=ext,
                            xref=xref,
                        )
                    )
                    print(f"      ✓ Extracted image xref={xref} as {ext} ({len(img_bytes)} bytes)")

                except Exception as e:
                    logger.debug(f"    Image xref={xref} page={page_num}: {e}")
        except Exception as e:
            logger.warning(f"  Image extraction failed on page {page_num}: {e}")
        return images

    def _extract_urls(self, page, raw_text: str) -> list[ExtractedURL]:
        urls: list[ExtractedURL] = []
        try:
            links = page.get_links()
            for link in links:
                uri = link.get("uri", "")
                if not uri:
                    continue
                rect = link.get("from")
                bbox = list(rect) if rect else [0, 0, 0, 0]
                # Grab surrounding text as context
                ctx_start = max(0, raw_text.find(uri) - 100)
                ctx_end = min(len(raw_text), raw_text.find(uri) + len(uri) + 100)
                context = raw_text[ctx_start:ctx_end] if uri in raw_text else raw_text[:200]
                urls.append(ExtractedURL(url=uri, bbox=bbox, page_text_context=context))
        except Exception as e:
            logger.debug(f"  URL extraction error: {e}")
        return urls

    def _load_plumber_pages(self, pdf_path: Path):
        try:
            import pdfplumber
            pdf = pdfplumber.open(str(pdf_path))
            return pdf.pages
        except ImportError:
            logger.debug("pdfplumber not installed; skipping table extraction from PDF layout")
            return None
        except Exception as e:
            logger.warning(f"pdfplumber failed on {pdf_path}: {e}")
            return None

    def _extract_tables_plumber(self, plumber_page, page_idx: int) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []
        try:
            raw_tables = plumber_page.extract_tables()
            for t_idx, table_rows in enumerate(raw_tables):
                if not table_rows:
                    continue
                headers = [str(c or "") for c in table_rows[0]]
                rows = [[str(c or "") for c in row] for row in table_rows[1:]]
                # pdfplumber doesn't give direct bbox easily; use page bbox as fallback
                bbox = list(plumber_page.bbox)
                tables.append(
                    ExtractedTable(
                        index=t_idx,
                        bbox=bbox,
                        rows=rows,
                        headers=headers,
                    )
                )
        except Exception as e:
            logger.debug(f"  Table extraction page {page_idx+1}: {e}")
        return tables
