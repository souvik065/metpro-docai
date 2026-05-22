"""
MacPro AI — OCR Engine.

Primary: EasyOCR
- Better accuracy on medical documents, handwriting, low-resolution scans
- GPU-capable (falls back to CPU automatically)
- No system binary required

Fallback: pytesseract (Tesseract)
- Requires system binary `tesseract`
- Faster on CPU for clean documents

Both return the same OCRResult type so the rest of the pipeline is OCR-engine-agnostic.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class OCRRegion:
    text: str
    confidence: float
    bbox: list[float]   # [x0, y0, x1, y1]


@dataclass
class OCRResult:
    full_text: str
    regions: list[OCRRegion] = field(default_factory=list)
    engine_used: str = "none"


class OCREngine:
    """
    Unified OCR interface.
    Instantiate once and call .run(image_bytes) repeatedly.
    """

    def __init__(self):
        self._easyocr_reader = None
        self._engine = settings.ocr_engine
        self._languages = settings.ocr_languages

    def run(self, image_bytes: bytes) -> OCRResult:
        """Run OCR on raw image bytes. Returns OCRResult."""
        if self._engine == "easyocr":
            return self._run_easyocr(image_bytes)
        elif self._engine == "tesseract":
            return self._run_tesseract(image_bytes)
        else:
            logger.warning(f"Unknown OCR engine '{self._engine}'; skipping OCR")
            return OCRResult(full_text="", engine_used="none")

    def run_file(self, image_path: str | Path) -> OCRResult:
        with open(str(image_path), "rb") as f:
            return self.run(f.read())

    # ── EasyOCR ───────────────────────────────────────────────────────────

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader (first call may download model)…")
                self._easyocr_reader = easyocr.Reader(
                    self._languages,
                    gpu=False,   # set True if GPU available
                    verbose=False,
                )
                logger.info("EasyOCR ready.")
            except ImportError:
                raise RuntimeError("EasyOCR not installed. Run: pip install easyocr")
        return self._easyocr_reader

    def _run_easyocr(self, image_bytes: bytes) -> OCRResult:
        try:
            import numpy as np
            from PIL import Image

            reader = self._get_easyocr_reader()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(img)
            results = reader.readtext(img_array, detail=1)

            regions: list[OCRRegion] = []
            texts: list[str] = []
            for (bbox_pts, text, confidence) in results:
                if not text.strip():
                    continue
                # bbox_pts: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                xs = [p[0] for p in bbox_pts]
                ys = [p[1] for p in bbox_pts]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                regions.append(OCRRegion(text=text, confidence=float(confidence), bbox=bbox))
                texts.append(text)

            return OCRResult(
                full_text=" ".join(texts),
                regions=regions,
                engine_used="easyocr",
            )
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}; trying tesseract fallback")
            return self._run_tesseract(image_bytes)

    # ── Tesseract ─────────────────────────────────────────────────────────

    def _run_tesseract(self, image_bytes: bytes) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            lang_str = "+".join(self._languages)
            data = pytesseract.image_to_data(
                img, lang=lang_str, output_type=pytesseract.Output.DICT
            )
            regions: list[OCRRegion] = []
            texts: list[str] = []
            for i, word in enumerate(data["text"]):
                if not word.strip():
                    continue
                conf = float(data["conf"][i])
                if conf < 0:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                regions.append(
                    OCRRegion(text=word, confidence=conf / 100.0, bbox=[x, y, x+w, y+h])
                )
                texts.append(word)

            full_text = pytesseract.image_to_string(img, lang=lang_str)
            return OCRResult(full_text=full_text, regions=regions, engine_used="tesseract")

        except ImportError:
            logger.warning("pytesseract not installed; returning empty OCR result")
            return OCRResult(full_text="", engine_used="none")
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRResult(full_text="", engine_used="none")
