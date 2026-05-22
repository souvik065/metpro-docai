"""Unit tests for OCR engine — mocks EasyOCR so no GPU/model download needed."""
import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ocr.engine import OCREngine, OCRResult, OCRRegion


def _make_fake_png() -> bytes:
    """Return a valid 1×1 white PNG as bytes."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (100, 50), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


class TestOCREngineEasyOCR:
    """Tests using a mocked EasyOCR reader."""

    def _make_engine(self):
        engine = OCREngine()
        engine._engine = "easyocr"
        # Pre-inject a mock reader so _get_easyocr_reader() returns it
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 10], [80, 10], [80, 30], [10, 30]], "Diagnosis: Pneumonia", 0.97),
            ([[10, 40], [90, 40], [90, 60], [10, 60]], "WBC: 12.3", 0.91),
        ]
        engine._easyocr_reader = mock_reader
        return engine

    def test_run_returns_ocr_result(self):
        engine = self._make_engine()
        result = engine.run(_make_fake_png())
        assert isinstance(result, OCRResult)

    def test_full_text_contains_detected_words(self):
        engine = self._make_engine()
        result = engine.run(_make_fake_png())
        assert "Pneumonia" in result.full_text
        assert "WBC" in result.full_text

    def test_regions_count_matches_detections(self):
        engine = self._make_engine()
        result = engine.run(_make_fake_png())
        assert len(result.regions) == 2

    def test_region_confidence(self):
        engine = self._make_engine()
        result = engine.run(_make_fake_png())
        assert result.regions[0].confidence == pytest.approx(0.97)

    def test_region_bbox_format(self):
        engine = self._make_engine()
        result = engine.run(_make_fake_png())
        bbox = result.regions[0].bbox
        assert len(bbox) == 4           # [x0, y0, x1, y1]
        assert bbox[2] > bbox[0]        # x1 > x0
        assert bbox[3] > bbox[1]        # y1 > y0

    def test_engine_used_label(self):
        engine = self._make_engine()
        result = engine.run(_make_fake_png())
        assert result.engine_used == "easyocr"

    def test_empty_detections(self):
        engine = self._make_engine()
        engine._easyocr_reader.readtext.return_value = []
        result = engine.run(_make_fake_png())
        assert result.full_text == ""
        assert result.regions == []


class TestOCREngineUnknown:
    def test_unknown_engine_returns_empty(self):
        engine = OCREngine()
        engine._engine = "nonexistent_ocr"
        result = engine.run(_make_fake_png())
        assert result.full_text == ""
        assert result.engine_used == "none"
