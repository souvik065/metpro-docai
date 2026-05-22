"""Unit tests for PDF parser — uses mocks so fitz is not required."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Minimal fitz stub so import doesn't fail without PyMuPDF ──────────────────
def _make_fitz_stub():
    fitz = types.ModuleType("fitz")

    class FakeRect:
        width = 595.0
        height = 842.0

    class FakePage:
        def get_text(self, mode):
            return "Sample medical report text. Patient: John Doe. Diagnosis: Pneumonia."

        def get_images(self, full=True):
            return []  # no embedded images for this stub

        def get_links(self):
            return [{"uri": "https://example.com/study", "from": [10, 20, 80, 30]}]

        @property
        def rect(self):
            return FakeRect()

    class FakeDoc:
        def __init__(self, path):
            self._pages = [FakePage(), FakePage()]

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, idx):
            return self._pages[idx]

        def extract_image(self, xref):
            return {"image": b"\x89PNG\r\n", "ext": "png"}

        def close(self):
            pass

    fitz.open = lambda path: FakeDoc(path)
    return fitz


sys.modules.setdefault("fitz", _make_fitz_stub())

from src.parsers.pdf_parser import PDFParser, ParsedPage


class TestPDFParser:
    def setup_method(self):
        self.parser = PDFParser(output_dir="/tmp/macpro_test_pdf")

    def test_parse_returns_list_of_parsed_pages(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")
        pages = self.parser.parse(str(fake_pdf))
        assert isinstance(pages, list)
        assert len(pages) == 2

    def test_page_numbers_are_one_based(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF fake")
        pages = self.parser.parse(str(fake_pdf))
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2

    def test_raw_text_extracted(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF fake")
        pages = self.parser.parse(str(fake_pdf))
        assert "Pneumonia" in pages[0].raw_text

    def test_url_extracted(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF fake")
        pages = self.parser.parse(str(fake_pdf))
        assert any(u.url == "https://example.com/study" for p in pages for u in p.urls)

    def test_page_dimensions(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF fake")
        pages = self.parser.parse(str(fake_pdf))
        assert pages[0].width == 595.0
        assert pages[0].height == 842.0

    def test_missing_file_returns_empty(self, tmp_path):
        pages = self.parser.parse(str(tmp_path / "nonexistent.pdf"))
        assert pages == []
