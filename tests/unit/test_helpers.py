"""Unit tests for src/utils/helpers.py"""
import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import chunk_text, detect_file_type, safe_str


class TestChunkText:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        text = "Hello world this is a test."
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_creates_multiple_chunks(self):
        # 600 words should produce 2 chunks with chunk_size=512 and overlap=64
        words = ["word"] * 600
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) >= 2

    def test_overlap_content(self):
        words = [f"w{i}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # The first word of chunk[1] should appear near the end of chunk[0]
        first_word_of_chunk1 = chunks[1].split()[0]
        assert first_word_of_chunk1 in chunks[0]

    def test_no_empty_chunks(self):
        text = "  " * 100  # lots of whitespace
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        assert all(c.strip() for c in chunks)

    def test_whitespace_only(self):
        assert chunk_text("   ") == []


class TestDetectFileType:
    def test_pdf(self):
        assert detect_file_type("report.pdf") == "pdf"

    def test_image_extensions(self):
        for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"]:
            assert detect_file_type(f"scan{ext}") == "image"

    def test_dicom(self):
        assert detect_file_type("chest.dcm") == "dicom"

    def test_unknown(self):
        assert detect_file_type("file.xyz") == "unknown"

    def test_case_insensitive(self):
        assert detect_file_type("report.PDF") == "pdf"
        assert detect_file_type("scan.PNG") == "image"


class TestSafeStr:
    def test_normal_string(self):
        assert safe_str("hello") == "hello"

    def test_truncation(self):
        long_str = "a" * 3000
        result = safe_str(long_str, max_len=2000)
        assert len(result) == 2000

    def test_non_string_input(self):
        assert safe_str(42) == "42"
        assert safe_str(None) == "None"
        assert safe_str([1, 2, 3]) == "[1, 2, 3]"
