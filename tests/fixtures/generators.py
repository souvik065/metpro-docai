"""
MacPro AI — Test fixtures.

Generates synthetic medical document fixtures so tests can run
without real patient data.
"""
from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path

# ── Minimal valid PDF builder (no third-party deps) ──────────────────────────

_PDF_TEMPLATE = """%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length {content_length} >>
stream
{content}
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
0
%%EOF
"""


def make_fake_pdf(text: str = "Sample medical report.") -> bytes:
    """Return a syntactically minimal PDF with the given text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    filled = _PDF_TEMPLATE.format(content=stream, content_length=len(stream))
    return filled.encode()


def make_fake_png(width: int = 64, height: int = 64, text_hint: str = "") -> bytes:
    """Return a minimal valid PNG file (white image)."""
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # Raw image data: white RGB pixels
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\xff\xff\xff" * width
    compressed = zlib.compress(raw)
    idat = make_chunk(b"IDAT", compressed)
    iend = make_chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def make_fake_dicom(
    patient_id: str = "TEST001",
    modality: str = "CR",
    study_date: str = "20240315",
    output_path: str | Path | None = None,
) -> bytes | None:
    """
    Create a minimal DICOM file using pydicom (optional).
    Returns bytes if pydicom available, None otherwise.
    Saves to output_path if given.
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
        import numpy as np
        from datetime import datetime

        ds = FileDataset(None, {}, file_meta=pydicom.Dataset(), preamble=b"\x00" * 128)
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds.PatientID = patient_id
        ds.PatientName = "Test^Patient"
        ds.StudyDate = study_date
        ds.Modality = modality
        ds.StudyDescription = "Test Study"
        ds.SeriesDescription = "Test Series"
        ds.InstanceNumber = "1"
        ds.Rows = 64
        ds.Columns = 64
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.zeros((64, 64), dtype=np.uint16).tobytes()

        if output_path:
            ds.save_as(str(output_path), write_like_original=False)
            return None
        else:
            buf = io.BytesIO()
            ds.save_as(buf, write_like_original=False)
            return buf.getvalue()
    except ImportError:
        return None


def write_fixtures(base_dir: str | Path) -> dict[str, Path]:
    """Write all fixture files to base_dir. Returns {name: path}."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # PDF
    pdf_path = base_dir / "sample_report.pdf"
    pdf_path.write_bytes(make_fake_pdf(
        "Patient: Jane Doe. Diagnosis: Pulmonary infection. "
        "WBC: 14.2 k/uL. X-ray shows infiltrates in right lower lobe."
    ))
    written["pdf"] = pdf_path

    # PNG (simulated X-ray / scan)
    png_path = base_dir / "chest_xray.png"
    png_path.write_bytes(make_fake_png(64, 64))
    written["png"] = png_path

    # DICOM (if pydicom available)
    dcm_path = base_dir / "chest.dcm"
    result = make_fake_dicom(
        patient_id="PAT001", modality="CR",
        study_date="20240315", output_path=dcm_path
    )
    if dcm_path.exists():
        written["dicom"] = dcm_path

    # Sample metadata JSON (for testing schema loading)
    meta_path = base_dir / "sample_meta.json"
    meta_path.write_text(json.dumps({
        "patient_id": "PAT001",
        "study_date": "2024-03-15",
        "modality": "CR",
        "findings": "Right lower lobe infiltrate",
    }, indent=2))
    written["meta"] = meta_path

    return written
