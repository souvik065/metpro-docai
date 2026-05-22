"""
MacPro AI — DICOM Parser.

Uses pydicom to read DICOM files and extract:
- Patient / study / series / instance metadata
- Pixel data → PNG for embedding and display

For advanced DICOM processing (windowing, segmentation):
SimpleITK or MONAI can be plugged in here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.helpers import get_logger, ensure_dir

logger = get_logger(__name__)


@dataclass
class DICOMResult:
    patient_id: str
    patient_name: str
    study_date: str
    study_description: str
    modality: str           # CT, MR, CR, DX (X-ray), US, etc.
    series_description: str
    instance_number: str
    rows: int
    columns: int
    pixel_image_path: Optional[str]   # path to saved PNG
    extra_meta: dict                   # any additional DICOM tags


class DICOMParser:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def parse(self, dcm_path: str | Path) -> Optional[DICOMResult]:
        try:
            import pydicom
        except ImportError:
            raise RuntimeError("pydicom not installed. Run: pip install pydicom")

        dcm_path = Path(dcm_path)
        logger.info(f"Parsing DICOM: {dcm_path.name}")

        try:
            ds = pydicom.dcmread(str(dcm_path))
        except Exception as e:
            logger.error(f"Failed to read DICOM {dcm_path}: {e}")
            return None

        def tag(attr: str, default: str = "") -> str:
            try:
                return str(getattr(ds, attr, default) or default)
            except Exception:
                return default

        modality = tag("Modality", "UNKNOWN")
        patient_id = tag("PatientID", "UNKNOWN")
        patient_name = tag("PatientName", "UNKNOWN")
        study_date = tag("StudyDate", "")
        study_desc = tag("StudyDescription", "")
        series_desc = tag("SeriesDescription", "")
        instance_num = tag("InstanceNumber", "0")
        rows = int(getattr(ds, "Rows", 0))
        columns = int(getattr(ds, "Columns", 0))

        # ── Extract pixel data → PNG ──────────────────────────────────────
        pixel_path: Optional[str] = None
        try:
            pixel_path = self._save_pixel_png(ds, dcm_path.stem)
        except Exception as e:
            logger.warning(f"  Could not extract pixel data from {dcm_path.name}: {e}")

        # ── Extra metadata ────────────────────────────────────────────────
        extra = {}
        for attr in ["Manufacturer", "InstitutionName", "BodyPartExamined",
                     "AcquisitionDate", "ContentDate"]:
            v = tag(attr)
            if v:
                extra[attr] = v

        return DICOMResult(
            patient_id=patient_id,
            patient_name=patient_name,
            study_date=study_date,
            study_description=study_desc,
            modality=modality,
            series_description=series_desc,
            instance_number=instance_num,
            rows=rows,
            columns=columns,
            pixel_image_path=pixel_path,
            extra_meta=extra,
        )

    def _save_pixel_png(self, ds, stem: str) -> str:
        """Convert DICOM pixel array to a normalized PNG and save it."""
        import numpy as np
        from PIL import Image

        pixel_array = ds.pixel_array  # may raise if no pixel data

        # Normalize to 8-bit
        arr = pixel_array.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = arr.astype(np.uint8)

        # Handle multi-frame or RGB
        if arr.ndim == 3 and arr.shape[0] > 3:
            # Multi-frame: save first frame
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pass  # RGB/RGBA, PIL handles it

        img = Image.fromarray(arr)
        if img.mode not in ("RGB", "L"):
            img = img.convert("L")

        out_path = self.output_dir / f"{stem}.png"
        img.save(str(out_path))
        logger.debug(f"  DICOM pixel saved → {out_path}")
        return str(out_path)
