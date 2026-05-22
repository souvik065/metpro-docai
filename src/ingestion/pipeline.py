"""
MacPro AI — Ingestion Pipeline.

Orchestrates:
1. File detection (PDF / Image / DICOM)
2. Parsing → structured pages
3. OCR on images (embedded or standalone)
4. Saving extracted assets to disk
5. Embedding text chunks and images
6. Upserting into Qdrant (vector DB) and SQLite (metadata)

Design:
- Stateless: each call to ingest_file() is independent
- Idempotent on file path (re-ingestion re-processes the file)
- Async DB writes; embedding/OCR are sync (CPU-bound, run in executor if needed)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from config.settings import settings
from src.embeddings.models import get_image_embedder, get_text_embedder
from src.indexing.vector_store import get_vector_store
from src.models.schema import Asset, AssetType, Document, FileType, Page, ProcessingStatus
from src.ocr.engine import OCREngine
from src.parsers.dicom_parser import DICOMParser
from src.parsers.image_parser import ImageParser
from src.parsers.pdf_parser import ExtractedImage, ParsedPage, PDFParser
from src.utils.helpers import chunk_text, detect_file_type, ensure_dir, get_logger, new_id

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, db_session: AsyncSession):
        self.session = db_session
        self.assets_dir = ensure_dir(settings.output_dir / "assets")
        self.ocr = OCREngine()
        self.text_embedder = get_text_embedder()
        self.image_embedder = get_image_embedder()
        self.vector_store = get_vector_store()

    # ── Public API ────────────────────────────────────────────────────────

    async def ingest_file(self, file_path: str | Path) -> Optional[Document]:
        """Ingest a single file. Returns the Document record on success."""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        file_type_str = detect_file_type(file_path)
        file_type = FileType(file_type_str) if file_type_str in FileType.__members__.values() else FileType.UNKNOWN

        print(f"\nStarting ingestion for {file_path} (detected type: {file_type})")
        # Create document record
        doc = Document(
            id=new_id(),
            filename=file_path.name,
            file_type=file_type,
            file_path=str(file_path.resolve()),
            status=ProcessingStatus.PROCESSING,
        )
        print(f"\nProcessing {file_path} (type: {file_type})...")
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)

        try:
            if file_type == FileType.PDF:
                await self._ingest_pdf(doc, file_path)
            elif file_type == FileType.IMAGE:
                await self._ingest_image(doc, file_path)
            elif file_type == FileType.DICOM:
                await self._ingest_dicom(doc, file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                doc.status = ProcessingStatus.FAILED

            doc.status = ProcessingStatus.DONE

        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}", exc_info=True)
            doc.status = ProcessingStatus.FAILED

        await self.session.commit()
        logger.info(f"✓ {file_path.name} → status={doc.status}")
        return doc

    async def ingest_folder(self, folder: str | Path, recursive: bool = True) -> list[Document]:
        """Ingest all supported files in a folder."""
        folder = Path(folder)
        pattern = "**/*" if recursive else "*"
        supported = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".webp"}
        files = [p for p in folder.glob(pattern) if p.suffix.lower() in supported and p.is_file()]
        logger.info(f"Found {len(files)} files in {folder}")
        docs: list[Document] = []
        for fp in files:
            doc = await self.ingest_file(fp)
            if doc:
                docs.append(doc)
        
        return docs

    # ── PDF ingestion ─────────────────────────────────────────────────────

    async def _ingest_pdf(self, doc: Document, file_path: Path) -> None:
        output_dir = ensure_dir(self.assets_dir / doc.id)
        parser = PDFParser(output_dir)
        pages = parser.parse(file_path)
        doc.page_count = len(pages)

        for parsed_page in pages:
            page_record = Page(
                id=new_id(),
                document_id=doc.id,
                page_number=parsed_page.page_number,
                raw_text=parsed_page.raw_text,
                width=parsed_page.width,
                height=parsed_page.height,
            )
            self.session.add(page_record)
            await self.session.flush()

            # ── Text chunks ───────────────────────────────────────────────
            if parsed_page.raw_text.strip():
                await self._index_text_chunks(doc, page_record, parsed_page.raw_text)

            # ── Images ────────────────────────────────────────────────────
            for img in parsed_page.images:
                await self._index_image(doc, page_record, img, output_dir)

            # ── Tables ────────────────────────────────────────────────────
            for table in parsed_page.tables:
                await self._index_table(doc, page_record, table)

            # ── URLs ──────────────────────────────────────────────────────
            for url_obj in parsed_page.urls:
                asset = Asset(
                    id=new_id(),
                    document_id=doc.id,
                    page_id=page_record.id,
                    asset_type=AssetType.URL,
                    content=url_obj.url,
                    bbox=url_obj.bbox,
                    meta={"context": url_obj.page_text_context},
                )
                self.session.add(asset)

        await self.session.commit()

    # ── Standalone image ingestion ─────────────────────────────────────────

    async def _ingest_image(self, doc: Document, file_path: Path) -> None:
        output_dir = ensure_dir(self.assets_dir / doc.id)
        parser = ImageParser()
        parsed_page = parser.parse(file_path)
        if not parsed_page:
            return

        doc.page_count = 1
        page_record = Page(
            id=new_id(),
            document_id=doc.id,
            page_number=1,
            raw_text="",
            width=parsed_page.width,
            height=parsed_page.height,
        )
        self.session.add(page_record)
        await self.session.flush()

        if parsed_page.images:
            await self._index_image(doc, page_record, parsed_page.images[0], output_dir)

        await self.session.commit()

    # ── DICOM ingestion ───────────────────────────────────────────────────

    async def _ingest_dicom(self, doc: Document, file_path: Path) -> None:
        output_dir = ensure_dir(self.assets_dir / doc.id)
        parser = DICOMParser(output_dir)
        result = parser.parse(file_path)
        if not result:
            return

        # Enrich document with DICOM metadata
        doc.patient_id = result.patient_id
        doc.study_date = result.study_date
        doc.modality = result.modality
        doc.study_description = result.study_description
        doc.page_count = 1
        doc.extra_meta = result.extra_meta

        page_record = Page(
            id=new_id(),
            document_id=doc.id,
            page_number=1,
            raw_text=f"{result.modality} {result.study_description} {result.series_description}",
            width=float(result.columns),
            height=float(result.rows),
            image_path=result.pixel_image_path,
        )
        self.session.add(page_record)
        await self.session.flush()

        # Index the pixel image with CLIP
        if result.pixel_image_path:
            asset_id = new_id()
            vec = self.image_embedder.embed_image_path(result.pixel_image_path)
            if vec:
                payload = {
                    "asset_id": asset_id,
                    "document_id": doc.id,
                    "page_id": page_record.id,
                    "page_number": 1,
                    "asset_type": AssetType.DICOM.value,
                    "filename": doc.filename,
                    "patient_id": doc.patient_id,
                    "modality": doc.modality,
                    "study_date": doc.study_date,
                    "path_or_uri": result.pixel_image_path,
                }
                self.vector_store.upsert_image(
                    point_id=asset_id, image_vector=vec, payload=payload
                )
            asset = Asset(
                id=asset_id,
                document_id=doc.id,
                page_id=page_record.id,
                asset_type=AssetType.DICOM,
                path_or_uri=result.pixel_image_path,
                vector_id=asset_id,
                meta={
                    "modality": result.modality,
                    "patient_id": result.patient_id,
                    "study_date": result.study_date,
                    "study_description": result.study_description,
                },
            )
            self.session.add(asset)

        # Also index descriptive text
        desc_text = (
            f"DICOM image. Modality: {result.modality}. "
            f"Study: {result.study_description}. Series: {result.series_description}. "
            f"Patient: {result.patient_id}. Date: {result.study_date}."
        )
        await self._index_text_chunks(doc, page_record, desc_text)

        await self.session.commit()

    # ── Sub-routines ──────────────────────────────────────────────────────

    async def _index_text_chunks(
        self, doc: Document, page: Page, text: str
    ) -> None:
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return
        vectors = self.text_embedder.embed_batch(chunks)
        for chunk, vec in zip(chunks, vectors):
            asset_id = new_id()
            payload = {
                "asset_id": asset_id,
                "document_id": doc.id,
                "page_id": page.id,
                "page_number": page.page_number,
                "asset_type": AssetType.TEXT.value,
                "filename": doc.filename,
                "patient_id": doc.patient_id,
                "modality": doc.modality,
                "snippet": chunk[:300],
            }
            self.vector_store.upsert_text(point_id=asset_id, vector=vec, payload=payload)
            asset = Asset(
                id=asset_id,
                document_id=doc.id,
                page_id=page.id,
                asset_type=AssetType.TEXT,
                content=chunk,
                vector_id=asset_id,
            )
            self.session.add(asset)

    async def _index_image(
        self,
        doc: Document,
        page: Page,
        extracted_img: ExtractedImage,
        output_dir: Path,
    ) -> None:
        # Save image to disk
        img_filename = f"p{page.page_number}_img{extracted_img.index}.{extracted_img.ext}"
        img_save_path = output_dir / img_filename
        with open(str(img_save_path), "wb") as f:
            f.write(extracted_img.data)

        asset_id = new_id()

        # Run OCR on image
        ocr_result = self.ocr.run(extracted_img.data)
        ocr_text = ocr_result.full_text.strip()

        # Image embedding (CLIP)
        img_vec = self.image_embedder.embed_image_bytes(extracted_img.data)

        # Text embedding of OCR text (if any)
        text_vec = None
        if ocr_text:
            text_vec = self.text_embedder.embed(ocr_text)

        payload = {
            "asset_id": asset_id,
            "document_id": doc.id,
            "page_id": page.id,
            "page_number": page.page_number,
            "asset_type": AssetType.IMAGE.value,
            "filename": doc.filename,
            "patient_id": doc.patient_id,
            "modality": doc.modality,
            "path_or_uri": str(img_save_path),
            "ocr_text": ocr_text[:500] if ocr_text else "",
        }

        if img_vec:
            self.vector_store.upsert_image(
                point_id=asset_id,
                image_vector=img_vec,
                payload=payload,
                text_vector=text_vec,
            )

        # If OCR text, also index it separately as an OCR asset for text search
        if ocr_text:
            await self._index_ocr_asset(doc, page, ocr_text, asset_id, str(img_save_path))

        asset = Asset(
            id=asset_id,
            document_id=doc.id,
            page_id=page.id,
            asset_type=AssetType.IMAGE,
            content=ocr_text or None,
            path_or_uri=str(img_save_path),
            bbox=extracted_img.bbox,
            vector_id=asset_id,
            meta={"ext": extracted_img.ext, "ocr_engine": ocr_result.engine_used},
        )
        self.session.add(asset)

    async def _index_ocr_asset(
        self, doc: Document, page: Page, ocr_text: str, parent_asset_id: str, img_path: str
    ) -> None:
        chunks = chunk_text(ocr_text, settings.chunk_size, settings.chunk_overlap)
        vectors = self.text_embedder.embed_batch(chunks)
        for chunk, vec in zip(chunks, vectors):
            asset_id = new_id()
            payload = {
                "asset_id": asset_id,
                "document_id": doc.id,
                "page_id": page.id,
                "page_number": page.page_number,
                "asset_type": AssetType.OCR.value,
                "filename": doc.filename,
                "patient_id": doc.patient_id,
                "modality": doc.modality,
                "snippet": chunk[:300],
                "parent_image_asset_id": parent_asset_id,
                "path_or_uri": img_path,
            }
            self.vector_store.upsert_text(point_id=asset_id, vector=vec, payload=payload)
            asset = Asset(
                id=asset_id,
                document_id=doc.id,
                page_id=page.id,
                asset_type=AssetType.OCR,
                content=chunk,
                path_or_uri=img_path,
                vector_id=asset_id,
                meta={"parent_image_asset_id": parent_asset_id},
            )
            self.session.add(asset)

    async def _index_table(self, doc: Document, page: Page, table) -> None:
        # Represent table as JSON string for embedding
        table_dict = {"headers": table.headers, "rows": table.rows}
        table_text = json.dumps(table_dict, ensure_ascii=False)
        vec = self.text_embedder.embed(table_text[:2000])
        asset_id = new_id()
        payload = {
            "asset_id": asset_id,
            "document_id": doc.id,
            "page_id": page.id,
            "page_number": page.page_number,
            "asset_type": AssetType.TABLE.value,
            "filename": doc.filename,
            "patient_id": doc.patient_id,
            "snippet": table_text[:300],
        }
        self.vector_store.upsert_text(point_id=asset_id, vector=vec, payload=payload)
        asset = Asset(
            id=asset_id,
            document_id=doc.id,
            page_id=page.id,
            asset_type=AssetType.TABLE,
            content=table_text,
            bbox=table.bbox,
            vector_id=asset_id,
            meta={"headers": table.headers, "row_count": len(table.rows)},
        )
        self.session.add(asset)
