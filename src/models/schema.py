import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from sqlmodel import Column, Field, JSON, Relationship, SQLModel


# ── Enumerations ──────────────────────────────────────────────────────────────

class AssetType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    OCR = "ocr"
    URL = "url"
    DICOM = "dicom"


class FileType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    DICOM = "dicom"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

# ─────────────────────────────────────────────────────────────
# ── Database tables ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
class Document(SQLModel, table=True):
    """Top-level record for an ingested file."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    filename: str
    file_type: FileType
    file_path: str
    page_count: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # Medical / DICOM / study metadata (nullable)
    patient_id: Optional[str] = None
    study_date: Optional[str] = None
    modality: Optional[str] = None
    study_description: Optional[str] = None
    extra_meta: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    # Realtionships 
    pages: list["Page"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

    assets: list["Asset"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class Page(SQLModel, table=True):
    """One page of a document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    page_number: int
    raw_text: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None
    image_path: Optional[str] = None   # rendered page PNG, if needed

    # Relationships

    document: Optional["Document"] = Relationship(
        back_populates="pages"
    )

    assets: list["Asset"] = Relationship(
        back_populates="page",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class Asset(SQLModel, table=True):
    """
    Extracted content from document.

    Can be:
    - text chunk
    - table
    - image
    - OCR
    - URL
    - DICOM image
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    page_id: Optional[str] = Field(default=None, foreign_key="page.id", index=True)
    asset_type: AssetType
    # Human-readable content (text, OCR text, table as CSV/JSON, URL string)
    content: Optional[str] = None
    # File path for binary assets (images, rendered tables)
    path_or_uri: Optional[str] = None
    # Bounding box on the source page [x0, y0, x1, y1]
    bbox: Optional[list[float]] = Field(default=None, sa_column=Column(JSON))
    # Qdrant point ID for this asset's vector
    vector_id: Optional[str] = None
    # Extra metadata (table headers, OCR confidence, URL context, etc.)
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships

    document: Optional["Document"] = Relationship(
        back_populates="assets"
    )

    page: Optional["Page"] = Relationship(
        back_populates="assets"
    )


# ─────────────────────────────────────────────────────────────
# ── API Models Pydantic-only models (request/response, not DB tables) ───────────────────
# ─────────────────────────────────────────────────────────────

class SourceReference(SQLModel):
    """A single piece of evidence returned with a query answer."""
    type: AssetType
    document_id: str
    filename: str
    page: Optional[int] = None
    asset_id: str
    snippet: Optional[str] = None
    path_or_uri: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(SQLModel):
    """Top-level response object for /query endpoint."""
    answer: str
    sources: list[SourceReference] = []
    query: str
    processing_time_ms: float


class IngestRequest(SQLModel):
    """Request body for /ingest endpoint."""
    folder_path: str
    recursive: bool = True


class IngestResponse(SQLModel):
    """Response for /ingest endpoint."""
    job_id: str
    message: str
    files_queued: int


class IndexStats(SQLModel):
    documents: int
    pages: int
    assets: int
    vectors: int


# ─────────────────────────────────────────────────────────────
# Fix Forward References
# ─────────────────────────────────────────────────────────────

Document.model_rebuild()
Page.model_rebuild()
Asset.model_rebuild()