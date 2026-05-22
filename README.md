# MacPro AI — Multimodal Medical RAG System

A production-ready pipeline for ingesting, indexing, and querying large collections of medical documents including PDFs, scanned images, X-rays, and DICOM files.

---

## Architecture Overview

```
                       ┌─────────────────────────────────────────────────┐
                       │                  MacPro AI                      │
                       │                                                 │
   Medical Files  ───► │  Ingestion Pipeline                             │
   (PDF/IMG/DCM)       │  ┌──────────┐  ┌─────┐  ┌──────────────────┐  │
                       │  │ Parsers  │─►│ OCR │─►│   Embeddings     │  │
                       │  │ PyMuPDF  │  │Easy │  │ text: MiniLM     │  │
                       │  │pdfplumber│  │ OCR │  │ image: CLIP      │  │
                       │  │ pydicom  │  └─────┘  └────────┬─────────┘  │
                       │  └──────────┘                    │            │
                       │                         ┌────────▼──────────┐ │
                       │                         │   Qdrant (local)  │ │
                       │  Metadata store         │ named vectors:    │ │
                       │  SQLite/PostgreSQL       │  "text" (384-dim) │ │
                       │  (Document/Page/Asset)  │  "image"(512-dim) │ │
                       │                         └────────┬──────────┘ │
                       │                                  │            │
   User Query    ───► │  Retrieval Pipeline               │            │
                       │  ┌─────────────────────────────────────────┐  │
                       │  │  Filter extraction → text search        │  │
                       │  │                    + CLIP image search  │  │
                       │  │              → merge & re-rank          │  │
                       │  └───────────────────────┬─────────────────┘  │
                       │                          │                    │
                       │  LLM Synthesis (Claude/GPT/Ollama)            │
                       │                          │                    │
   Structured    ◄─── │  { answer, sources: [{type, doc, page, ...}] }│
   Response            └─────────────────────────────────────────────-─┘
```

---

## Design Decisions

| Concern | Choice | Reason |
|---|---|---|
| PDF parsing | **PyMuPDF** (primary) + pdfplumber (tables) | Fastest library; handles encrypted & scanned PDFs; pdfplumber adds accurate table detection |
| OCR | **EasyOCR** (primary) → pytesseract (fallback) | Better accuracy on medical scans; no system binary; GPU-ready |
| Text embeddings | **sentence-transformers/all-MiniLM-L6-v2** | 384-dim, runs on CPU, no API key needed |
| Image embeddings | **CLIP (ViT-B/32)** | Shared text+image space enables cross-modal retrieval without fine-tuning |
| Vector DB | **Qdrant** (local file mode) | Named vectors allow text and image embeddings on same point; zero-setup local mode; swappable to remote |
| Metadata store | **SQLite** (dev) / PostgreSQL (prod) | SQLModel unifies ORM + Pydantic schemas; swap via one env var |
| Orchestration | **Custom pipeline** (not LlamaIndex/LangChain) | Fewer hidden abstractions; each component is explicitly swappable |
| LLM | **Anthropic Claude** (configurable) | Also supports OpenAI and Ollama via the same interface |
| API | **FastAPI** | Async-native; auto OpenAPI docs at `/docs` |
| Medical images | **pydicom** | Industry standard; pixel extraction → PIL → CLIP |

---

## Project Structure

```
macpro-ai/
├── config/
│   └── settings.py          # All tunables (env-overridable)
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app + lifespan
│   │   └── routes/
│   │       ├── health.py    # GET /health
│   │       ├── ingest.py    # POST /ingest/folder, /ingest/file
│   │       └── query.py     # POST /query, GET /query/search
│   ├── embeddings/
│   │   └── models.py        # TextEmbedder + ImageEmbedder (CLIP)
│   ├── indexing/
│   │   └── vector_store.py  # Qdrant wrapper (text + image named vectors)
│   ├── ingestion/
│   │   └── pipeline.py      # Orchestrates parse → OCR → embed → store
│   ├── models/
│   │   └── schema.py        # SQLModel ORM tables + Pydantic response models
│   ├── ocr/
│   │   └── engine.py        # EasyOCR + Tesseract with unified interface
│   ├── parsers/
│   │   ├── pdf_parser.py    # PyMuPDF + pdfplumber
│   │   ├── dicom_parser.py  # pydicom → PNG
│   │   └── image_parser.py  # PIL for standalone images
│   ├── retrieval/
│   │   ├── pipeline.py      # Query → filter → text+image search → merge
│   │   └── synthesizer.py   # LLM answer generation
│   └── utils/
│       ├── database.py      # Async SQLAlchemy engine + session
│       └── helpers.py       # Logging, chunking, file detection
├── tests/
│   ├── fixtures/
│   │   └── generators.py    # Synthetic PDF/PNG/DICOM fixture builders
│   └── unit/
│       ├── test_helpers.py
│       ├── test_pdf_parser.py
│       ├── test_ocr.py
│       ├── test_retrieval.py
│       └── test_ingestion.py
├── data/
│   ├── input/               # Drop medical files here
│   ├── output/              # Extracted assets (images, etc.)
│   └── qdrant/              # Local Qdrant storage
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── ingest.py                # CLI: python ingest.py --folder data/input
├── query.py                 # CLI: python query.py "blood test results"
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quickstart (Local, No Docker)

### 1. Clone and install

```bash
git clone <repo>
cd macpro-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY (or switch LLM_PROVIDER=ollama)
```

### 3. Drop files in the input folder

```bash
cp /path/to/your/medical/pdfs/*.pdf data/input/
cp /path/to/xrays/*.dcm data/input/
```

### 4. Ingest via CLI

```bash
python ingest.py --folder data/input
```

### 5. Query via CLI

```bash
python query.py "Show the report and X-ray related to lung infection"
python query.py "Give blood test results for patient P001"
python query.py "Find the page where the diagnosis is mentioned"
```

---

## Quickstart (Docker)

```bash
cp .env.example .env  # fill in API keys
docker-compose up --build
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### `POST /ingest/folder`
Ingest an entire folder of medical files (runs as background job).

```json
// Request
{ "folder_path": "/app/data/input", "recursive": true }

// Response
{ "job_id": "uuid", "message": "Ingestion started.", "files_queued": 42 }
```

### `POST /ingest/file`
Upload and ingest a single file via multipart form.

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@chest_xray.dcm"
```

### `GET /ingest/status/{job_id}`
Check background ingestion job status.

### `POST /query`
Natural-language query with LLM synthesis.

```json
// Request
{
  "query": "Show the X-ray and report related to lung infection",
  "patient_id": "P001",   // optional filter
  "modality": "CR",       // optional filter
  "top_k": 5              // optional
}

// Response
{
  "answer": "The chest X-ray (CR) from 2024-03-15 shows right lower lobe infiltrate...",
  "query": "Show the X-ray and report related to lung infection",
  "processing_time_ms": 843.2,
  "sources": [
    {
      "type": "image",
      "document_id": "abc123",
      "filename": "chest.dcm",
      "page": 1,
      "asset_id": "xyz789",
      "snippet": "DICOM image. Modality: CR. Study: Chest PA...",
      "path_or_uri": "data/output/assets/abc123/chest.png",
      "score": 0.921
    },
    {
      "type": "text",
      "document_id": "def456",
      "filename": "radiology_report.pdf",
      "page": 2,
      "asset_id": "uvw012",
      "snippet": "Impression: Right lower lobe pneumonia...",
      "path_or_uri": null,
      "score": 0.887
    }
  ]
}
```

### `GET /query/search?q=...&top_k=5`
Pure vector search without LLM synthesis — useful for debugging.

---

# To get Run
```bash
tar -xzf macpro-ai.tar.gz && cd macpro-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY

# Ingest
python ingest.py --folder data/input

# Query
python query.py "Show X-ray and report for lung infection"

# Or run the API
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```


## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Tests use mocked embedders, OCR, and vector store — no GPU or API keys needed.

---

## Swapping Components

Everything is designed for easy substitution:

| Component | How to swap |
|---|---|
| LLM | Change `LLM_PROVIDER` in `.env` (anthropic / openai / ollama) |
| Embedding model | Change `TEXT_EMBED_MODEL` or `CLIP_MODEL` in `.env` |
| OCR engine | Change `OCR_ENGINE=tesseract` in `.env` |
| Vector DB | Edit `src/indexing/vector_store.py` — implement `upsert_text`, `upsert_image`, `search_text`, `search_image` |
| Metadata DB | Change `DATABASE_URL` to a PostgreSQL URL |
| PDF parser | Replace `PDFParser` in `src/parsers/` — implement `parse() → list[ParsedPage]` |

---

## Extending the Pipeline

### Add a new file type
1. Create `src/parsers/my_parser.py` implementing `parse() → list[ParsedPage]`
2. Add a new `FileType` enum value in `src/models/schema.py`
3. Add a branch in `IngestionPipeline.ingest_file()` in `src/ingestion/pipeline.py`

### Add BM25 sparse search (hybrid retrieval)
1. Install `rank_bm25`
2. Build a BM25 index over all `Asset.content` values at startup
3. In `RetrievalPipeline.retrieve()`, add BM25 scores and blend with the dense scores using `hybrid_alpha`

### Add a reranker
After merging text and image hits in `RetrievalPipeline._merge_results()`, pass results through a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).

---

## Production Checklist

- [ ] Switch `DATABASE_URL` to PostgreSQL
- [ ] Set `QDRANT_HOST` to a dedicated Qdrant instance
- [ ] Enable GPU for EasyOCR (`gpu=True` in `src/ocr/engine.py`)
- [ ] Add auth (FastAPI `HTTPBearer` or OAuth2)
- [ ] Replace in-memory `_jobs` dict in `ingest.py` with Celery + Redis
- [ ] Add object storage (S3/GCS) for extracted assets instead of local disk
- [ ] Add HIPAA-compliant logging and audit trails
- [ ] Set up model caching to avoid re-downloading on each container restart
