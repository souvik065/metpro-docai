"""
MacPro AI — Central configuration.
All tunables live here; override via environment variables or .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")

    # ── Database ───────────────────────────────────────────────────────────
    # Swap to postgresql+asyncpg://user:pass@host/db for production
    database_url: str = "sqlite+aiosqlite:///data/macpro.db"

    # ── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_path: str = "data/qdrant"          # local file-based mode
    qdrant_host: str = ""                     # set to use remote Qdrant
    qdrant_port: int = 6333
    qdrant_collection: str = "macpro_medical"

    # ── Embedding models ───────────────────────────────────────────────────
    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # CLIP: used for image embeddings AND cross-modal text→image retrieval
    clip_model: str = "openai/clip-vit-base-patch32"
    text_embed_dim: int = 384
    image_embed_dim: int = 512

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 512        # tokens
    chunk_overlap: int = 64

    # ── OCR ────────────────────────────────────────────────────────────────
    ocr_engine: str = "easyocr"  # "easyocr" | "tesseract"
    ocr_languages: list[str] = ["en"]

    # ── LLM (for answer synthesis) ─────────────────────────────────────────
    llm_provider: str = "openai"          # "anthropic" | "openai" | "ollama" | "azure_openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 1024

    # ── Retrieval ──────────────────────────────────────────────────────────
    top_k_text: int = 5
    top_k_image: int = 3
    hybrid_alpha: float = 0.5   # weight for dense vs sparse (future BM25)

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"


settings = Settings()
