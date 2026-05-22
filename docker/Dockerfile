# MacPro AI — Dockerfile
# Multi-stage build: keeps final image lean and pre-caches AI models

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for PyMuPDF, OpenCV, EasyOCR, and building python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate dependencies cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache AI models to prevent runtime download delays and timeouts
ENV HF_HOME=/usr/share/model-cache/huggingface
ENV EASYOCR_MODULE_PATH=/usr/share/model-cache/easyocr

COPY docker/download_models.py .
RUN python download_models.py

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment (with all dependencies)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded AI models cache
COPY --from=builder /usr/share/model-cache/ /usr/share/model-cache/
ENV HF_HOME=/usr/share/model-cache/huggingface
ENV EASYOCR_MODULE_PATH=/usr/share/model-cache/easyocr

# System runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY src/ src/
COPY config/ config/
COPY tests/ tests/
COPY pyproject.toml .

# Ensure Python can find the project root
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
