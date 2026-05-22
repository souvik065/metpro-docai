# MacPro AI — Dockerfile
# Multi-stage build: keeps final image lean and pre-caches AI models

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /build

# System deps for PyMuPDF, OpenCV, EasyOCR, and building python wheels (strictly pinned)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx=22.3.6-1+deb12u1 \
    libglib2.0-0=2.74.6-2+deb12u9 \
    libgomp1=12.2.0-14+deb12u1 \
    git=1:2.39.5-0+deb12u3 \
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
FROM python:3.11.9-slim-bookworm AS runtime

WORKDIR /app

# Copy virtual environment (with all dependencies)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded AI models cache
COPY --from=builder /usr/share/model-cache/ /usr/share/model-cache/
ENV HF_HOME=/usr/share/model-cache/huggingface
ENV EASYOCR_MODULE_PATH=/usr/share/model-cache/easyocr

# System runtime libs only (strictly pinned)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx=22.3.6-1+deb12u1 \
    libglib2.0-0=2.74.6-2+deb12u9 \
    libgomp1=12.2.0-14+deb12u1 \
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
