# =============================================================================
# InsightPulse — Application Dockerfile
# =============================================================================
# Multi-stage build for the FastAPI + LangGraph application.
# Stage 1: Install dependencies in a builder image.
# Stage 2: Copy only what's needed into a slim runtime image.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first: the default PyPI wheel drags in the multi-GB CUDA
# stack, which a demo container never uses.
RUN pip install --no-cache-dir --prefix=/install \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency specification AND sources — hatchling cannot build the
# wheel from pyproject.toml alone (packages = ["src/insightpulse"], and the
# declared readme must exist for metadata).
COPY pyproject.toml README.md ./
COPY src/ src/
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim image with only runtime dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install minimal runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ /app/src/
COPY config/ /app/config/
COPY data/ /app/data/

# Create storage directory for SQLite and artifacts
RUN mkdir -p /app/storage && chown -R appuser:appuser /app

USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "insightpulse.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
