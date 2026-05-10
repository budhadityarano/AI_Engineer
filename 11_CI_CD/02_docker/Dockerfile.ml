# Multi-stage ML Docker image
# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ─────────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean production image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Security: run as non-root
RUN groupadd --gid 1001 mluser && \
    useradd --uid 1001 --gid mluser --no-create-home mluser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Set environment
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/mluser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Switch to non-root
USER mluser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
