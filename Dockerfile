# Production image for the WikiArt visual similarity engine (CPU-only PyTorch).
#
# Prefer docker-compose.yml, which wires up the dataset mount, persistent
# embeddings volume, persistent model cache and shared memory for you.

FROM python:3.11-slim AS runtime

# libgomp1 provides the OpenMP runtime required by FAISS and PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    WIKIART_DATASET_ROOT=/data/wikiart \
    EMBEDDINGS_DIR=/app/embeddings \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# CPU-only PyTorch first (avoids multi-GB CUDA wheels), then the rest.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install -r requirements.txt

COPY src ./src
COPY demo ./demo
COPY scripts ./scripts

# Unprivileged user. The dirs are created + chowned here so that *named volumes*
# mounted over them inherit the right ownership on first use.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/embeddings /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

# Default: interactive demo. Override with any `python -m src.*` command.
CMD ["streamlit", "run", "demo/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]