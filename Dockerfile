# Production image for the WikiArt visual similarity engine.
#
# The dataset and the generated embeddings are mounted at runtime as volumes so
# the image stays small and portable - it contains only code and dependencies.
#
# Build:
#   docker build -t wikiart-similarity .
#
# Generate embeddings (dataset mounted read-only, embeddings mounted read-write):
#   docker run --rm \
#     -v /data/wikiart:/data/wikiart:ro \
#     -v $(pwd)/embeddings:/app/embeddings \
#     -e WIKIART_DATASET_ROOT=/data/wikiart \
#     wikiart-similarity python -m src.embed
#
# Launch the interactive demo on http://localhost:8501:
#   docker run --rm -p 8501:8501 \
#     -v /data/wikiart:/data/wikiart:ro \
#     -v $(pwd)/embeddings:/app/embeddings \
#     -e WIKIART_DATASET_ROOT=/data/wikiart \
#     wikiart-similarity

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
    EMBEDDINGS_DIR=/app/embeddings

WORKDIR /app

# Install CPU-only PyTorch wheels first to avoid pulling large CUDA packages,
# then the remaining dependencies. Layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install -r requirements.txt

COPY src ./src
COPY demo ./demo
COPY scripts ./scripts

# Run as an unprivileged user; pre-create writable mount points.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/embeddings /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Default to the interactive demo; override with any `python -m src.*` command.
CMD ["streamlit", "run", "demo/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
