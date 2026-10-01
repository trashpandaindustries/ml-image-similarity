"""Embedding loading, similarity search and top-k retrieval.

Run as a module::

    python -m src.search --query "/path/to/image.jpg" --top-k 5

The search backend sits behind the :class:`SimilaritySearcher` abstraction. When
FAISS is installed a flat inner-product index (``IndexFlatIP``) is used; otherwise
the code falls back to a vectorised NumPy dot-product. Both operate on
L2-normalised vectors, so the inner product *is* cosine similarity and the public
API is identical regardless of backend.

For this dataset (~81k x 512 ``float32`` = ~167 MB) an exact flat index is the
right engineering choice: it returns exact nearest neighbours, needs no training
or tuning, and answers a query in single-digit milliseconds. Approximate indexes
(IVF/HNSW) only earn their added complexity at 10-100x this scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import Config
from .logging_utils import configure_logging
from .store import EmbeddingStore

logger = logging.getLogger("src.search")

# Errors raised by Pillow/decoding when a query image is corrupt or unsupported.
_DECODE_ERRORS: tuple[type[Exception], ...] = (
    UnidentifiedImageError,
    Image.DecompressionBombError,
    OSError,
    ValueError,
)


class InvalidQueryImageError(ValueError):
    """Raised when a query image exists but cannot be opened or decoded.

    Carries a clear, user-facing message; the underlying technical cause is
    preserved via exception chaining and logged at error level.
    """


# --------------------------------------------------------------------------------------
# Backend abstraction
# --------------------------------------------------------------------------------------
class SimilaritySearcher(ABC):
    """Backend-agnostic nearest-neighbour search over unit-norm vectors."""

    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D [N, D] array.")
        self._n, self._dim = embeddings.shape

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return self._n

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim

    @property
    @abstractmethod
    def backend(self) -> str:
        """Human-readable backend name."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-``k`` matches for one or more query vectors.

        Args:
            queries: ``[D]`` or ``[Q, D]`` L2-normalised query vector(s).
            k: Number of neighbours to return per query.

        Returns:
            ``(scores, indices)`` each of shape ``[Q, k]``, sorted by descending
            similarity. Scores are cosine similarities in ``[-1, 1]``.
        """

    @staticmethod
    def _as_2d(queries: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        return q[None, :] if q.ndim == 1 else q


class FaissSearcher(SimilaritySearcher):
    """Exact inner-product search backed by a FAISS ``IndexFlatIP``."""

    def __init__(self, embeddings: np.ndarray) -> None:
        super().__init__(embeddings)
        import faiss

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    @property
    def backend(self) -> str:
        return "faiss.IndexFlatIP"

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self._as_2d(queries)
        k = min(k, self._n)
        scores, indices = self._index.search(q, k)
        return scores, indices


class NumpySearcher(SimilaritySearcher):
    """Vectorised exact search using a single BLAS matrix multiply."""

    def __init__(self, embeddings: np.ndarray) -> None:
        super().__init__(embeddings)
        # Keep a contiguous float32 copy for fast, cache-friendly GEMM.
        self._matrix = np.ascontiguousarray(embeddings, dtype=np.float32)

    @property
    def backend(self) -> str:
        return "numpy.dot"

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self._as_2d(queries)
        k = min(k, self._n)
        # [Q, N] similarity matrix; one GEMM call.
        sims = q @ self._matrix.T
        # argpartition finds the top-k unordered in O(N), then we sort only k.
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        indices = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(part_scores, order, axis=1)
        return scores, indices


def build_searcher(embeddings: np.ndarray, prefer_faiss: bool = True) -> SimilaritySearcher:
    """Construct the best available searcher for the given embeddings.

    Args:
        embeddings: ``[N, D]`` matrix of L2-normalised vectors.
        prefer_faiss: When ``True`` use FAISS if it can be imported.

    Returns:
        A ready-to-query :class:`SimilaritySearcher`.
    """
    if prefer_faiss:
        try:
            searcher: SimilaritySearcher = FaissSearcher(embeddings)
            logger.info("Search backend: %s (%d vectors)", searcher.backend, searcher.size)
            return searcher
        except ImportError:
            logger.info("FAISS not installed; falling back to NumPy backend.")
        except Exception as exc:  # noqa: BLE001 - a broken FAISS must still fall back
            logger.warning("FAISS unavailable (%s); falling back to NumPy backend.", exc)
    searcher = NumpySearcher(embeddings)
    logger.info("Search backend: %s (%d vectors)", searcher.backend, searcher.size)
    return searcher


# --------------------------------------------------------------------------------------
# High-level engine
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchResult:
    """A single ranked search hit."""

    rank: int
    score: float
    path: str  # relative to the dataset root
    abspath: str
    style: str
    artist: str
    title: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimilarityEngine:
    """Loads a persisted index and answers image-similarity queries.

    The engine lazily loads the embedding model only when a query needs to be
    encoded, so read-only workflows (e.g. re-ranking precomputed vectors) never
    pay the model-loading cost.
    """

    def __init__(self, config: Config, prefer_faiss: bool = True, mmap: bool = False) -> None:
        self._config = config
        self._store = EmbeddingStore(config.embeddings_dir)
        index = self._store.load(mmap=mmap)
        self._embeddings = index.embeddings
        self._manifest: pd.DataFrame = index.manifest.reset_index(drop=True)
        self._meta = index.meta
        self._searcher = build_searcher(np.asarray(self._embeddings), prefer_faiss=prefer_faiss)
        self._path_to_row = {p: i for i, p in enumerate(self._manifest["path"])}
        self._model = None  # lazily initialised

    # --- Introspection ------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._searcher.backend

    @property
    def size(self) -> int:
        return self._searcher.size

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    # --- Querying -----------------------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from .model import EmbeddingModel

            self._model = EmbeddingModel(
                self._meta.get("model_name", self._config.model_name),
                self._meta.get("pretrained", self._config.pretrained),
                self._config.device,
                fallback_dim=self._meta.get("output_dim", self._config.output_dim),
            )
        return self._model

    def _row_to_result(self, rank: int, row_idx: int, score: float) -> SearchResult:
        row = self._manifest.iloc[row_idx]
        rel = str(row["path"])
        return SearchResult(
            rank=rank,
            score=float(score),
            path=rel,
            abspath=str(self._config.dataset_root / rel),
            style=str(row.get("style", "")),
            artist=str(row.get("artist", "")),
            title=str(row.get("title", "")),
        )

    def search_vector(
        self, vector: np.ndarray, k: int = 5, exclude_rows: set[int] | None = None
    ) -> list[SearchResult]:
        """Return the top-``k`` results for a precomputed query vector."""
        exclude_rows = exclude_rows or set()
        # Over-fetch so we can drop excluded rows (e.g. the query itself).
        fetch = min(self.size, k + len(exclude_rows) + 1)
        scores, indices = self._searcher.search(vector.astype(np.float32), fetch)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in exclude_rows or idx < 0:
                continue
            results.append(self._row_to_result(len(results) + 1, int(idx), float(score)))
            if len(results) == k:
                break
        return results

    def search(
        self, image_path: str | Path, k: int = 5, exclude_self: bool = True
    ) -> list[SearchResult]:
        """Return the top-``k`` images most similar to ``image_path``.

        Args:
            image_path: Query image on disk (in or outside the dataset).
            k: Number of results to return.
            exclude_self: Drop the query image from results when it is part of
                the indexed corpus.

        Returns:
            Ranked results, most similar first.

        Raises:
            FileNotFoundError: If the query path does not exist.
            InvalidQueryImageError: If the file exists but cannot be decoded.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        try:
            vector = self._get_model().encode_file(str(image_path))
        except _DECODE_ERRORS as exc:
            logger.error("Failed to decode query image %s: %s", image_path, exc)
            raise InvalidQueryImageError(
                f"Could not read '{image_path}' as an image. The file may be "
                f"corrupt, truncated, or in an unsupported format."
            ) from exc

        exclude: set[int] = set()
        if exclude_self:
            try:
                rel = image_path.resolve().relative_to(
                    self._config.dataset_root.resolve()
                ).as_posix()
                if rel in self._path_to_row:
                    exclude.add(self._path_to_row[rel])
            except ValueError:
                pass  # query lives outside the dataset root
        return self.search_vector(vector, k=k, exclude_rows=exclude)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _format_table(query: str, results: list[SearchResult]) -> str:
    lines = [f"\nQuery: {query}", "-" * 72]
    for r in results:
        lines.append(
            f"{r.rank}. score={r.score:.4f}  [{r.style}]  {r.artist} - {r.title}\n"
            f"     {r.path}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search for visually similar images.")
    parser.add_argument("--query", required=True, type=str, help="Path to the query image.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5).")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--embeddings-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-faiss", action="store_true", help="Force the NumPy backend.")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON.")
    parser.add_argument("--log-level", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for similarity search."""
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = Config.from_env(args.dataset_root).merged_with(
        embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else None,
        device=args.device,
    )
    engine = SimilarityEngine(config, prefer_faiss=not args.no_faiss)
    try:
        results = engine.search(args.query, k=args.top_k)
    except (FileNotFoundError, InvalidQueryImageError) as exc:
        # The technical cause is already logged; show the user a clean message.
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
    else:
        print(_format_table(args.query, results))


if __name__ == "__main__":
    main()
