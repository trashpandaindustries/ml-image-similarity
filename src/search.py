"""Embedding loading, similarity search and top-k retrieval.

Run as a module::

    python -m src.search --query "/path/to/image.jpg" --top-k 5
    python -m src.search --query img.jpg --identify
    python -m src.search --query img.jpg --style Realism --backend pg

Storage sits behind :class:`~src.backend.VectorBackend`:

* ``FlatFileBackend`` (here): ``embeddings.npy`` + ``manifest.csv`` searched with
  FAISS ``IndexFlatIP`` or a NumPy dot product (exact).
* ``PgVectorBackend`` (``src.pg_backend``): Supabase/Postgres + pgvector HNSW.

Both operate on L2-normalised vectors, so inner product / cosine distance give
identical scores in ``[-1, 1]`` and the public API is the same either way.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .backend import ArtistVote, SearchResult, VectorBackend, split_path
from .config import Config
from .logging_utils import configure_logging
from .store import EmbeddingStore

logger = logging.getLogger("src.search")

__all__ = [
    "ArtistVote",
    "FaissSearcher",
    "FlatFileBackend",
    "InvalidQueryImageError",
    "NumpySearcher",
    "SearchResult",
    "SimilarityEngine",
    "SimilaritySearcher",
    "build_searcher",
    "make_backend",
]

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
# In-memory searcher abstraction (used by the flat-file backend)
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
        self._matrix = np.ascontiguousarray(embeddings, dtype=np.float32)

    @property
    def backend(self) -> str:
        return "numpy.dot"

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self._as_2d(queries)
        k = min(k, self._n)
        sims = q @ self._matrix.T
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        indices = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(part_scores, order, axis=1)
        return scores, indices


def build_searcher(embeddings: np.ndarray, prefer_faiss: bool = True) -> SimilaritySearcher:
    """Construct the best available searcher for the given embeddings."""
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
# Flat-file backend
# --------------------------------------------------------------------------------------
class FlatFileBackend:
    """``VectorBackend`` over the ``embeddings.npy`` + ``manifest.csv`` index."""

    def __init__(self, config: Config, prefer_faiss: bool = True, mmap: bool = False) -> None:
        self._config = config
        index = EmbeddingStore(config.embeddings_dir).load(mmap=mmap)
        self._embeddings = index.embeddings
        self._manifest: pd.DataFrame = index.manifest.reset_index(drop=True)
        self._meta = index.meta
        self._searcher = build_searcher(np.asarray(self._embeddings), prefer_faiss=prefer_faiss)
        paths = self._manifest["path"].tolist()
        self._path_to_row = {p: i for i, p in enumerate(paths)}
        self._slugs = np.array([split_path(p)[1] for p in paths])
        self._styles = self._manifest["style"].fillna("").astype(str).to_numpy()

    @property
    def backend_name(self) -> str:
        return self._searcher.backend

    @property
    def size(self) -> int:
        return self._searcher.size

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

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

    def search(
        self,
        vector: np.ndarray,
        k: int,
        *,
        style: str | None = None,
        artist: str | None = None,
        exclude_path: str | None = None,
    ) -> list[SearchResult]:
        exclude: set[int] = set()
        if exclude_path is not None and exclude_path in self._path_to_row:
            exclude.add(self._path_to_row[exclude_path])
        q = np.asarray(vector, dtype=np.float32)

        if style or artist:
            mask = np.ones(self.size, dtype=bool)
            if style:
                mask &= self._styles == style
            if artist:
                mask &= self._slugs == artist.lower()
            rows = np.flatnonzero(mask)
            if exclude:
                rows = rows[~np.isin(rows, list(exclude))]
            if not len(rows):
                return []
            sims = np.asarray(self._embeddings)[rows] @ q
            top = np.argsort(-sims)[:k]
            return [
                self._row_to_result(i + 1, int(rows[j]), float(sims[j]))
                for i, j in enumerate(top)
            ]

        fetch = min(self.size, k + len(exclude) + 1)
        scores, indices = self._searcher.search(q, fetch)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in exclude or idx < 0:
                continue
            results.append(self._row_to_result(len(results) + 1, int(idx), float(score)))
            if len(results) == k:
                break
        return results

    def identify(self, vector: np.ndarray, k: int = 50, top: int = 5) -> list[ArtistVote]:
        agg: dict[str, dict[str, Any]] = {}
        for hit in self.search(vector, k):
            slug = split_path(hit.path)[1]
            if not slug:
                continue
            entry = agg.setdefault(
                slug, {"name": hit.artist, "weight": 0.0, "votes": 0, "best": -1.0}
            )
            entry["weight"] += hit.score
            entry["votes"] += 1
            entry["best"] = max(entry["best"], hit.score)
        ranked = sorted(agg.items(), key=lambda kv: kv[1]["weight"], reverse=True)[:top]
        return [
            ArtistVote(slug=s, name=e["name"], weight=e["weight"], votes=e["votes"], best=e["best"])
            for s, e in ranked
        ]

    def sample_paths(self, n: int, seed: str | int = 0) -> list[str]:
        state = abs(hash(str(seed))) % (2**32) if not isinstance(seed, int) else seed
        return self._manifest.sample(min(n, len(self._manifest)), random_state=state)[
            "path"
        ].tolist()


def make_backend(config: Config, prefer_faiss: bool = True, mmap: bool = False) -> VectorBackend:
    """Build the storage backend selected by ``config.store_backend``."""
    # line num check
    if config.store_backend == "pg":
        from .pg_backend import PgVectorBackend  # lazy: psycopg is optional for flat mode

        return PgVectorBackend.from_config(config)
    return FlatFileBackend(config, prefer_faiss=prefer_faiss, mmap=mmap)


# --------------------------------------------------------------------------------------
# High-level engine
# --------------------------------------------------------------------------------------
class SimilarityEngine:
    """Answers image-similarity queries against a :class:`VectorBackend`.

    The embedding model is loaded lazily, only when a query image must be
    encoded, so read-only workflows never pay the model-loading cost.
    """

    def __init__(
        self,
        config: Config,
        prefer_faiss: bool = True,
        mmap: bool = False,
        backend: VectorBackend | None = None,
    ) -> None:
        self._config = config
        self._backend: VectorBackend = backend or make_backend(config, prefer_faiss, mmap)
        self._model = None  # lazily initialised

    # --- Introspection ------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend.backend_name

    @property
    def size(self) -> int:
        return self._backend.size

    @property
    def meta(self) -> dict[str, Any]:
        return self._backend.meta

    def sample_paths(self, n: int, seed: str | int = 0) -> list[str]:
        """Return ``n`` indexed relative paths (used by demos and benchmarks)."""
        return self._backend.sample_paths(n, seed)

    # --- Model --------------------------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from .model import EmbeddingModel

            meta = self._backend.meta
            self._model = EmbeddingModel(
                meta.get("model_name", self._config.model_name),
                meta.get("pretrained", self._config.pretrained),
                self._config.device,
                fallback_dim=meta.get("dim", self._config.output_dim),
            )
        return self._model

    def _encode(self, image_path: str | Path) -> tuple[Path, np.ndarray]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")
        try:
            return image_path, self._get_model().encode_file(str(image_path))
        except _DECODE_ERRORS as exc:
            logger.error("Failed to decode query image %s: %s", image_path, exc)
            raise InvalidQueryImageError(
                f"Could not read '{image_path}' as an image. The file may be "
                f"corrupt, truncated, or in an unsupported format."
            ) from exc

    def _relative_key(self, image_path: Path) -> str | None:
        """Path relative to the dataset root, or ``None`` if outside it."""
        try:
            return image_path.resolve().relative_to(self._config.dataset_root.resolve()).as_posix()
        except ValueError:
            return None

    # --- Querying -----------------------------------------------------------------
    def search_vector(
        self,
        vector: np.ndarray,
        k: int = 5,
        exclude_path: str | None = None,
        style: str | None = None,
        artist: str | None = None,
    ) -> list[SearchResult]:
        """Return the top-``k`` results for a precomputed query vector."""
        return self._backend.search(
            vector, k, style=style, artist=artist, exclude_path=exclude_path
        )

    def search(
        self,
        image_path: str | Path,
        k: int = 5,
        exclude_self: bool = True,
        style: str | None = None,
        artist: str | None = None,
    ) -> list[SearchResult]:
        """Return the top-``k`` images most similar to ``image_path``.

        Raises:
            FileNotFoundError: If the query path does not exist.
            InvalidQueryImageError: If the file exists but cannot be decoded.
        """
        path, vector = self._encode(image_path)
        exclude = self._relative_key(path) if exclude_self else None
        return self.search_vector(vector, k=k, exclude_path=exclude, style=style, artist=artist)

    def identify(self, image_path: str | Path, k: int = 50, top: int = 5) -> list[ArtistVote]:
        """Rank likely artists for ``image_path`` by similarity-weighted kNN vote."""
        _, vector = self._encode(image_path)
        return self._backend.identify(vector, k=k, top=top)


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


def _format_votes(query: str, votes: list[ArtistVote]) -> str:
    lines = [f"\nQuery: {query}  (artist vote over nearest neighbours)", "-" * 72]
    for i, v in enumerate(votes, 1):
        lines.append(
            f"{i}. {v.name}  weight={v.weight:.3f}  votes={v.votes}  best={v.best:.4f}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search for visually similar images.")
    parser.add_argument("--query", required=True, type=str, help="Path to the query image.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5).")
    parser.add_argument("--identify", action="store_true", help="Rank likely artists instead.")
    parser.add_argument("--style", type=str, default=None, help="Filter by style folder.")
    parser.add_argument("--artist", type=str, default=None, help="Filter by artist slug.")
    parser.add_argument("--backend", type=str, default=None, choices=["flat", "pg"])
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
        store_backend=args.backend,
    )
    engine = SimilarityEngine(config, prefer_faiss=not args.no_faiss)
    try:
        if args.identify:
            votes = engine.identify(args.query)
            payload: list[dict[str, Any]] = [v.as_dict() for v in votes]
            text = _format_votes(args.query, votes)
        else:
            results = engine.search(
                args.query, k=args.top_k, style=args.style, artist=args.artist
            )
            payload = [r.as_dict() for r in results]
            text = _format_table(args.query, results)
    except (FileNotFoundError, InvalidQueryImageError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(json.dumps(payload, indent=2) if args.json else text)


if __name__ == "__main__":
    main()
