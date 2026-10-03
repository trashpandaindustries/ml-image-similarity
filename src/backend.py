"""Storage-agnostic read interface for the similarity engine.

``SimilarityEngine`` talks only to a :class:`VectorBackend`. Two implementations
exist: ``FlatFileBackend`` (``src.search``; npy + manifest.csv) and
``PgVectorBackend`` (``src.pg_backend``; Supabase/Postgres + pgvector). This
module deliberately imports neither, so both can depend on it without cycles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

import numpy as np


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


@dataclass(frozen=True)
class ArtistVote:
    """Similarity-weighted vote for an artist among the nearest neighbours."""

    slug: str
    name: str
    weight: float  # sum of cosine scores of that artist's neighbours
    votes: int     # how many of the k neighbours belong to the artist
    best: float    # highest single cosine score

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_path(rel_path: str) -> tuple[str, str, str, str]:
    """Derive ``(style, artist_slug, artist_name, title)`` from a relative path.

    ``style/artist/title.ext`` keys the artist off the folder name. Flat layouts
    (``style/artist_title.ext``) fall back to the filename prefix before the
    first underscore. ``artist_slug`` is ``""`` when no artist can be derived.
    """
    parts = PurePosixPath(rel_path).parts
    stem = PurePosixPath(rel_path).stem
    style = parts[0] if len(parts) > 1 else ""
    if len(parts) >= 3:
        raw_artist, raw_title = parts[1], stem
    else:
        raw_artist, _, raw_title = stem.partition("_")
        raw_title = raw_title or stem
    slug = raw_artist.strip().lower()
    name = raw_artist.replace("-", " ").replace("_", " ").strip()
    title = raw_title.replace("-", " ").replace("_", " ").strip()
    return style, slug, name, title


@runtime_checkable
class VectorBackend(Protocol):
    """Read-side operations the engine needs from any storage backend."""

    @property
    def backend_name(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def meta(self) -> dict[str, Any]: ...

    def search(
        self,
        vector: np.ndarray,
        k: int,
        *,
        style: str | None = None,
        artist: str | None = None,
        exclude_path: str | None = None,
    ) -> list[SearchResult]:
        """Top-``k`` neighbours of a unit-norm vector, optionally filtered."""

    def identify(self, vector: np.ndarray, k: int = 50, top: int = 5) -> list[ArtistVote]:
        """Aggregate the ``k`` nearest neighbours into ranked artist votes."""

    def sample_paths(self, n: int, seed: str | int = 0) -> list[str]:
        """Return ``n`` indexed relative paths (deterministic for a given seed)."""
