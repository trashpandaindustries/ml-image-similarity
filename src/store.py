"""Efficient, incremental embedding persistence.

The on-disk index is three small, transparent artifacts inside ``embeddings/``:

- ``embeddings.npy``   - ``float32`` ``[N, D]`` matrix of L2-normalised vectors.
- ``manifest.csv``     - one row per image: relative path, metadata, and the
  ``size``/``mtime`` fingerprint used to detect changes for incremental runs.
- ``index_meta.json``  - model identity, dimensionality, counts and timing.

Row *i* of ``embeddings.npy`` corresponds to row *i* of ``manifest.csv``. Keeping
the vectors in a contiguous ``.npy`` array (rather than one file per image) makes
loading a single ``mmap``-able read and keeps the search layer allocation-free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("src.store")

# Columns persisted in the manifest, in order. ``size`` and ``mtime`` form the
# change-detection fingerprint; the rest are for display and enrichment.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "path",
    "style",
    "artist",
    "title",
    "size",
    "mtime",
)


@dataclass(frozen=True)
class IndexData:
    """An in-memory view of the persisted index.

    Attributes:
        embeddings: ``float32`` ``[N, D]`` matrix of L2-normalised vectors.
        manifest: Per-row metadata aligned with ``embeddings``.
        meta: Index metadata sidecar (model, dim, counts, timings).
    """

    embeddings: np.ndarray
    manifest: pd.DataFrame
    meta: dict[str, Any]

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])


class EmbeddingStore:
    """Reads and writes the three-file embedding index atomically."""

    def __init__(self, embeddings_dir: Path) -> None:
        self.dir = Path(embeddings_dir)
        self.embeddings_path = self.dir / "embeddings.npy"
        self.manifest_path = self.dir / "manifest.csv"
        self.meta_path = self.dir / "index_meta.json"

    # --- Existence ---------------------------------------------------------------
    def exists(self) -> bool:
        """Return ``True`` when a complete index is present on disk."""
        return (
            self.embeddings_path.exists()
            and self.manifest_path.exists()
            and self.meta_path.exists()
        )

    # --- Loading -----------------------------------------------------------------
    def load_manifest(self) -> pd.DataFrame:
        """Load the manifest as a DataFrame (empty if none exists)."""
        if not self.manifest_path.exists():
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        df = pd.read_csv(self.manifest_path, dtype={"path": str})
        df["path"] = df["path"].astype(str)
        return df

    def load_meta(self) -> dict[str, Any]:
        """Load the index metadata sidecar (empty dict if none exists)."""
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, mmap: bool = False) -> IndexData:
        """Load the full index into memory.

        Args:
            mmap: When ``True`` the embedding matrix is memory-mapped rather than
                read into RAM. Useful for very large indexes on the search path.

        Raises:
            FileNotFoundError: If no complete index exists.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"No embedding index found in {self.dir}. Run `python -m src.embed` first."
            )
        embeddings = np.load(
            self.embeddings_path, mmap_mode="r" if mmap else None
        )
        manifest = self.load_manifest()
        meta = self.load_meta()
        if len(manifest) != embeddings.shape[0]:
            raise ValueError(
                "Index corrupt: manifest rows (%d) != embedding rows (%d)."
                % (len(manifest), embeddings.shape[0])
            )
        logger.info(
            "Loaded index: %d vectors x %d dims (%s)",
            embeddings.shape[0],
            embeddings.shape[1],
            "mmap" if mmap else "in-memory",
        )
        return IndexData(embeddings=embeddings, manifest=manifest, meta=meta)

    # --- Saving ------------------------------------------------------------------
    def save(
        self,
        embeddings: np.ndarray,
        manifest: pd.DataFrame,
        meta: dict[str, Any],
    ) -> None:
        """Persist the index atomically (write-temp-then-rename per file).

        Args:
            embeddings: ``float32`` ``[N, D]`` matrix to store.
            manifest: Metadata DataFrame with :data:`MANIFEST_COLUMNS`.
            meta: Index metadata sidecar to store as JSON.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.shape[0] != len(manifest):
            raise ValueError("embeddings and manifest length mismatch on save.")

        meta = {
            **meta,
            "count": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        self._atomic_np_save(self.embeddings_path, embeddings)
        self._atomic_write(
            self.manifest_path,
            lambda p: manifest[list(MANIFEST_COLUMNS)].to_csv(p, index=False),
        )
        self._atomic_write(
            self.meta_path,
            lambda p: p.write_text(json.dumps(meta, indent=2), encoding="utf-8"),
        )
        logger.info(
            "Saved index: %d vectors -> %s", embeddings.shape[0], self.dir
        )

    # --- Atomic write helpers ----------------------------------------------------
    @staticmethod
    def _atomic_np_save(path: Path, array: np.ndarray) -> None:
        # Write through a file handle so np.save does not append its own ".npy"
        # suffix to the temp name, then rename atomically into place.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, array)
        tmp.replace(path)

    @staticmethod
    def _atomic_write(path: Path, writer) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        writer(tmp)
        tmp.replace(path)
