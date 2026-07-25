"""Runtime configuration.

All tunables live in a single immutable :class:`Config` dataclass. Values are
resolved with the precedence *CLI argument > environment variable > default*,
so the same code runs unchanged on a laptop, a CI runner or a GPU box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# Repository root (…/ml-image-similarity), derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Image extensions treated as dataset images. The WikiArt corpus is entirely
# ``.jpg``; the rest are included so the pipeline generalises to mixed corpora.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Top-level directory names that are never image sources (report figures,
# analysis notebooks, and our own output directory when nested under the root).
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {"notebooks", "reports", "embeddings", ".git", "__pycache__"}
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the whole pipeline.

    Attributes:
        dataset_root: Directory containing the image corpus. Read from
            ``WIKIART_DATASET_ROOT`` or ``DATASET_ROOT`` when not passed.
        embeddings_dir: Directory where embeddings and metadata are persisted.
        model_name: OpenCLIP architecture name (e.g. ``"ViT-B-32"``).
        pretrained: OpenCLIP pretrained tag (e.g. ``"laion2b_s34b_b79k"``).
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        batch_size: Number of images per forward pass.
        num_workers: DataLoader worker processes used for image decoding.
        image_extensions: File extensions treated as images.
        exclude_dirs: Top-level directory names skipped during traversal.
    """

    dataset_root: Path
    embeddings_dir: Path = REPO_ROOT / "embeddings"
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"
    batch_size: int = 64
    num_workers: int = 6
    checkpoint_every: int = 10_000
    image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS

    # --- Derived persistence paths -------------------------------------------------
    @property
    def embeddings_path(self) -> Path:
        """Path to the ``float32`` ``[N, D]`` embedding matrix."""
        return self.embeddings_dir / "embeddings.npy"

    @property
    def manifest_path(self) -> Path:
        """Path to the per-image metadata manifest (CSV)."""
        return self.embeddings_dir / "manifest.csv"

    @property
    def meta_path(self) -> Path:
        """Path to the index metadata sidecar (JSON)."""
        return self.embeddings_dir / "index_meta.json"

    # --- Constructors --------------------------------------------------------------
    @classmethod
    def from_env(cls, dataset_root: str | os.PathLike[str] | None = None) -> "Config":
        """Build a configuration from environment variables and defaults.

        Args:
            dataset_root: Explicit dataset root; overrides the environment.

        Raises:
            ValueError: If no dataset root can be resolved.
        """
        root = (
            Path(dataset_root).expanduser()
            if dataset_root is not None
            else _env_path("WIKIART_DATASET_ROOT") or _env_path("DATASET_ROOT")
        )
        if root is None:
            raise ValueError(
                "Dataset root not set. Pass --dataset-root or set "
                "WIKIART_DATASET_ROOT / DATASET_ROOT."
            )

        embeddings_dir = _env_path("EMBEDDINGS_DIR") or (cls.embeddings_dir)
        return cls(
            dataset_root=root,
            embeddings_dir=embeddings_dir,
            model_name=os.environ.get("MODEL_NAME", cls.model_name),
            pretrained=os.environ.get("MODEL_PRETRAINED", cls.pretrained),
            device=os.environ.get("DEVICE", cls.device),
            batch_size=_env_int("BATCH_SIZE", cls.batch_size),
            num_workers=_env_int("NUM_WORKERS", cls.num_workers),
            checkpoint_every=_env_int("CHECKPOINT_EVERY", cls.checkpoint_every),
        )

    def merged_with(self, **overrides: object) -> "Config":
        """Return a copy with the given non-``None`` fields overridden."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean) if clean else self

    def validate(self) -> "Config":
        """Validate that the dataset root exists and return ``self``.

        Raises:
            FileNotFoundError: If the dataset root does not exist.
            NotADirectoryError: If the dataset root is not a directory.
        """
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.dataset_root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.dataset_root}")
        return self
