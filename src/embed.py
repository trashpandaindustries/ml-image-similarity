"""Dataset traversal, batch embedding generation and persistence.

Run as a module::

    python -m src.embed --dataset-root "/path/to/wikiart"

Behaviour
---------
* **Filesystem is authoritative.** Images are discovered by walking the dataset
  root; ``classes.csv`` (if present) is joined purely as metadata *enrichment*.
* **Incremental by default.** Each run fingerprints every image by ``(size,
  mtime)``. Only new or modified images are encoded; unchanged vectors are
  copied forward, and images deleted from disk are dropped from the index.
* **Memory-efficient.** Image decoding is parallelised across DataLoader worker
  processes while the model runs a single batched forward pass at a time, so RAM
  is bounded by ``batch_size`` rather than the dataset size.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config
from .logging_utils import configure_logging
from .model import EmbeddingModel
from .store import MANIFEST_COLUMNS, EmbeddingStore

# Some WikiArt JPEGs are slightly truncated; allow Pillow to load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("src.embed")


@dataclass(frozen=True)
class ImageRecord:
    """A single discovered image and its change-detection fingerprint."""

    path: str  # POSIX path relative to the dataset root (the manifest key)
    abspath: str
    style: str
    size: int
    mtime: int


# --------------------------------------------------------------------------------------
# Dataset traversal & metadata
# --------------------------------------------------------------------------------------
def scan_images(config: Config) -> list[ImageRecord]:
    """Walk the dataset root and return a deterministic list of image records.

    Top-level directories in :attr:`Config.exclude_dirs` (report figures,
    notebooks, our own output) are skipped so only true dataset images are
    indexed.

    Args:
        config: Active configuration.

    Returns:
        Image records sorted by relative path for reproducible ordering.
    """
    root = config.dataset_root
    exts = {e.lower() for e in config.image_extensions}
    records: list[ImageRecord] = []

    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name in config.exclude_dirs:
            continue
        candidates: Iterable[Path] = [entry] if entry.is_file() else entry.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            stat = path.stat()
            rel = path.relative_to(root)
            records.append(
                ImageRecord(
                    path=rel.as_posix(),
                    abspath=str(path),
                    style=rel.parts[0] if len(rel.parts) > 1 else "",
                    size=int(stat.st_size),
                    mtime=int(stat.st_mtime),
                )
            )

    records.sort(key=lambda r: r.path)
    logger.info("Discovered %d images under %s", len(records), root)
    return records


def load_metadata(config: Config) -> dict[str, str]:
    """Load an optional ``path -> artist`` map from ``classes.csv``.

    The join is best-effort enrichment; a missing or malformed CSV is logged and
    ignored rather than failing the run.

    Args:
        config: Active configuration.

    Returns:
        Mapping from relative image path to artist name (may be empty).
    """
    csv_path = config.dataset_root / "classes.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, usecols=["filename", "artist"], dtype=str)
        mapping = dict(zip(df["filename"].astype(str), df["artist"].fillna("")))
        logger.info("Loaded artist metadata for %d images from classes.csv", len(mapping))
        return mapping
    except Exception as exc:  # noqa: BLE001 - enrichment must never be fatal
        logger.warning("Could not read classes.csv (%s); continuing without it.", exc)
        return {}


def _title_from_path(rel_path: str) -> tuple[str, str]:
    """Derive ``(artist_slug, title)`` from a ``style/artist_title.ext`` path."""
    p = Path(rel_path)
    stem = p.stem
    if len(p.parts) >= 3:  # style/artist/title.ext
        artist = p.parts[1].replace("-", " ").replace("_", " ").strip()
        title = stem.replace("-", " ").replace("_", " ").strip()
        return artist, title
    artist_slug, _, title_slug = stem.partition("_")

    artist = artist_slug.replace("-", " ").strip()
    title = (title_slug or stem).replace("-", " ").strip()
    return artist, title


def build_manifest_row(record: ImageRecord, artist_lookup: dict[str, str]) -> dict[str, Any]:
    """Build a single manifest row, preferring CSV metadata over parsed names."""
    parsed_artist, title = _title_from_path(record.path)
    artist = artist_lookup.get(record.path) or parsed_artist
    return {
        "path": record.path,
        "style": record.style,
        "artist": artist,
        "title": title,
        "size": record.size,
        "mtime": record.mtime,
    }


# --------------------------------------------------------------------------------------
# Torch dataset for parallel decoding
# --------------------------------------------------------------------------------------
def _load_and_preprocess(
    abspath: str, preprocess: Callable[[Image.Image], torch.Tensor], image_size: int
) -> torch.Tensor | None:
    """Decode (draft mode) and preprocess one image; return ``None`` on failure."""
    try:
        image = Image.open(abspath)
        image.draft("RGB", (image_size, image_size))
        return preprocess(image.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop the run
        logger.warning("Skipping unreadable image %s (%s)", abspath, exc)
        return None


class ImageDataset(Dataset):
    """Yields ``(global_index, preprocessed_tensor | None)`` for a set of files.

    Only the (picklable) preprocessing transform is sent to worker processes, so
    the model weights are never duplicated across workers.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        preprocess: Callable[[Image.Image], torch.Tensor],
        image_size: int,
    ) -> None:
        self._records = records
        self._preprocess = preprocess
        self._image_size = image_size

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor | None]:
        tensor = _load_and_preprocess(
            self._records[index].abspath, self._preprocess, self._image_size
        )
        return index, tensor


def _collate(
    batch: list[tuple[int, torch.Tensor | None]]
) -> tuple[list[int], torch.Tensor | None]:
    """Drop failed decodes and stack the rest into one batch tensor."""
    good = [(idx, tensor) for idx, tensor in batch if tensor is not None]
    if not good:
        return [], None
    indices = [idx for idx, _ in good]
    tensors = torch.stack([tensor for _, tensor in good])
    return indices, tensors


def iter_embeddings(
    model: EmbeddingModel, records: list[ImageRecord], config: Config
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(record_index, vector)`` for each successfully encoded image.

    Decoding is parallelised across worker processes while the model runs one
    batched forward pass at a time, so peak RAM is bounded by ``batch_size``
    rather than the dataset size. Unreadable images are simply not yielded.

    Args:
        model: Loaded embedding model.
        records: Images to encode.
        config: Active configuration (batch size, workers).

    Yields:
        Tuples of the record's index within ``records`` and its L2-normalised
        embedding vector.
    """
    dataset = ImageDataset(records, model.preprocess, model.image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=(model.device.type == "cuda"),
        persistent_workers=False,
    )
    for indices, batch in loader:
        if batch is None:
            continue
        vectors = model.encode_batch(batch)
        for offset, index in enumerate(indices):
            yield index, vectors[offset]


# --------------------------------------------------------------------------------------
# Incremental orchestration
# --------------------------------------------------------------------------------------
def _reusable_rows(
    manifest: pd.DataFrame, meta: dict[str, Any], config: Config
) -> dict[str, int]:
    """Map ``path -> existing row index`` for vectors that can be reused.

    A previous vector is reusable only when the model identity matches; a model
    change forces a full rebuild.
    """
    if not len(manifest):
        return {}
    if meta.get("model_name") != config.model_name or meta.get("pretrained") != config.pretrained:
        logger.warning(
            "Existing index built with a different model (%s/%s); rebuilding fully.",
            meta.get("model_name"),
            meta.get("pretrained"),
        )
        return {}
    return {row.path: i for i, row in enumerate(manifest.itertuples(index=False))}


def run(config: Config, limit: int | None = None) -> dict[str, Any]:
    """Generate or incrementally update the embedding index.

    Args:
        config: Active configuration.
        limit: Optional cap on the number of images (useful for smoke tests).

    Returns:
        The index metadata sidecar that was persisted.
    """
    config.validate()
    store = EmbeddingStore(config.embeddings_dir)

    records = scan_images(config)
    if limit is not None:
        records = records[:limit]
        logger.info("Limiting run to first %d images.", len(records))
    if not records:
        raise RuntimeError(f"No images found under {config.dataset_root}.")

    prev = store.load() if store.exists() else None
    prev_manifest = prev.manifest if prev is not None else pd.DataFrame(columns=MANIFEST_COLUMNS)
    prev_meta = prev.meta if prev is not None else {}
    reusable = _reusable_rows(prev_manifest, prev_meta, config)

    
    #m
    fingerprints = {
        row.path: (int(row.size), int(row.mtime))
        for row in prev_manifest.itertuples(index=False)
    } if len(prev_manifest) else {}

    # Vectors we can carry forward unchanged (same path, same size/mtime).
    reused: dict[str, np.ndarray] = {}
    if prev is not None:
        for record in records:
            row = reusable.get(record.path)
            if row is not None and fingerprints.get(record.path) == (record.size, record.mtime):
                reused[record.path] = prev.embeddings[row]

    to_embed = [r for r in records if r.path not in reused]
    removed = max(0, len(prev_manifest) - len(reused)) if prev is not None else 0
    logger.info(
        "Index plan: %d total, %d reused, %d to (re)embed, %d removed.",
        len(records), len(reused), len(to_embed), removed,
    )

    if not to_embed and prev is not None and len(prev_manifest) == len(records):
        logger.info("Index already up to date; nothing to do.")
        return prev_meta

    model = EmbeddingModel(
        config.model_name,
        config.pretrained,
        config.device,
        fallback_dim=config.output_dim,
    )
    artist_lookup = load_metadata(config)

    def persist(embedded: dict[str, np.ndarray], seconds: float, final: bool) -> None:
        """Assemble reused + embedded vectors (filesystem order) and save."""
        vectors = np.empty((len(records), model.embedding_dim), dtype=np.float32)
        rows: list[dict[str, Any]] = []
        write = 0
        for record in records:
            vec = reused.get(record.path)
            if vec is None:
                vec = embedded.get(record.path)
            if vec is None:
                continue  # not embedded yet (checkpoint) or unreadable (final)
            vectors[write] = vec
            rows.append(build_manifest_row(record, artist_lookup))
            write += 1
        meta = {
            "model_name": config.model_name,
            "pretrained": config.pretrained,
            "device": model.device.type,
            "normalized": True,
            "metric": "cosine (inner product on unit vectors)",
            "last_images_embedded": len(embedded),
            "last_embedding_seconds": round(seconds, 3),
            "images_per_second": round(len(embedded) / seconds, 2) if seconds else None,
            "full_build": prev is None,
            "complete": final,
        }
        store.save(vectors[:write], pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)), meta)

    embedded: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    with tqdm(total=len(to_embed), desc="Embedding", unit="img", smoothing=0.05) as bar:
        for index, vector in iter_embeddings(model, to_embed, config):
            embedded[to_embed[index].path] = vector
            bar.update(1)


            # Periodic checkpoint so a long run can resume after interruption.

            
            
            if config.checkpoint_every and len(embedded) % config.checkpoint_every == 0:
                persist(embedded, time.perf_counter() - start, final=False)
                logger.info("Checkpoint saved at %d embedded images.", len(embedded))

    elapsed = time.perf_counter() - start
    failed = len(to_embed) - len(embedded)
    if failed:
        logger.warning("%d image(s) could not be encoded and were skipped.", failed)
    logger.info(
        "Embedded %d images in %.1fs (%.1f img/s).",
        len(embedded), elapsed, (len(embedded) / elapsed) if elapsed else 0.0,
    )

    persist(embedded, elapsed, final=True)
    meta = store.load_meta()
    if config.store_backend == "pg":
        index = store.load()
        from .pg_backend import PgVectorBackend  # optional dependency for flat-file users

        PgVectorBackend.from_config(config).sync(index.embeddings, index.manifest, meta)
        logger.info("Synced %d vectors to pgvector.", len(index))
    return meta


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate WikiArt image embeddings.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to the image corpus.")
    parser.add_argument("--embeddings-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--model", type=str, default=None, help="OpenCLIP architecture.")
    parser.add_argument("--pretrained", type=str, default=None, help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Persist a resumable checkpoint every N embedded images.")
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N images.")
    parser.add_argument("--backend", choices=["flat", "pg", "supabase"], default="supabase", help="Optionally sync the completed index to pgvector.")
    parser.add_argument("--log-level", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for embedding generation."""
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = Config.from_env(args.dataset_root).merged_with(
        embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else None,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        store_backend="pg" if args.backend == "supabase" else args.backend,
    )
    meta = run(config, limit=args.limit)
    logger.info("Done. Index holds %d vectors of dim %d.", meta.get("count"), meta.get("dim"))


if __name__ == "__main__":
    main()
