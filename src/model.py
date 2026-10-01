"""Model loading, preprocessing and embedding generation.

This module wraps an OpenCLIP vision encoder behind a small, typed interface
(:class:`EmbeddingModel`). Callers never touch OpenCLIP or PyTorch details
directly, which keeps the rest of the codebase model-agnostic.

Design notes
------------
* **OpenCLIP ViT-B-32 (laion2b_s34b_b79k)** is the default. It produces
  512-dimensional embeddings and offers the best quality/latency trade-off for
  CPU-only inference at this dataset's scale (see the README for the rationale).
* Embeddings are **L2-normalised** at generation time so that an inner product
  is exactly cosine similarity. This lets the search layer use a plain
  ``IndexFlatIP`` / dot-product without re-normalising per query.
* JPEG decoding uses Pillow's :meth:`~PIL.Image.Image.draft` mode. Because every
  image is downscaled to 224 px, DCT-scaled decoding of large source JPEGs is
  ~2x faster with no measurable quality impact on the embeddings.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger("src.model")

# Route Python's SSL through the OS trust store when available. This transparently
# fixes model downloads on machines behind a corporate proxy (whose root CA is
# trusted by the OS but absent from certifi) and is a harmless no-op elsewhere.
try:  # pragma: no cover - environment dependent
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - optional dependency; never fatal
    pass

import open_clip  # noqa: E402  (imported after the optional truststore hook)
import torch  # noqa: E402


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference into a concrete :class:`torch.device`.

    Args:
        preference: ``"auto"``, ``"cpu"`` or ``"cuda"``. ``"auto"`` selects CUDA
            when available and falls back to CPU otherwise.

    Returns:
        The resolved device.
    """
    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    if pref != "auto":
        raise ValueError(f"Unknown device preference: {preference!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingModel:
    """A pretrained OpenCLIP image encoder with a minimal, typed API.

    Args:
        model_name: OpenCLIP architecture (e.g. ``"ViT-B-32"``).
        pretrained: OpenCLIP pretrained tag (e.g. ``"laion2b_s34b_b79k"``).
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)

        logger.info(
            "Loading OpenCLIP %s (%s) on %s", model_name, pretrained, self.device
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval().to(self.device)

        # PyTorch defaults CPU inference to the physical-core count, which is
        # optimal for this GEMM-bound ViT; override with the standard
        # ``OMP_NUM_THREADS`` / ``torch.set_num_threads`` if needed.
        if self.device.type == "cpu":
            logger.info("CPU inference using %d threads", torch.get_num_threads())

        self._model = model
        self._preprocess = preprocess
        self._embedding_dim = 768
        self._image_size = self._infer_image_size(model)
        logger.info(
            "Model ready: dim=%d, input=%dpx", self._embedding_dim, self._image_size
        )

    # --- Introspection ------------------------------------------------------------
    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the produced embeddings."""
        return self._embedding_dim

    @property
    def image_size(self) -> int:
        """Square input resolution expected by the model (pixels)."""
        return self._image_size

    @staticmethod
    def _infer_image_size(model: torch.nn.Module) -> int:
        size = getattr(model.visual, "image_size", 224)
        if isinstance(size, (tuple, list)):
            return int(size[0])
        return int(size)

    # --- Image loading & preprocessing --------------------------------------------
    def load_image(self, path: str) -> Image.Image:
        """Open an image as RGB using fast draft-mode JPEG decoding.

        Args:
            path: Filesystem path to the image.

        Returns:
            The decoded RGB image.
        """
        image = Image.open(path)
        # DCT-scaled decode; a hint only, ignored for non-JPEG formats.
        image.draft("RGB", (self._image_size, self._image_size))
        return image.convert("RGB")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply the model's preprocessing transform to a PIL image."""
        return self._preprocess(image)

    def preprocess_file(self, path: str) -> torch.Tensor:
        """Load and preprocess an image file into a model-ready tensor."""
        return self._preprocess(self.load_image(path))

    # --- Embedding generation ------------------------------------------------------
    @torch.inference_mode()
    def encode_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Encode a preprocessed batch into L2-normalised embeddings.

        Args:
            batch: Tensor of shape ``[B, C, H, W]``.

        Returns:
            ``float32`` array of shape ``[B, D]`` with unit-norm rows.
        """
        batch = batch.to(self.device, non_blocking=True)
        features = self._model.encode_image(batch)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.detach().cpu().to(torch.float32).numpy()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Encode a sequence of PIL images into L2-normalised embeddings."""
        batch = torch.stack([self._preprocess(img) for img in images])
        return self.encode_batch(batch)

    def encode_file(self, path: str) -> np.ndarray:
        """Encode a single image file into a ``[D]`` L2-normalised vector."""
        return self.encode_images([self.load_image(path)])[0]