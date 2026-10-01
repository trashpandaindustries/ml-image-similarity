"""Fast, dataset-free tests for the storage and search layers.

These exercise the parts of the pipeline that do not require the (large) model
download or dataset: index persistence, backend parity and the incremental
manifest fingerprinting logic. Run with ``pytest``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import UnidentifiedImageError

from src.config import Config
from src.search import (
    FaissSearcher,
    InvalidQueryImageError,
    NumpySearcher,
    SimilarityEngine,
    build_searcher,
)
from src.store import MANIFEST_COLUMNS, EmbeddingStore


def _unit_rows(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, d)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def _manifest(n: int) -> pd.DataFrame:
    rows = [
        {
            "path": f"Style/artist_{i}.jpg",
            "style": "Style",
            "artist": f"artist {i}",
            "title": f"title {i}",
            "size": 100 + i,
            "mtime": 1000 + i,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def test_store_roundtrip(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path)
    assert not store.exists()

    embeddings = _unit_rows(20, 512)
    manifest = _manifest(20)
    store.save(embeddings, manifest, {"model_name": "ViT-B-16-SigLIP"})

    assert store.exists()
    loaded = store.load()
    assert loaded.embeddings.shape == (20, 512)
    assert loaded.embeddings.dtype == np.float32
    np.testing.assert_allclose(loaded.embeddings, embeddings, rtol=0, atol=1e-6)
    assert list(loaded.manifest["path"]) == list(manifest["path"])
    assert loaded.meta["count"] == 20 and loaded.meta["dim"] == 512


def test_numpy_search_ranks_self_first() -> None:
    embeddings = _unit_rows(50, 64)
    searcher = NumpySearcher(embeddings)
    scores, indices = searcher.search(embeddings[7], k=3)
    assert indices[0, 0] == 7  # a vector is most similar to itself
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-5)
    # Scores are sorted in descending order.
    assert scores[0, 0] >= scores[0, 1] >= scores[0, 2]


def test_backend_parity() -> None:
    faiss = pytest.importorskip("faiss")  # noqa: F841
    embeddings = _unit_rows(200, 128, seed=1)
    queries = _unit_rows(5, 128, seed=2)

    np_scores, np_idx = NumpySearcher(embeddings).search(queries, k=5)
    fa_scores, fa_idx = FaissSearcher(embeddings).search(queries, k=5)

    np.testing.assert_array_equal(np_idx, fa_idx)
    np.testing.assert_allclose(np_scores, fa_scores, atol=1e-5)


def test_build_searcher_fallback() -> None:
    embeddings = _unit_rows(10, 32)
    forced = build_searcher(embeddings, prefer_faiss=False)
    assert forced.backend == "numpy.dot"
    assert forced.size == 10 and forced.dim == 32


def test_config_env_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIKIART_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("BATCH_SIZE", "17")
    config = Config.from_env()
    assert config.dataset_root == tmp_path
    assert config.batch_size == 17
    # CLI-style override wins over the environment.
    assert config.merged_with(batch_size=99).batch_size == 99
    assert config.merged_with(batch_size=None).batch_size == 17


def test_config_missing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIKIART_DATASET_ROOT", raising=False)
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    with pytest.raises(ValueError):
        Config.from_env()


def _make_engine(tmp_path: Path) -> tuple[SimilarityEngine, np.ndarray]:
    """Build a small on-disk index and a SimilarityEngine (no model loaded)."""
    embeddings = _unit_rows(8, 16)
    EmbeddingStore(tmp_path / "emb").save(
        embeddings, _manifest(8), {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"}
    )
    config = Config(dataset_root=tmp_path, embeddings_dir=tmp_path / "emb")
    return SimilarityEngine(config), embeddings


def test_search_missing_query_raises_filenotfound(tmp_path: Path) -> None:
    engine, _ = _make_engine(tmp_path)
    with pytest.raises(FileNotFoundError):
        engine.search(tmp_path / "does-not-exist.jpg", k=3)


def test_search_corrupt_query_raises_invalid(tmp_path: Path) -> None:
    engine, _ = _make_engine(tmp_path)
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"this is not a valid image")

    class _FakeModel:  # simulates a decode failure without loading the real model
        def encode_file(self, path: str) -> np.ndarray:
            raise UnidentifiedImageError("cannot identify image file")

    engine._model = _FakeModel()  # noqa: SLF001
    with pytest.raises(InvalidQueryImageError) as exc_info:
        engine.search(bad, k=3)
    assert "corrupt.jpg" in str(exc_info.value)


def test_search_valid_query_returns_results(tmp_path: Path) -> None:
    engine, embeddings = _make_engine(tmp_path)
    query = tmp_path / "query.jpg"
    query.write_bytes(b"raw-query-bytes")

    class _FakeModel:  # returns a known indexed vector, bypassing image decoding
        def encode_file(self, path: str) -> np.ndarray:
            return embeddings[2]

    engine._model = _FakeModel()  # noqa: SLF001
    results = engine.search(query, k=3, exclude_self=False)
    assert len(results) == 3
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
