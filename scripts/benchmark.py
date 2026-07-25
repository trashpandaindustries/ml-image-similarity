"""Measure end-to-end performance of the similarity engine on the real index.

All figures are *measured* from the persisted index and live query runs - none
are estimated. Run after ``python -m src.embed`` has produced an index::

    python scripts/benchmark.py --queries 50

The script reports embedding throughput (read from the index metadata written by
``src.embed``), query latency (measured here), index size on disk, embedding
dimensionality and peak process RAM, then writes ``embeddings/benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Allow running as a plain script (``python scripts/benchmark.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.search import SimilarityEngine  # noqa: E402

logger = logging.getLogger("src.benchmark")


def _peak_rss_mb(process: psutil.Process) -> float:
    """Return the peak working-set (Windows) or current RSS in MiB."""
    info = process.memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    return peak / (1024 * 1024)


def _hardware_summary(device: str) -> str:
    cpu = platform.processor() or platform.machine()
    cores = psutil.cpu_count(logical=True)
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001
        pass
    return f"CPU: {cpu} ({cores} logical cores)"


def run_benchmark(config: Config, n_queries: int, top_k: int) -> dict:
    """Run the benchmark and return a dictionary of measured metrics."""
    process = psutil.Process()
    engine = SimilarityEngine(config)
    manifest = engine._manifest  # noqa: SLF001 - internal read for sampling queries
    meta = engine.meta

    # Sample real dataset images as queries.
    sample = manifest.sample(min(n_queries, len(manifest)), random_state=0)
    query_paths = [config.dataset_root / p for p in sample["path"]]

    # Warm up (first call loads the model and JITs the encode path).
    engine.search(query_paths[0], k=top_k)

    end_to_end: list[float] = []
    search_only: list[float] = []
    for path in query_paths:
        t0 = time.perf_counter()
        vec = engine._get_model().encode_file(str(path))  # noqa: SLF001
        t1 = time.perf_counter()
        engine.search_vector(vec, k=top_k)
        t2 = time.perf_counter()
        end_to_end.append((t2 - t0) * 1000.0)
        search_only.append((t2 - t1) * 1000.0)

    emb_path = config.embeddings_path
    index_bytes = emb_path.stat().st_size if emb_path.exists() else 0
    manifest_bytes = config.manifest_path.stat().st_size if config.manifest_path.exists() else 0

    metrics = {
        "hardware": _hardware_summary(meta.get("device", "cpu")),
        "device": meta.get("device", "cpu"),
        "model": f"{meta.get('model_name')} ({meta.get('pretrained')})",
        "num_images": meta.get("count"),
        "embedding_dim": meta.get("dim"),
        "embedding_seconds_total": meta.get("last_embedding_seconds"),
        "images_per_second": meta.get("images_per_second"),
        "search_backend": engine.backend,
        "query_count": len(end_to_end),
        "avg_query_latency_ms": round(statistics.mean(end_to_end), 2),
        "p50_query_latency_ms": round(statistics.median(end_to_end), 2),
        "p95_query_latency_ms": round(float(np.percentile(end_to_end, 95)), 2),
        "avg_search_only_ms": round(statistics.mean(search_only), 3),
        "index_size_mb": round(index_bytes / (1024 * 1024), 2),
        "manifest_size_mb": round(manifest_bytes / (1024 * 1024), 2),
        "peak_rss_mb": round(_peak_rss_mb(process), 1),
    }
    return metrics


def _render_markdown(m: dict) -> str:
    rows = [
        ("Hardware", m["hardware"]),
        ("Model", m["model"]),
        ("Search backend", m["search_backend"]),
        ("Images indexed", f"{m['num_images']:,}"),
        ("Embedding dimensionality", m["embedding_dim"]),
        ("Total embedding time", f"{m['embedding_seconds_total']:.0f} s "
                                 f"({m['embedding_seconds_total'] / 60:.1f} min)"),
        ("Images / second (embedding)", m["images_per_second"]),
        ("Avg query latency (encode + search)", f"{m['avg_query_latency_ms']} ms"),
        ("p95 query latency", f"{m['p95_query_latency_ms']} ms"),
        ("Avg search-only latency", f"{m['avg_search_only_ms']} ms"),
        ("Embedding index size", f"{m['index_size_mb']} MB"),
        ("Manifest size", f"{m['manifest_size_mb']} MB"),
        ("Peak process RAM", f"{m['peak_rss_mb']} MB"),
    ]
    width = max(len(k) for k, _ in rows)
    lines = ["", f"| {'Metric'.ljust(width)} | Value |", f"| {'-' * width} | --- |"]
    lines += [f"| {k.ljust(width)} | {v} |" for k, v in rows]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the similarity engine.")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--embeddings-dir", type=str, default=None)
    parser.add_argument("--queries", type=int, default=50, help="Number of timed queries.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log-level", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the benchmark."""
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)
    random.seed(0)
    config = Config.from_env(args.dataset_root).merged_with(
        embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else None,
    )
    metrics = run_benchmark(config, n_queries=args.queries, top_k=args.top_k)
    out = config.embeddings_dir / "benchmark.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(_render_markdown(metrics))
    logger.info("Benchmark metrics written to %s", out)


if __name__ == "__main__":
    main()
