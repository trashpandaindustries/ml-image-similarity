"""Production-quality visual image similarity search engine.

The package is organised into small, single-responsibility modules:

- :mod:`src.config`  - runtime configuration (env-var and CLI overridable).
- :mod:`src.model`   - model loading, preprocessing and embedding generation.
- :mod:`src.store`   - efficient, incremental embedding persistence.
- :mod:`src.embed`   - dataset traversal and batch embedding generation.
- :mod:`src.search`  - similarity search behind a backend-agnostic API.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
