"""Centralised logging configuration.

A single :func:`configure_logging` helper is used by every entry point so that
CLI tools, the demo and the benchmark all emit consistent, timestamped logs.
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Configure the root logger once and return the package logger.

    Args:
        level: Logging level as a name (``"INFO"``) or numeric value. When
            ``None`` the ``LOG_LEVEL`` environment variable is used, defaulting
            to ``"INFO"``.

    Returns:
        The ``"src"`` package logger, ready for use by callers.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        root.addHandler(handler)

    # Keep the root logger quiet so third-party INFO chatter (e.g. OpenCLIP's
    # model-construction logs, emitted on the root logger) is suppressed, while
    # our own package logger emits at the requested level via the same handler.
    root.setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(level)

    for noisy in ("PIL", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("src")
