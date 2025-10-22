"""Centrilised logging helpers for the trapiche package.

"""
from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a package-aware logger. If name is None the package logger
    (``trapiche``) is returned.
    """
    if name is None:
        name = __package__ or "trapiche"
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """
    Parameters
    - level: logging level (defaults to INFO)
    - fmt: optional format string. If omitted a concise default is used.
    """
    logger = get_logger()
    # If already configured with handlers, don't add duplicates
    if logger.handlers:
        logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    if fmt is None:
        fmt = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
