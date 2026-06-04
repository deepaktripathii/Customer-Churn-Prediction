"""Logging helpers with a consistent format across scripts, API and dashboard."""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger once.

    Repeated calls are cheap and do not duplicate handlers, which matters when
    uvicorn imports the app and the scripts import the same modules.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level or os.getenv("RETAINIQ_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = logging.getLevelName(resolved.upper())
    logging.basicConfig(
        level=resolved,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
    )
    # Uvicorn access logs are already formatted by the server itself.
    logging.getLogger("uvicorn.access").propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
