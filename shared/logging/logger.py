"""
shared.logging.logger — Structured logging configuration.

Uses loguru for its superior ergonomics over the stdlib logging module:
- Automatic serialization to JSON
- Structured context (bind)
- Exception formatting with full traceback
- Log rotation and retention
- Zero-configuration sink management

Usage:
    from shared.logging.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Detection complete", num_detections=284, model="yolo11s")
    logger.bind(session_id="abc123").info("Processing frame")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger


def configure_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    serialize: bool = False,
    colorize: bool = True,
) -> None:
    """
    Configure global logging.

    Should be called once at application startup (in main.py or CLI entry).

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for persistent logging.
        rotation: Log rotation size (e.g., "100 MB", "1 week")
        retention: Log retention period (e.g., "30 days")
        serialize: If True, output JSON-formatted logs (for log aggregation)
        colorize: If True, use ANSI colors in terminal output.
    """
    # Remove default sink
    _loguru_logger.remove()

    # Console sink
    _loguru_logger.add(
        sys.stderr,
        level=log_level,
        colorize=colorize,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=True,
        serialize=serialize,
    )

    # File sink (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_file),
            level=log_level,
            rotation=rotation,
            retention=retention,
            serialize=True,  # Always JSON in files for parsing
            compression="gz",
            backtrace=True,
            diagnose=True,
        )


def get_logger(name: str) -> "BoundLogger":
    """
    Get a logger bound to a module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Starting pipeline")
    """
    return BoundLogger(name)


class BoundLogger:
    """
    Thin wrapper around loguru that binds a module name automatically.

    Supports structured logging via keyword arguments:
        logger.info("Detection done", count=42, model="yolo11s")
    """

    def __init__(self, name: str, **context: Any) -> None:
        self._name = name
        self._context = context
        self._logger = _loguru_logger.bind(module=name, **context)

    def bind(self, **kwargs: Any) -> "BoundLogger":
        """Return a new logger with additional context."""
        new = BoundLogger(self._name, **{**self._context, **kwargs})
        return new

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).debug(self._format(message, kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).info(self._format(message, kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).warning(self._format(message, kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).error(self._format(message, kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).critical(self._format(message, kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1, exception=True).error(self._format(message, kwargs))

    @staticmethod
    def _format(message: str, kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return message
        kv_str = " | ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{message} | {kv_str}"
