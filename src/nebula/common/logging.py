"""Structured logging (structlog -> stdout + JSON lines file under logs/)."""

from __future__ import annotations

import logging
import sys

import structlog

from nebula.common.paths import get_paths

_CONFIGURED = False


def configure_logging(level: str = "INFO", to_file: bool = True) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if to_file:
        logs = get_paths().logs
        logs.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logs / "nebula.jsonl", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
