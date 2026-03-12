# =============================================================================
# utils/logger.py  –  Per-process rotating file logger
# =============================================================================
"""
Each process calls setup_process_logger(process_name) once at startup.
All subsequent get_logger(name) calls return child loggers of that root.

Log files  →  logs/<process_name>.log  (10 MB × 5 backups)
Console    →  INFO and above
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_root_logger: Optional[logging.Logger] = None
_process_name: str = "main"


def setup_process_logger(process_name: str,
                         log_dir: str = "logs",
                         level: int = logging.DEBUG) -> logging.Logger:
    """Configure logging for the calling process.

    Must be called once at the start of each subprocess.
    Returns the root logger for this process.
    """
    global _root_logger, _process_name
    _process_name = process_name

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"IVS.{process_name}")
    logger.setLevel(level)
    logger.handlers.clear()

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        log_path / f"{process_name}.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

    _root_logger = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger.  setup_process_logger must have been called."""
    global _root_logger
    if _root_logger is None:
        # Fallback: basic stderr logger (should not happen in production)
        _root_logger = logging.getLogger("IVS.unknown")
        if not _root_logger.handlers:
            _root_logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s"
            ))
            _root_logger.addHandler(ch)
    return _root_logger.getChild(name)
