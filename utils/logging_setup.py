"""
Production-grade logging helper for StockSim.
-------------------------------------------

* honours the env-var LOG_LEVEL
* isolates loggers (propagate = False)
* size-rotates each file: 20 MB × 3
* prevents duplicate handlers on hot reload
* optional colourised stdout via **colorlog** (falls back gracefully)
"""

from __future__ import annotations
import logging, os, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

try:
    from colorlog import ColoredFormatter
    HAVE_COLORLOG = True
except ImportError:
    HAVE_COLORLOG = False

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def setup_logger(
    name: str,
    log_file: Union[str, Path],
    *,
    log_to_stdout: bool = True,
    retain_mb: int = 20,
    backups: int = 3,
) -> logging.Logger:
    """
    Return a singleton logger writing to *log_file*.

    The level comes from the env variable LOG_LEVEL (default INFO).
    Changing LOG_LEVEL between calls upgrades/downgrades existing loggers.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)

    # Re-initialise ↔ adjust existing logger if needed
    if getattr(logger, "_configured", False):
        logger.setLevel(level)
        for hdlr in logger.handlers:
            hdlr.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False                # stop bubbling to root

    log_file = Path(log_file).expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------- file handler --------------------------- #
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        log_file,
        maxBytes=retain_mb * 2**20,
        backupCount=backups,
        encoding="utf-8",
        delay=True,                         # open lazily
    )
    fh.setFormatter(file_fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # --------------------------- console handler ------------------------ #
    if log_to_stdout:
        if HAVE_COLORLOG:
            console_fmt = ColoredFormatter(
                "%(asctime)s | "
                "%(log_color)s%(levelname)-8s%(reset)s | "
                "%(name)s | %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red,bg_white',
                },
            )
        else:
            console_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(console_fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    logger._configured = True
    logger.debug("Logger “%s” ready at level %s", name, level_name)
    return logger
