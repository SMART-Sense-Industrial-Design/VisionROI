from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import threading

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_loggers: dict[tuple[str, str], logging.Logger] = {}
_lock = threading.Lock()

_DATA_SOURCES_ROOT = Path(__file__).resolve().parents[2] / "data_sources"
_INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "inference_modules"


def get_logger(module_name: str, source: str | None = None) -> logging.Logger:
    """คืน logger ที่ตั้งค่า handler ตาม module และ source"""
    src = source or ""
    key = (module_name, src)
    logger = _loggers.get(key)
    if logger is not None:
        return logger
    with _lock:
        logger = _loggers.get(key)
        if logger is not None:
            return logger
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.INFO)
        if src:
            log_dir = _DATA_SOURCES_ROOT / src
        else:
            log_dir = _INFERENCE_ROOT / module_name
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = TimedRotatingFileHandler(str(log_dir / "custom.log"), when="D", interval=1, backupCount=7)
        handler.setFormatter(_formatter)
        logger.addHandler(handler)
        logger.propagate = False
        _loggers[key] = logger
        return logger
