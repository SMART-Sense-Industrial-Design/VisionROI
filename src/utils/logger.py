from __future__ import annotations

import logging
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import threading

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_loggers: dict[tuple[str, str, str, str], logging.Logger] = {}
_lock = threading.Lock()

_SUPPRESS_INFO_MODULES = {
    "easy_ocr",
    "rapid_ocr",
    "typhoon_ocr",
    "base_ocr",
    "tesseract_ocr",
}


class _DropOcrResultFilter(logging.Filter):
    """กรองข้อความ OCR result ที่ต้องรอรวมผลก่อนบันทึก"""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple guard
        if record.levelno <= logging.INFO:
            try:
                message = record.getMessage()
            except Exception:  # pragma: no cover - หาก message มีปัญหาให้ผ่านต่อ
                return True
            if "OCR result" in message:
                return False
        return True

_LOG_ROOT = Path(__file__).resolve().parents[2] / "logs"


def _sanitize_component(name: str | None) -> str:
    if not name:
        return "default"
    sanitized = re.sub(r"[^\w.-]+", "_", name.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "default"


def _normalize_log_filename(name: str | None) -> str:
    if not name:
        return "custom.log"
    sanitized = re.sub(r"[^\w.-]+", "_", name.strip())
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = "custom"
    if not sanitized.lower().endswith(".log"):
        sanitized = f"{sanitized}.log"
    return sanitized


def get_logger(
    module_name: str,
    source: str | None = None,
    log_name: str | None = None,
    subdir: str | None = None,
) -> logging.Logger:
    """คืน logger ที่ตั้งค่า handler ตาม module และ source"""
    src = source or ""
    filename = _normalize_log_filename(log_name)
    normalized_subdir = _sanitize_component(subdir) if subdir else ""
    key = (module_name, src, filename, normalized_subdir)
    logger = _loggers.get(key)
    if logger is not None:
        return logger
    with _lock:
        logger = _loggers.get(key)
        if logger is not None:
            return logger
        logger_name = module_name if not src else f"{module_name}:{src}"
        if normalized_subdir:
            logger_name = f"{logger_name}:{normalized_subdir}"
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            for existing in list(logger.handlers):
                logger.removeHandler(existing)
                existing.close()
        logger.setLevel(logging.INFO)
        primary_component = _sanitize_component(src or module_name)
        log_dir = _LOG_ROOT / f"log_{primary_component}"
        if normalized_subdir:
            log_dir = log_dir / normalized_subdir
        _LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = TimedRotatingFileHandler(
            str(log_dir / filename), when="D", interval=1, backupCount=7
        )
        handler.setFormatter(_formatter)
        if module_name in _SUPPRESS_INFO_MODULES:
            handler.addFilter(_DropOcrResultFilter())
        logger.addHandler(handler)
        logger.propagate = False
        _loggers[key] = logger
        return logger
