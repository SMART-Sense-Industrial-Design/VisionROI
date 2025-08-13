from __future__ import annotations

import time
from PIL import Image
import cv2
from logging.handlers import TimedRotatingFileHandler
import logging
import os
from datetime import datetime
import threading
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None

try:
    import easyocr
except Exception:  # pragma: no cover - fallback when easyocr missing
    easyocr = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler: TimedRotatingFileHandler | None = None
_current_source: str | None = None
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()
_reader_run_lock = threading.Lock()


def _configure_logger(source: str | None) -> None:
    """ตั้งค่า handler ของ logger ให้บันทึกตามโฟลเดอร์ของ source"""
    global _handler, _current_source
    source = source or ""
    if _current_source == source:
        return
    log_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "custom.log"
    if _handler:
        logger.removeHandler(_handler)
        _handler.close()
    _handler = TimedRotatingFileHandler(str(log_path), when="D", interval=1, backupCount=7)
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _current_source = source


def _get_reader() -> easyocr.Reader:
    """สร้างและคืนค่า easyocr.Reader แบบ singleton"""
    if easyocr is None:
        raise RuntimeError("easyocr library is not installed")
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = easyocr.Reader(["en", "th"], gpu=False)
        return _reader


# ตัวแปรควบคุมเวลาเรียก OCR แยกตาม roi พร้อมตัวล็อกป้องกันการเข้าถึงพร้อมกัน
last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()
_imwrite_lock = threading.Lock()


def _save_image_async(path, image) -> None:
    """บันทึกรูปภาพแบบแยกเธรด"""
    try:
        with _imwrite_lock:
            cv2.imwrite(path, image)
    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to save image {path}: {e}")


def _run_ocr_async(frame, roi_id, save, source) -> None:
    """ประมวลผล OCR และบันทึกรูปในเธรดแยก"""
    try:
        reader = _get_reader()
        with _reader_run_lock:
            ocr_result = reader.readtext(frame, detail=0)
        text = " ".join(ocr_result)
        logger.info(
            f"roi_id={roi_id} OCR result: {text}" if roi_id is not None else f"OCR result: {text}"
        )
        with _last_ocr_lock:
            last_ocr_results[roi_id] = text
    except Exception as e:  # pragma: no cover - log any OCR error
        logger.exception(f"roi_id={roi_id} OCR error: {e}")

    if save:
        base_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
        roi_folder = f"{roi_id}" if roi_id is not None else "roi"
        save_dir = base_dir / "images" / roi_folder
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        path = save_dir / filename
        _save_image_async(str(path), frame)


def process(
    frame,
    roi_id=None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
):
    """ประมวลผล ROI และเรียก OCR เมื่อเวลาห่างจากครั้งก่อน >= 2 วินาที
    บันทึกรูปภาพแบบไม่บล็อกเมื่อระบุให้บันทึก"""

    _configure_logger(source)

    if cam_id is not None:
        try:
            import app  # type: ignore
            app.save_roi_flags[cam_id] = True
        except Exception:  # pragma: no cover
            pass

    if isinstance(frame, Image.Image) and np is not None:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    current_time = time.monotonic()

    with _last_ocr_lock:
        last_time = last_ocr_times.get(roi_id)
        diff_time = 0 if last_time is None else current_time - last_time
        if last_time is None or diff_time >= 3:
            last_ocr_times[roi_id] = current_time
            should_ocr = True
        else:
            should_ocr = False

    result_text = last_ocr_results.get(roi_id, "")

    if should_ocr:
        threading.Thread(
            target=_run_ocr_async, args=(frame.copy(), roi_id, save, source), daemon=True
        ).start()

    return result_text
