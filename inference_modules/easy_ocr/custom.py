from __future__ import annotations

import time
from PIL import Image
import cv2
import logging
import os
from datetime import datetime
import threading
from pathlib import Path
import gc
from src.utils.logger import get_logger

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None

try:
    import easyocr
except Exception:  # pragma: no cover - fallback when easyocr missing
    easyocr = None

MODULE_NAME = "easy_ocr"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()


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


def process(
    frame,
    roi_id=None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
    interval: float = 1.0,
):
    """ประมวลผล ROI และเรียก OCR เมื่อเวลาห่างจากครั้งก่อน >= interval วินาที
    (ค่าเริ่มต้น 3 วินาที) บันทึกรูปภาพแบบไม่บล็อกเมื่อระบุให้บันทึก"""
    logger = get_logger(MODULE_NAME, source)

    if cam_id is not None:
        try:
            import app  # type: ignore
            app.save_roi_flags[cam_id] = False
        except Exception:  # pragma: no cover
            pass

    if isinstance(frame, Image.Image) and np is not None:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    current_time = time.monotonic()

    with _last_ocr_lock:
        last_time = last_ocr_times.get(roi_id)
    diff_time = 0 if last_time is None else current_time - last_time
    should_ocr = last_time is None or diff_time >= interval

    if should_ocr:
        with _last_ocr_lock:
            last_ocr_times[roi_id] = current_time
        try:
            reader = _get_reader()
            ocr_result = reader.readtext(frame, detail=0)
            text = " ".join(ocr_result)
            logger.info(
                f"roi_id={roi_id} {MODULE_NAME} OCR result: {text}"
                if roi_id is not None
                else f"{MODULE_NAME} OCR result: {text}"
            )
            with _last_ocr_lock:
                last_ocr_results[roi_id] = text
        except Exception as e:  # pragma: no cover - log any OCR error
            logger.exception(f"roi_id={roi_id} {MODULE_NAME} OCR error: {e}")

        if save:
            base_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
            roi_folder = f"{roi_id}" if roi_id is not None else "roi"
            save_dir = base_dir / "images" / roi_folder
            os.makedirs(save_dir, exist_ok=True)
            filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
            path = save_dir / filename
            threading.Thread(target=_save_image_async, args=(str(path), frame), daemon=True).start()

        return last_ocr_results.get(roi_id, "")

    return None


def stop(roi_id) -> None:
    """ลบข้อมูลของ ROI ที่หยุดใช้งาน"""
    with _last_ocr_lock:
        last_ocr_times.pop(roi_id, None)
        last_ocr_results.pop(roi_id, None)


def cleanup() -> None:
    """รีเซ็ตสถานะของโมดูลและบังคับเก็บขยะ"""
    global _reader
    with _reader_lock:
        _reader = None
    with _last_ocr_lock:
        last_ocr_times.clear()
        last_ocr_results.clear()
    gc.collect()
