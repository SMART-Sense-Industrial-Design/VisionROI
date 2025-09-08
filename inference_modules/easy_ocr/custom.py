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



from inference_modules.base_ocr import BaseOCR, np, Image, cv2

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


class EasyOCR(BaseOCR):
    MODULE_NAME = "easy_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._reader: easyocr.Reader | None = None  # type: ignore[name-defined]
        self._reader_lock = threading.Lock()

    def _get_reader(self) -> easyocr.Reader:
        if easyocr is None:
            raise RuntimeError("easyocr library is not installed")
        with self._reader_lock:
            if self._reader is None:
                self._reader = easyocr.Reader(["en", "th"], gpu=False)
            return self._reader

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        reader = self._get_reader()
        ocr_result = reader.readtext(frame, detail=0)
        text = " ".join(ocr_result)
        self.logger.info(
            f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {text}"
            if roi_id is not None
            else f"{self.MODULE_NAME} OCR result: {text}"
        )
        if save:
            base_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
            roi_folder = f"{roi_id}" if roi_id is not None else "roi"
            save_dir = base_dir / "images" / roi_folder
            os.makedirs(save_dir, exist_ok=True)
            filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
            path = save_dir / filename
            save_image_async(str(path), frame)

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

