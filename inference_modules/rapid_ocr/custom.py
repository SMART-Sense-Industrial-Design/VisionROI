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
    from rapidocr import RapidOCR
except Exception:  # pragma: no cover - fallback when rapidocr missing
    RapidOCR = None  # type: ignore[assignment]

MODULE_NAME = "rapid_ocr"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    """สร้างและคืนค่า RapidOCR แบบ singleton"""
    if RapidOCR is None:
        raise RuntimeError("rapidocr library is not installed")
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = RapidOCR()
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


def _run_ocr_async(frame, roi_id, save, source) -> str:
    """ประมวลผล OCR และบันทึกรูป"""
    try:
        reader = _get_reader()
        # RapidOCR อาจคืนผลลัพธ์เป็น tuple (result, time) หรือเพียง result อย่างเดียว
        result = reader(frame)
        if (
            isinstance(result, (list, tuple))
            and len(result) == 2
            and isinstance(result[0], (list, tuple))
            and not isinstance(result[1], (list, tuple, dict))
        ):
            ocr_result = result[0]
        else:
            ocr_result = result

        text_items: list[str] = []
        if isinstance(ocr_result, (list, tuple)):
            for res in ocr_result:
                if isinstance(res, (list, tuple)) and len(res) > 1:
                    text_items.append(str(res[1]))
                elif isinstance(res, dict) and "text" in res:
                    text_items.append(str(res["text"]))
        elif isinstance(ocr_result, dict) and "text" in ocr_result:
            text_items.append(str(ocr_result["text"]))
        elif hasattr(ocr_result, "text"):
            text_items.append(str(getattr(ocr_result, "text")))
        elif hasattr(ocr_result, "texts"):
            texts_attr = getattr(ocr_result, "texts")
            if isinstance(texts_attr, (list, tuple)):
                text_items.extend(str(t) for t in texts_attr)
            elif texts_attr is not None:
                text_items.append(str(texts_attr))
        elif hasattr(ocr_result, "txts"):
            txts_attr = getattr(ocr_result, "txts")
            if isinstance(txts_attr, (list, tuple)):
                text_items.extend(str(t) for t in txts_attr)
            elif txts_attr is not None:
                text_items.append(str(txts_attr))
        text = " ".join(text_items)

        logger.info(
            f"roi_id={roi_id} {MODULE_NAME} OCR result: {text}"
            if roi_id is not None
            else f"{MODULE_NAME} OCR result: {text}"
        )
        with _last_ocr_lock:
            last_ocr_results[roi_id] = text
    except Exception as e:  # pragma: no cover - log any OCR error
        logger.exception(f"roi_id={roi_id} {MODULE_NAME} OCR error: {e}")
        text = ""

    if save:
        base_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
        roi_folder = f"{roi_id}" if roi_id is not None else "roi"
        save_dir = base_dir / "images" / roi_folder
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        path = save_dir / filename
        _save_image_async(str(path), frame)

    return text


def process(
    frame,
    roi_id=None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
    interval: float = 3.0,
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
        return _run_ocr_async(frame, roi_id, save, source)

    return None


def stop(roi_id) -> None:
    """ลบข้อมูลของ ROI ที่หยุดใช้งาน"""
    with _last_ocr_lock:
        last_ocr_times.pop(roi_id, None)
        last_ocr_results.pop(roi_id, None)


def cleanup() -> None:
    """รีเซ็ตสถานะและคืนทรัพยากรที่ใช้โดยโมดูล OCR"""
    global _reader

    with _reader_lock:
        _reader = None

    with _last_ocr_lock:
        last_ocr_times.clear()
        last_ocr_results.clear()

    gc.collect()
