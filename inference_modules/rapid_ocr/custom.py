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
    from rapidocr import RapidOCR
except Exception:  # pragma: no cover - fallback when rapidocr missing
    RapidOCR = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler: TimedRotatingFileHandler | None = None
_current_source: str | None = None
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_reader = None
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


def _run_ocr_async(frame, roi_id, save, source) -> None:
    """ประมวลผล OCR และบันทึกรูปในเธรดแยก"""
    try:
        reader = _get_reader()
        with _reader_run_lock:
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
            app.save_roi_flags[cam_id] = False
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
