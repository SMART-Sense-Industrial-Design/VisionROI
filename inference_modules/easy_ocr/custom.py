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
import gc
from queue import Queue, Empty, Full

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

_roi_queues: dict = {}
_worker_thread: threading.Thread | None = None
_worker_stop_event = threading.Event()
_worker_thread_lock = threading.Lock()


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


def _worker() -> None:
    """worker หลักที่ดึงงานจากคิวของแต่ละ ROI"""
    while not _worker_stop_event.is_set():
        processed = False
        for roi_id, q in list(_roi_queues.items()):
            try:
                frame, save, source = q.get_nowait()
            except Empty:
                continue
            processed = True
            try:
                _run_ocr_async(frame, roi_id, save, source)
            finally:
                q.task_done()
        if not processed:
            time.sleep(0.01)


def _ensure_worker() -> None:
    global _worker_thread
    with _worker_thread_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_stop_event.clear()
            _worker_thread = threading.Thread(target=_worker, daemon=True)
            _worker_thread.start()


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
    should_ocr = last_time is None or diff_time >= interval

    result_text = last_ocr_results.get(roi_id, "")
    if should_ocr:
        with _last_ocr_lock:
            last_ocr_times[roi_id] = current_time
        q = _roi_queues.setdefault(roi_id, Queue(maxsize=1))
        item = (frame.copy(), save, source)
        try:
            q.put_nowait(item)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(item)
            except Full:
                pass
        _ensure_worker()

    return result_text


def stop(roi_id) -> None:
    """ลบข้อมูลและคิวของ ROI ที่หยุดใช้งาน"""
    q = _roi_queues.pop(roi_id, None)
    if q:
        with q.mutex:
            q.queue.clear()
            q.unfinished_tasks = 0
            q.all_tasks_done.notify_all()
    with _last_ocr_lock:
        last_ocr_times.pop(roi_id, None)
        last_ocr_results.pop(roi_id, None)


def cleanup() -> None:
    """รีเซ็ตสถานะของโมดูลและบังคับเก็บขยะ"""
    global _reader, _handler, _current_source, _worker_thread
    _worker_stop_event.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=1)
    _worker_thread = None
    _roi_queues.clear()
    with _reader_lock:
        _reader = None
    with _last_ocr_lock:
        last_ocr_times.clear()
        last_ocr_results.clear()
    if _handler:
        logger.removeHandler(_handler)
        try:
            _handler.close()
        finally:
            _handler = None
    _current_source = None
    gc.collect()
