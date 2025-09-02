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
from queue import Queue, Empty
from dataclasses import dataclass
import numpy as np
from typing import Any

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover
    easyocr = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler: TimedRotatingFileHandler | None = None
_current_source: str | None = None
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_reader: Any | None = None
_reader_lock = threading.Lock()
_reader_call_lock = threading.Lock()

@dataclass
class OcrTask:
    source: str
    roi_id: int | str | None
    frame_bgr: np.ndarray
    save: bool

# key = (source, roi_id)
_queues: dict[tuple[str, int | str | None], Queue[OcrTask]] = {}
_workers: dict[tuple[str, int | str | None], threading.Thread] = {}
_qw_lock = threading.Lock()  # protect _queues/_workers creation

# per-(source, roi) bookkeeping
_last_ocr_times: dict[tuple[str, int | str | None], float] = {}
_last_ocr_results: dict[tuple[str, int | str | None], str] = {}
_state_lock = threading.Lock()

_imwrite_lock = threading.Lock()

# === NEW: per-source stop flags ===
_stop_events: dict[str, threading.Event] = {}


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


def _get_reader() -> Any:
    """สร้างและคืนค่า easyocr.Reader แบบ singleton"""
    if easyocr is None:
        raise RuntimeError("easyocr library is not installed")
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = easyocr.Reader(["en", "th"], gpu=False)
        return _reader


def _save_image_async(path, image) -> None:
    """บันทึกรูปภาพแบบแยกเธรด"""
    try:
        with _imwrite_lock:
            cv2.imwrite(path, image)
    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to save image {path}: {e}")


def _run_ocr(frame_bgr) -> str:
    """เรียก OCR; serialize call ด้วย _reader_call_lock เพื่อความปลอดภัย"""
    reader = _get_reader()
    with _reader_call_lock:
        ocr_result = reader.readtext(frame_bgr, detail=0)
    return " ".join(ocr_result)


def _save_frame_if_needed(frame_bgr, source: str, roi_id: int | str | None, save: bool) -> None:
    if not save:
        return
    base_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
    roi_folder = f"{roi_id}" if roi_id is not None else "roi"
    save_dir = base_dir / "images" / roi_folder
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = str(save_dir / f"{ts}.jpg")
    threading.Thread(target=_save_image_async, args=(path, frame_bgr), daemon=True).start()


def _worker_loop_for_key(key: tuple[str, int | str | None]) -> None:
    """Worker ต่อ ROI: ดึงจากคิวของ key นี้เท่านั้น"""
    q = _queues[key]
    source, roi_id = key
    src_name = (source or "")
    stop_event = _stop_events.setdefault(src_name, threading.Event())
    logger.info(f"OCR worker started for source={source} roi_id={roi_id}")
    while True:
        if stop_event.is_set():
            break
        try:
            task: OcrTask = q.get(timeout=1.0)
        except Empty:
            continue

        try:
            if stop_event.is_set():
                break
            text = _run_ocr(task.frame_bgr)
            logger.info(
                f"source={task.source} roi_id={task.roi_id} OCR result: {text}"
                if task.roi_id is not None
                else f"source={task.source} OCR result: {text}"
            )
            _save_frame_if_needed(task.frame_bgr, task.source, task.roi_id, task.save)

            with _state_lock:
                _last_ocr_results[key] = text
                _last_ocr_times[key] = time.monotonic()
        except Exception as e:  # pragma: no cover
            logger.exception(f"OCR error for {key}: {e}")
            # ป้องกัน spam: อัพเดตเวลาแม้ผิดพลาด
            with _state_lock:
                _last_ocr_times[key] = time.monotonic()
        finally:
            try:
                del task
            except Exception:
                pass
            q.task_done()
    logger.info(f"OCR worker stopped for source={source} roi_id={roi_id}")


def _ensure_queue_and_worker(key: tuple[str, int | str | None], max_queue_size: int) -> None:
    """สร้างคิว/worker สำหรับ key ถ้ายังไม่มี"""
    with _qw_lock:
        if key not in _queues:
            _queues[key] = Queue(maxsize=max_queue_size)
        _stop_events.setdefault((key[0] or ""), threading.Event())
        if key not in _workers:
            t = threading.Thread(target=_worker_loop_for_key, args=(key,), daemon=False)
            t.start()
            _workers[key] = t


def _enqueue_latest_with_drop_oldest(q: Queue[OcrTask], item: OcrTask) -> None:
    """
    ถ้าคิวเต็ม: ดรอปงานเก่าสุด 1 ชิ้น แล้วใส่ของใหม่ (เพื่อเก็บเฟรมล่าสุด)
    """
    while True:
        try:
            q.put_nowait(item)
            return
        except Exception:
            try:
                q.get_nowait()
                q.task_done()
            except Empty:
                pass


def process(
    frame,
    roi_id=None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
    interval: float = 1.0,
    max_queue_size: int = 3,
):
    """
    ประมวลผลแบบ "คิวแยกต่อ ROI":
      - แต่ละ (source, roi_id) มีคิวและ worker ของตัวเอง
      - ถึงเวลา (>= interval) จึง enqueue งานเข้า **คิวของ ROI นั้น**
      - ถ้าคิวเต็ม จะดรอปงานเก่าสุดและเก็บงานล่าสุด (ลด memory growth)
      - คืนค่าผล OCR ล่าสุดของ ROI นั้น (ถ้ายังไม่เสร็จ จะเป็นค่าก่อนหน้า)
    """
    _configure_logger(source)

    # Block enqueue when stopping
    src_name = (source or "")
    ev = _stop_events.get(src_name)
    if ev is not None and ev.is_set():
        key = (src_name, roi_id)
        with _state_lock:
            return _last_ocr_results.get(key, "")

    # respect app-level stop flag
    try:
        import app  # type: ignore
        if cam_id is not None and app.is_stopping(str(cam_id)):
            key = (src_name, roi_id)
            with _state_lock:
                return _last_ocr_results.get(key, "")
    except Exception:
        pass

    if cam_id is not None:
        try:
            import app  # type: ignore
            app.save_roi_flags[cam_id] = False
        except Exception:  # pragma: no cover
            pass

    # แปลง PIL -> BGR ถ้าจำเป็น
    if isinstance(frame, Image.Image) and np is not None:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame

    key = (src_name, roi_id)
    now = time.monotonic()

    # สร้างคิวและเวิร์กเกอร์ของ ROI นี้ถ้ายังไม่มี
    _ensure_queue_and_worker(key, max_queue_size=max_queue_size)

    with _state_lock:
        last_time = _last_ocr_times.get(key)
        result_text = _last_ocr_results.get(key, "")

    due = (last_time is None) or ((now - last_time) >= interval)

    if due:
        task = OcrTask(source=key[0], roi_id=key[1], frame_bgr=frame_bgr.copy(), save=save)
        q = _queues[key]
        _enqueue_latest_with_drop_oldest(q, task)

    return result_text


def stop_workers_for_source(source: str) -> None:
    """
    หยุด worker ทั้งหมดของ source นี้:
    - set stop_event เพื่อให้ worker หลุดลูป
    - drain คิว, join เธรดทุกตัว
    - ล้าง state และปล่อย reader/logger เพื่อคืนหน่วยความจำ
    """
    src_name = (source or "")
    ev = _stop_events.setdefault(src_name, threading.Event())
    ev.set()

    # ค้นหาคีย์ทั้งหมดของ source นี้
    with _qw_lock:
        target_keys = [k for k in list(_queues.keys()) if (k[0] or "") == src_name]

    # drain คิว
    for k in target_keys:
        q = _queues.get(k)
        if q is not None:
            while True:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                finally:
                    try:
                        q.task_done()
                    except Exception:
                        pass
                try:
                    del item
                except Exception:
                    pass

    # join เธรด
    for k in target_keys:
        t = _workers.get(k)
        if t is not None:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

    # ล้าง dict และสถานะ
    with _qw_lock:
        for k in target_keys:
            _queues.pop(k, None)
            _workers.pop(k, None)
        with _state_lock:
            for k in target_keys:
                _last_ocr_results.pop(k, None)
                _last_ocr_times.pop(k, None)

    # ปล่อย reader
    global _reader
    try:
        _reader = None
    except Exception:
        pass

    # ปลด handler logger
    global _handler, _current_source
    if _handler:
        try:
            logger.removeHandler(_handler)
            _handler.close()
        except Exception:
            pass
        _handler = None
    _current_source = None

    # เคลียร์ event (เปิดทาง start ใหม่)
    _stop_events.pop(src_name, None)

    # เก็บขยะ
    try:
        import gc
        gc.collect()
    except Exception:
        pass


def stop_all_workers() -> None:
    """
    หยุด worker ทุก source ของ EasyOCR:
    - set stop_event ให้ทุก source
    - drain คิว, join เธรดทุกตัว
    - ล้าง state และปล่อย reader/logger
    - คืนหน่วยความจำ GPU (ถ้ามี)
    """
    try:
        with _qw_lock:
            sources = { (k[0] or "") for k in list(_queues.keys()) }
    except Exception:
        sources = set()

    for src in list(sources):
        _stop_events.setdefault(src, threading.Event()).set()

    try:
        with _qw_lock:
            target_keys = list(_queues.keys())
    except Exception:
        target_keys = []

    for k in target_keys:
        q = _queues.get(k)
        if q is not None:
            while True:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                finally:
                    try:
                        q.task_done()
                    except Exception:
                        pass
                try:
                    del item
                except Exception:
                    pass

    for k in target_keys:
        t = _workers.get(k)
        if t is not None:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

    with _qw_lock:
        for k in target_keys:
            _queues.pop(k, None)
            _workers.pop(k, None)
        with _state_lock:
            for k in target_keys:
                _last_ocr_results.pop(k, None)
                _last_ocr_times.pop(k, None)

    global _reader
    try: _reader = None
    except Exception: pass

    global _handler, _current_source
    if _handler:
        try:
            logger.removeHandler(_handler)
            _handler.close()
        except Exception:
            pass
        _handler = None
    _current_source = None

    _stop_events.clear()

    # คืน GPU mem ของ torch ถ้าเปิดใช้
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        import gc
        gc.collect()
    except Exception:
        pass
