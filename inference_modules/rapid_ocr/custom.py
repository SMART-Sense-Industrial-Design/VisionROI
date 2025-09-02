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
from typing import Optional, Tuple, Dict, Any

try:
    # *** โค้ดเดิมของคุณอาจ import RapidOCR ตรงนี้ ***
    from rapidocr import RapidOCR  # type: ignore
except Exception:  # pragma: no cover - fallback when rapidocr missing
    RapidOCR = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler: TimedRotatingFileHandler | None = None
_current_source: str | None = None
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

# ===== RapidOCR singleton + mutex สำหรับเรียกใช้งาน =====
_reader = None
_reader_lock = threading.Lock()        # create reader
_reader_call_lock = threading.Lock()   # serialize reader(...)

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

# simplified cache for tests/external access keyed only by roi_id
last_ocr_results: dict[int | str | None, str] = {}

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


def _get_reader():
    """สร้างและคืนค่า RapidOCR แบบ singleton"""
    if RapidOCR is None:
        raise RuntimeError("rapidocr library is not installed")
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = RapidOCR()
        return _reader


def _save_image_async(path, image) -> None:
    """บันทึกรูปภาพแบบแยกเธรด"""
    try:
        with _imwrite_lock:
            cv2.imwrite(path, image)
    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to save image {path}: {e}")


def _extract_text_from_rapidocr_output(result_obj) -> str:
    """
    รองรับรูปแบบผลลัพธ์ RapidOCR หลายแบบ:
    - [(box, text, score), ...]
    - ([ (box, text, score), ... ], other-stuff)
    - หรือ list/tuple/dict ที่ index 1 เป็นข้อความ
    """
    if (
        isinstance(result_obj, (list, tuple))
        and len(result_obj) >= 2
        and isinstance(result_obj[0], (list, tuple))
        and not isinstance(result_obj[1], (list, tuple, dict))
    ):
        ocr_result = result_obj[0]
    else:
        ocr_result = result_obj

    texts: list[str] = []
    if isinstance(ocr_result, (list, tuple)):
        for res in ocr_result:
            # คาดหวังรูปแบบ [box, text, score] หรือ tuple ที่ index 1 เป็นข้อความ
            if isinstance(res, (list, tuple)) and len(res) > 1:
                texts.append(str(res[1]))
            elif isinstance(res, dict) and "text" in res:
                texts.append(str(res["text"]))
    elif isinstance(ocr_result, dict) and "text" in ocr_result:
        texts.append(str(ocr_result["text"]))
    else:
        try:
            return " ".join(map(str, ocr_result))
        except Exception:
            return str(ocr_result)
    return " ".join(texts)


def _run_ocr(frame_bgr) -> str:
    """เรียก RapidOCR; serialize call ด้วย _reader_call_lock เพื่อความปลอดภัย"""
    reader = _get_reader()
    with _reader_call_lock:
        result_obj = reader(frame_bgr)
    return _extract_text_from_rapidocr_output(result_obj)


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
    """Worker ต่อ ROI: ดึงจากคิวของ key นี้เท่านั้น และหยุดได้ด้วย stop_event ของ source"""
    q = _queues[key]
    source, roi_id = key
    src_name = (source or "")
    stop_event = _stop_events.setdefault(src_name, threading.Event())
    logger.info(f"OCR worker started for source={src_name} roi_id={roi_id}")
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
                if task.roi_id is not None else f"source={task.source} OCR result: {text}"
            )
            _save_frame_if_needed(task.frame_bgr, task.source, task.roi_id, task.save)

            with _state_lock:
                _last_ocr_results[key] = text
                _last_ocr_times[key] = time.monotonic()
        except Exception as e:  # pragma: no cover
            logger.exception(f"OCR error for {key}: {e}")
            with _state_lock:
                _last_ocr_times[key] = time.monotonic()
        finally:
            try:
                del task  # ลด reference ของเฟรมจากคิวให้ไวขึ้น
            except Exception:
                pass
            q.task_done()
    logger.info(f"OCR worker stopped for source={src_name} roi_id={roi_id}")


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
    interval: float = 3.0,
    max_queue_size: int = 3,
):
    """
    ประมวลผล ROI + RapidOCR แบบ "คิวแยกต่อ ROI":
      - แต่ละ (source, roi_id) มีคิวและ worker ของตัวเอง
      - ถึงเวลา (>= interval ต่อ ROI) จึง enqueue งานใหม่
      - ถ้าคิวเต็ม จะดรอปงานเก่าสุด แล้วเก็บงานใหม่สุด (ลด memory growth)
      - คืนค่าข้อความล่าสุดของ ROI นั้น (ถ้ายังไม่เสร็จจะเป็นค่าก่อนหน้า)
    พารามิเตอร์:
      - interval: เวลาขั้นต่ำต่อ ROI ก่อนจะ OCR ใหม่ (ค่าเริ่มต้น 3.0 วินาที)
      - max_queue_size: ขนาดสูงสุดต่อคิวของ ROI แต่ละตัว (ค่าเริ่มต้น 4)
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
    หยุด worker ทุก source:
    - set stop_event ให้ทุก source
    - drain คิว, join เธรดทุกตัว
    - ล้าง state และปล่อย reader/logger
    ใช้ตอน SIGINT/SIGTERM เพื่อไม่ให้ traceback จาก threading._shutdown
    """
    # 1) สร้างรายการ source ทั้งหมดจาก keys ปัจจุบัน
    try:
        with _qw_lock:
            sources = { (k[0] or "") for k in list(_queues.keys()) }
    except Exception:
        sources = set()

    # 2) set stop_event ให้ทุก source
    for src in list(sources):
        _stop_events.setdefault(src, threading.Event()).set()

    # 3) รวม key ทั้งหมดอีกครั้ง (กัน race)
    try:
        with _qw_lock:
            target_keys = list(_queues.keys())
    except Exception:
        target_keys = []

    # 4) drain คิวทุก key
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

    # 5) join เธรดทุก key
    for k in target_keys:
        t = _workers.get(k)
        if t is not None:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

    # 6) ล้างโครงสร้างทั้งหมด
    with _qw_lock:
        for k in target_keys:
            _queues.pop(k, None)
            _workers.pop(k, None)
        with _state_lock:
            for k in target_keys:
                _last_ocr_results.pop(k, None)
                _last_ocr_times.pop(k, None)

    # 7) ปล่อย reader + logger
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

    # 8) เคลียร์ทุก stop_event และ GC
    _stop_events.clear()
    try:
        import gc
        gc.collect()
    except Exception:
        pass
