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
from inspect import signature, Parameter

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None

# ---- ใช้ rapidocr-onnxruntime โดยตรง ----
RapidOCR = None
_rapidocr_err: str | None = None
try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    RapidOCR = _RapidOCR  # type: ignore[assignment]
except Exception as e:  # pragma: no cover - fallback when rapidocr_onnxruntime missing
    _rapidocr_err = f"rapidocr-onnxruntime import error: {e}"

# onnxruntime สำหรับตรวจ EP และแสดงผล provider ที่มีอยู่
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler: TimedRotatingFileHandler | None = None
_current_source: str | None = None
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_reader = None
_reader_lock = threading.Lock()

# คิวและตัวจัดการ worker สำหรับแยกงาน OCR ตาม ROI
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


def _available_providers() -> list[str]:
    if ort is None:  # pragma: no cover
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover
        return []


def _parse_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _build_providers_from_env() -> tuple[list, list]:
    """
    คืนค่า (providers, provider_options) ตามสิ่งที่เครื่องรองรับ + ค่า ENV
    ลำดับดีฟอลต์: TensorRT -> CUDA -> CPU
    สามารถ override ด้วย RAPIDOCR_PROVIDERS (คอมม่า-คั่น)
    """
    avail = _available_providers()
    if avail:
        logger.info(f"ONNX Runtime available providers: {avail}")
    else:
        logger.info("ONNX Runtime providers not available or ORT not installed; will fall back to defaults.")

    # อ่าน override จาก ENV (ถ้ามี)
    override = os.getenv("RAPIDOCR_PROVIDERS")
    if override:
        desired = [x.strip() for x in override.split(",") if x.strip()]
    else:
        desired = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    # กรองเฉพาะที่มีอยู่จริง (แต่คง CPUExecutionProvider ไว้เป็น fallback แม้ ORT รายงานไม่ครบ)
    providers: list = []
    provider_options: list = []
    for p in desired:
        if p == "CPUExecutionProvider":
            providers.append(p)
            provider_options.append({})
            continue
        if (not avail) or (p in avail):
            if p == "TensorrtExecutionProvider":
                # ตั้งค่า TensorRT EP จาก ENV
                trt_fp16 = _parse_env_bool("RAPIDOCR_TRT_FP16", True)
                # หน่วย MB -> bytes
                trt_ws_mb = int(os.getenv("RAPIDOCR_TRT_WORKSPACE", "2048"))
                trt_ws = int(trt_ws_mb) * 1024 * 1024
                providers.append(p)
                provider_options.append({
                    "trt_fp16_enable": trt_fp16,
                    "trt_max_workspace_size": trt_ws,
                })
            else:
                providers.append(p)
                provider_options.append({})
        else:
            logger.info(f"Skip provider '{p}' (not available on this system).")

    # อย่างน้อยต้องมี CPU
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
        provider_options.append({})

    return providers, provider_options


def _class_accepts_params(cls, *param_names: str) -> dict[str, bool]:
    """
    ตรวจว่า __init__ ของ RapidOCR รองรับพารามิเตอร์ที่เราจะส่งหรือไม่
    เพื่อความเข้ากันได้กับเวอร์ชันต่าง ๆ
    """
    try:
        sig = signature(cls)
    except Exception:
        try:
            sig = signature(cls.__init__)  # type: ignore[attr-defined]
        except Exception:
            return {name: False for name in param_names}

    params = sig.parameters
    result = {}
    for name in param_names:
        result[name] = (name in params) or any(
            p.kind == Parameter.VAR_KEYWORD for p in params.values()
        )
    return result


def _get_reader():
    """สร้างและคืนค่า RapidOCR แบบ singleton (เลือก EP: TensorRT → CUDA → CPU)"""
    if RapidOCR is None:
        # ยกข้อความ error จากตอน import มาแสดงให้ชัด
        raise RuntimeError(_rapidocr_err or "rapidocr-onnxruntime library is not installed")

    global _reader
    with _reader_lock:
        if _reader is not None:
            return _reader

        providers, provider_options = _build_providers_from_env()
        accepts = _class_accepts_params(RapidOCR, "providers", "provider_options", "use_cuda")

        kwargs = {}
        if accepts.get("providers", False):
            kwargs["providers"] = providers
        if accepts.get("provider_options", False):
            kwargs["provider_options"] = provider_options

        # fallback: บางเวอร์ชันอาจมีเพียง use_cuda (ไม่รองรับ providers)
        if not kwargs and accepts.get("use_cuda", False):
            kwargs["use_cuda"] = ("CUDAExecutionProvider" in providers or
                                  "TensorrtExecutionProvider" in providers)

        try:
            _reader = RapidOCR(**kwargs)  # type: ignore[misc]
            chosen = kwargs.get("providers") or (["CUDA" if kwargs.get("use_cuda") else "CPU"])
            logger.info(f"RapidOCR initialized with providers: {chosen}")
        except TypeError as e:
            # ถ้าเวอร์ชันไม่รับ kwargs พวกนี้จริง ๆ ให้ลองเรียกแบบว่าง
            logger.warning(f"RapidOCR init with providers failed ({e}); falling back to default init().")
            _reader = RapidOCR()  # type: ignore[call-arg]
        except Exception as e:
            logger.exception(f"RapidOCR init error: {e}")
            raise

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


def _extract_texts(ocr_result) -> str:
    """รองรับฟอร์แมตผลลัพธ์ RapidOCR หลายแบบ -> รวมเป็นสตริงเดียว"""
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
    return " ".join(text_items)


def _run_ocr_async(frame, roi_id, save, source) -> None:
    """ประมวลผล OCR และบันทึกรูปในเธรดแยก"""
    try:
        reader = _get_reader()
        result = reader(frame)

        # RapidOCR บางรุ่นคืน (result, time) ให้เลือกเฉพาะ result
        if (
            isinstance(result, (list, tuple))
            and len(result) == 2
            and isinstance(result[0], (list, tuple))
            and not isinstance(result[1], (list, tuple, dict))
        ):
            ocr_result = result[0][1]
        else:
            ocr_result = result

        text = _extract_texts(ocr_result)

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
    interval: float = 3.0,
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
    """รีเซ็ตสถานะและคืนทรัพยากรที่ใช้โดยโมดูล OCR"""
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
        except Exception:  # pragma: no cover
            logger.exception("Failed to close log handler")
        _handler = None
    _current_source = None

    gc.collect()
