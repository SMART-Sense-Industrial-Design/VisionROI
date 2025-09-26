from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from typing import Any
from PIL import Image
import cv2
import logging
import os
from datetime import datetime
import threading
from pathlib import Path
import gc
from src.utils.logger import get_logger
from src.utils.image import save_image_async

from inference_modules.base_ocr import BaseOCR, np, Image, cv2

# ------------------------------
# RapidOCR (ONNXRuntime) backend
# ------------------------------
try:
    # เปลี่ยนมาใช้ rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR as RapidOCRLib  # type: ignore
except Exception:  # pragma: no cover - fallback when rapidocr_onnxruntime missing
    RapidOCRLib = None  # type: ignore[assignment]

# onnxruntime ใช้สำหรับตรวจสอบ/เลือก EP
try:
    import onnxruntime as ort
except Exception:
    ort = None  # type: ignore

MODULE_NAME = "rapid_ocr"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_reader = None
_reader_lock = threading.Lock()

# ค่าดีฟอลต์ของ ORT/TensorRT (สามารถ override ได้ด้วย env ภายนอก)
def _ensure_default_ort_envs() -> None:
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_PATH", str((_data_sources_root / "trt_cache").resolve()))
    # เปิด FP16 บน Jetson (หากต้องการปิด ให้ตั้งค่าเป็น "0" ภายนอก)
    os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")


def _choose_providers() -> list[str]:
    """
    เลือก EP ตามความพร้อมของระบบโดยเรียงลำดับความเร็ว:
    TensorRT -> CUDA -> CPU
    สามารถ override ด้วย env: RAPIDOCR_ORT_PROVIDERS="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
    """
    env_val = os.getenv("RAPIDOCR_ORT_PROVIDERS")
    if env_val:
        providers = [p.strip() for p in env_val.split(",") if p.strip()]
        return providers

    available = []
    if ort is not None:
        try:
            available = ort.get_available_providers()
            print(f"onnxruntime available providers: {available}")
        except Exception:
            available = []

    preferred = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    if not available:
        return preferred

    chosen = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _new_reader_instance() -> Any:
    """สร้างอินสแตนซ์ RapidOCRLib (onnxruntime backend) พร้อม providers"""
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime is not installed")

    _ensure_default_ort_envs()
    providers = _choose_providers()
    logger.info(f"[RapidOCR-ORT] requested providers={providers}")

    try:
        reader = RapidOCRLib(providers=providers)  # type: ignore[call-arg]
        logger.info("[RapidOCR-ORT] RapidOCRLib accepted 'providers' argument ✅")
    except TypeError:
        reader = RapidOCRLib()  # type: ignore[call-arg]
        logger.warning("[RapidOCR-ORT] RapidOCRLib did NOT accept 'providers' argument ❌ (fallback to default)")

    # แสดง providers ที่ถูกใช้จริงในแต่ละ session ถ้าเข้าถึงได้
    try:
        if hasattr(reader, "det_sess"):
            _det_prov = getattr(reader.det_sess, "get_providers", lambda: [])()
            logger.info(f"[RapidOCR-ORT] det_sess providers={_det_prov}")
        if hasattr(reader, "rec_sess"):
            _rec_prov = getattr(reader.rec_sess, "get_providers", lambda: [])()
            logger.info(f"[RapidOCR-ORT] rec_sess providers={_rec_prov}")
        if hasattr(reader, "cls_sess"):
            _cls_prov = getattr(reader.cls_sess, "get_providers", lambda: [])()
            logger.info(f"[RapidOCR-ORT] cls_sess providers={_cls_prov}")
    except Exception as e:
        logger.debug(f"[RapidOCR-ORT] cannot inspect actual providers: {e}")

    return reader


def _get_reader():
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime library is not installed")
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = _new_reader_instance()
        return _reader


class RapidOCR(BaseOCR):
    MODULE_NAME = "rapid_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._reader_lock = threading.Lock()
        self._reader = None

    def _get_reader(self):
        if RapidOCRLib is None:
            raise RuntimeError("rapidocr_onnxruntime library is not installed")
        with self._reader_lock:
            if self._reader is None:
                self._reader = _new_reader_instance()
            return self._reader

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        text = ""
        if RapidOCRLib is not None:
            try:
                reader = self._get_reader()
                result = _normalise_reader_output(reader(frame))
                text = _extract_text(result)
            except Exception as e:
                self.logger.exception(
                    f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {e}"
                )

        if text:
            self.logger.info(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {text}"
                if roi_id is not None
                else f"{self.MODULE_NAME} OCR result: {text}"
            )

        if save:
            self._save_image(frame, roi_id, source)

        return text

    def _cleanup_extra(self) -> None:
        global _reader
        with _reader_lock:
            _reader = None
        with _last_ocr_lock:
            last_ocr_times.clear()
            last_ocr_results.clear()


last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


def _normalise_reader_output(result: Any) -> Any:
    if (
        isinstance(result, (list, tuple))
        and len(result) == 2
        and isinstance(result[0], (list, tuple))
        and not isinstance(result[1], (list, tuple, dict))
    ):
        return result[0]
    return result


def _extract_text(ocr_result: Any) -> str:
    if ocr_result is None:
        return ""

    pieces: list[str] = []
    queue: deque[Any] = deque([ocr_result])

    while queue:
        current = queue.popleft()
        if current is None:
            continue

        if isinstance(current, str):
            if current:
                pieces.append(current)
            continue

        if isinstance(current, dict):
            text_value = current.get("text")
            if text_value is not None:
                _append_text_value(text_value, pieces, queue)
            continue

        text_attr = getattr(current, "text", None)
        if text_attr is not None:
            handled = _append_text_value(text_attr, pieces, queue)
            if handled:
                continue

        for attr_name in ("texts", "txts"):
            if hasattr(current, attr_name):
                attr_value = getattr(current, attr_name)
                if attr_value is not None:
                    _append_text_value(attr_value, pieces, queue)
                break
        else:
            if isinstance(current, (list, tuple)):
                if len(current) > 1 and not isinstance(current[1], (list, tuple, dict)):
                    text_candidate = current[1]
                    if text_candidate is not None:
                        pieces.append(str(text_candidate))
                else:
                    queue.extend(current)
            else:
                pieces.append(str(current))

    return " ".join(pieces)


def _append_text_value(value: Any, pieces: list[str], queue: deque[Any]) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        if value:
            pieces.append(value)
        return True

    if isinstance(value, dict):
        queue.append(value)
        return True

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        queue.extend(item for item in value if item is not None)
        return True

    pieces.append(str(value))
    return True


def _run_ocr_async(frame, roi_id, save, source) -> str:
    try:
        reader = _get_reader()
        result = _normalise_reader_output(reader(frame))
        text = _extract_text(result)

        logger.info(
            f"roi_id={roi_id} {MODULE_NAME} OCR result: {text}"
            if roi_id is not None
            else f"{MODULE_NAME} OCR result: {text}"
        )
        with _last_ocr_lock:
            last_ocr_results[roi_id] = text
    except Exception as e:
        logger.exception(f"roi_id={roi_id} {MODULE_NAME} OCR error: {e}")
        text = ""

    if save:
        base_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
        roi_folder = f"{roi_id}" if roi_id is not None else "roi"
        save_dir = base_dir / "images" / roi_folder
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        path = save_dir / filename
        save_image_async(str(path), frame)

    return text


def process(
    frame,
    roi_id=None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
    interval: float = 3.0,
):
    logger = get_logger(MODULE_NAME, source)

    if cam_id is not None:
        try:
            import app  # type: ignore
            app.save_roi_flags[cam_id] = False
        except Exception:
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


def cleanup() -> None:
    global _reader
    with _reader_lock:
        _reader = None
    with _last_ocr_lock:
        last_ocr_times.clear()
        last_ocr_results.clear()
    gc.collect()
