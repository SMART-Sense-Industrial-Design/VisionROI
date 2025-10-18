from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any
from PIL import Image
import cv2
import logging
import os
import platform
from datetime import datetime
import threading
from pathlib import Path
import gc
from numbers import Number
from src.utils.logger import get_logger
from src.utils.image import save_image_async

from inference_modules.base_ocr import BaseOCR, np, Image, cv2

try:
    from rapidocr import RapidOCR as RapidOCRLib  # type: ignore
    from rapidocr.utils.typings import EngineType  # type: ignore
except Exception:
    RapidOCRLib = None  # type: ignore[assignment]
    EngineType = None  # type: ignore[assignment]

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
# RapidOCR ยังไม่ยืนยันความเป็น thread-safe ของ reader
# จึงต้องล็อกระหว่างการเรียกใช้งานจริงเพื่อให้ผลลัพธ์นิ่ง
_reader_infer_lock = threading.Lock()

last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


_BACKEND_OVERRIDE_KEYS = ("RAPIDOCR_BACKENDS", "RAPIDOCR_BACKEND")
_FORCE_PI_ENV = "VISIONROI_FORCE_PI5"


def _read_device_model() -> str:
    path = Path("/proc/device-tree/model")
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text.replace("\x00", "").strip()
    except Exception:
        return ""


def _is_raspberry_pi5() -> bool:
    override = os.getenv(_FORCE_PI_ENV)
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}

    model = _read_device_model().lower()
    if "raspberry" in model and "pi 5" in model:
        return True

    try:
        uname = platform.uname()
        hints = " ".join(
            [uname.system, uname.node, uname.release, uname.version, uname.machine, uname.processor]
        ).lower()
        if "raspberry" in hints and "pi" in hints and "5" in hints:
            return True
    except Exception:
        pass
    return False


def _gpu_execution_available() -> bool:
    if ort is None:
        return False
    try:
        providers = ort.get_available_providers()
        logger.info(f"[RapidOCR] onnxruntime available providers: {providers}")
        return any(p in {"CUDAExecutionProvider", "TensorrtExecutionProvider"} for p in providers)
    except Exception as exc:
        logger.debug(f"[RapidOCR] failed to query onnxruntime providers: {exc}")
        return False


def _parse_backend_overrides() -> list[str]:
    for key in _BACKEND_OVERRIDE_KEYS:
        value = os.getenv(key)
        if value:
            tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
            if tokens:
                logger.warning(f"[RapidOCR] using backend override from {key}: {tokens}")
                return tokens
    return []


def _candidate_backends() -> list[str]:
    overrides = _parse_backend_overrides()
    if overrides:
        return overrides

    candidates: list[str] = []
    if _is_raspberry_pi5():
        candidates.append("paddle")
    if _gpu_execution_available():
        candidates.append("onnxruntime-cuda")
    candidates.append("onnxruntime")
    return candidates


def _ensure_engine_enum(name: str) -> "EngineType":
    if EngineType is None:
        raise RuntimeError("rapidocr EngineType enum is unavailable")
    mapping = {
        "onnxruntime": EngineType.ONNXRUNTIME,
        "onnxruntime-cuda": EngineType.ONNXRUNTIME,
        "ort": EngineType.ONNXRUNTIME,
        "cpu": EngineType.ONNXRUNTIME,
        "openvino": EngineType.OPENVINO,
        "paddle": EngineType.PADDLE,
        "torch": EngineType.TORCH,
    }
    name_lower = name.lower()
    if name_lower not in mapping:
        raise ValueError(f"unknown RapidOCR backend '{name}'")
    return mapping[name_lower]


def _build_params_for_backend(backend: str) -> dict[str, Any]:
    engine_enum = _ensure_engine_enum(backend)
    params: dict[str, Any] = {
        "Global.use_det": False,
        "Global.use_cls": False,
        "Det.engine_type": engine_enum,
        "Cls.engine_type": engine_enum,
        "Rec.engine_type": engine_enum,
    }

    backend_lower = backend.lower()
    if backend_lower in {"onnxruntime", "onnxruntime-cuda", "ort", "cpu"}:
        use_cuda = backend_lower == "onnxruntime-cuda"
        params["EngineConfig.onnxruntime.use_cuda"] = use_cuda
        if use_cuda:
            device_id = os.getenv("ORT_CUDA_DEVICE_ID")
            if device_id is not None:
                try:
                    params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = int(device_id)
                except ValueError:
                    logger.debug(f"[RapidOCR] invalid ORT_CUDA_DEVICE_ID value: {device_id}")
    elif backend_lower == "paddle":
        params["EngineConfig.paddle.use_cuda"] = False
    return params


def _log_backend_summary(reader: Any, backend: str) -> None:
    try:
        cfg = getattr(reader, "cfg", None)
        if cfg is None:
            logger.warning(f"[RapidOCR] initialized backend='{backend}' (cfg unavailable)")
            return
        det_engine = getattr(getattr(cfg, "Det", None), "engine_type", None)
        rec_engine = getattr(getattr(cfg, "Rec", None), "engine_type", None)
        cls_engine = getattr(getattr(cfg, "Cls", None), "engine_type", None)
        logger.warning(
            "[RapidOCR] initialized backend='%s' det=%s rec=%s cls=%s",
            backend,
            getattr(det_engine, "value", det_engine),
            getattr(rec_engine, "value", rec_engine),
            getattr(cls_engine, "value", cls_engine),
        )
    except Exception as exc:
        logger.debug(f"[RapidOCR] unable to inspect backend config: {exc}")


def _warmup_reader(reader: Any) -> None:
    try:
        import numpy as _np

        dummy = _np.zeros((8, 8, 3), dtype=_np.uint8)
        _run_reader(reader, dummy)
        logger.info("[RapidOCR] warm-up call executed successfully")
    except Exception as exc:
        logger.debug(f"[RapidOCR] warm-up call failed: {exc}")


def _new_reader_instance() -> Any:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr library is not installed")

    candidates = _candidate_backends()
    errors: list[str] = []
    logger.warning(f"[RapidOCR] backend candidates: {candidates}")

    for backend in candidates:
        try:
            params = _build_params_for_backend(backend)
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            logger.warning(f"[RapidOCR] skipping backend {backend}: {exc}")
            continue

        try:
            reader = RapidOCRLib(params=params)  # type: ignore[call-arg]
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            logger.warning(f"[RapidOCR] backend {backend} failed to initialize: {exc}")
            continue

        _log_backend_summary(reader, backend)
        _warmup_reader(reader)
        return reader

    error_msg = "; ".join(errors) if errors else "no backend candidates"
    raise RuntimeError(f"RapidOCR initialization failed: {error_msg}")


def _get_global_reader():
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr library is not installed")
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = _new_reader_instance()
        return _reader


# ======================
# Normalise/Extract text
# ======================
def _normalise_reader_output(result: Any) -> Any:
    if (
        isinstance(result, (list, tuple))
        and len(result) == 2
        and isinstance(result[0], (list, tuple))
        and not isinstance(result[1], (list, tuple, dict))
    ):
        return result[0]
    return result


def _result_structure_is_empty(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> bool:
    if _depth > 5:
        return False
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Number):
        return True
    if _seen is None:
        _seen = set()
    try:
        obj_id = id(value)
    except Exception:
        obj_id = None
    if obj_id is not None:
        if obj_id in _seen:
            return True
        _seen.add(obj_id)
    if isinstance(value, Mapping):
        if not value:
            return True
        for key in ("text", "texts", "txts"):
            if key in value and not _result_structure_is_empty(
                value[key], _depth=_depth + 1, _seen=_seen
            ):
                return False
        return all(
            _result_structure_is_empty(v, _depth=_depth + 1, _seen=_seen)
            for v in value.values()
        )
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        empty = True
        for item in value:
            empty = False
            if not _result_structure_is_empty(
                item, _depth=_depth + 1, _seen=_seen
            ):
                return False
        return empty
    return False


def _log_empty_text_warning(
    logger_obj: logging.Logger,
    module_name: str,
    roi_id: Any,
    reason: str,
    details: str | None = None,
) -> None:
    prefix = f"roi_id={roi_id} " if roi_id is not None else ""
    message = f"{prefix}{module_name} OCR returned empty text (reason={reason})"
    if details:
        message = f"{message}: {details}"
    logger_obj.warning(message)


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
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        queue.extend(item for item in value if item is not None)
        return True
    if isinstance(value, Number):
        return True
    pieces.append(str(value))
    return True


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
                if len(current) == 2:
                    first, second = current
                    if isinstance(first, str) and isinstance(second, Number):
                        if first:
                            pieces.append(first)
                        continue
                str_items = [item for item in current if isinstance(item, str) and item]
                if str_items:
                    pieces.extend(str_items)
                else:
                    queue.extend(item for item in current if item is not None)
            else:
                if isinstance(current, Number):
                    continue
                pieces.append(str(current))
    return " ".join(pieces)


# ===========
# OCR runners
# ===========
def _prepare_frame_for_reader(frame):
    """แปลงภาพให้พร้อมสำหรับ RapidOCR reader."""
    if np is not None and hasattr(frame, "flags") and hasattr(frame, "ndim"):
        # RapidOCR คาดหวัง array ที่ contiguous เสมอ
        try:
            needs_copy = False
            base = getattr(frame, "base", None)
            if base is not None:
                # เมื่อ ROI เป็น view ของเฟรมต้นฉบับ buffer จะถูกเปลี่ยนตามเฟรมใหม่
                # ทำให้ผล OCR แกว่ง ดังนั้นบังคับให้ทำสำเนาเต็มก่อน
                try:
                    if base is frame:
                        share_memory = False
                    else:
                        share_memory = bool(np.shares_memory(frame, base))
                except ValueError:
                    # shares_memory บางครั้งโยน ValueError หาก shape ไม่สอดคล้อง
                    share_memory = True
                except Exception:
                    share_memory = True
                if share_memory:
                    needs_copy = True

            # numpy array ที่ไม่มี owndata (เช่น view ที่ base ไม่ได้เป็น ndarray)
            # ก็มีพฤติกรรมแกว่งเหมือนกัน ต้อง copy เช่นกัน
            if not needs_copy and hasattr(frame.flags, "owndata"):
                try:
                    if not frame.flags.owndata:
                        needs_copy = True
                except AttributeError:
                    needs_copy = True

            if needs_copy:
                frame = frame.copy()

            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
        except AttributeError:
            pass
    return frame


def _run_reader(reader, frame):
    """เรียก RapidOCR แบบบังคับใช้เฉพาะการ Rec."""

    kwargs = {
        "use_det": False,
        "use_cls": False,
        "use_rec": True,
    }

    # ป้องกัน race condition ระหว่างหลายเธรด
    with _reader_infer_lock:
        try:
            return reader(frame, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword" not in message and "positional arguments" not in message:
                raise
            return reader(frame)


def _run_ocr_async(frame, roi_id, save, source) -> str:
    empty_reason: str | None = None
    empty_details: str | None = None
    raw_result: Any = None
    text = ""
    try:
        reader = _get_global_reader()
    except Exception as e:
        logger.exception(f"roi_id={roi_id} {MODULE_NAME} OCR error: {e}")
        empty_reason = "reader_init_failed"
        empty_details = f"{type(e).__name__}: {e}"
        reader = None

    if reader is not None:
        try:
            frame = _prepare_frame_for_reader(frame)
            raw_result = _normalise_reader_output(_run_reader(reader, frame))
            text = _extract_text(raw_result)

            logger.info(
                f"roi_id={roi_id} {MODULE_NAME} OCR result: {text}"
                if roi_id is not None
                else f"{MODULE_NAME} OCR result: {text}"
            )
            with _last_ocr_lock:
                last_ocr_results[roi_id] = text

            if not text and not _result_structure_is_empty(raw_result):
                empty_reason = "unexpected_payload"
                empty_details = f"raw_type={type(raw_result).__name__}"
        except Exception as e:
            logger.exception(f"roi_id={roi_id} {MODULE_NAME} OCR error: {e}")
            empty_reason = "runtime_exception"
            empty_details = f"{type(e).__name__}: {e}"
            text = ""

    if not text and empty_reason:
        _log_empty_text_warning(logger, MODULE_NAME, roi_id, empty_reason, empty_details)

    if save:
        base_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
        roi_folder = f"{roi_id}" if roi_id is not None else "roi"
        save_dir = base_dir / "images" / roi_folder
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        path = save_dir / filename
        save_image_async(str(path), frame)

    return text


class RapidOCR(BaseOCR):
    MODULE_NAME = "rapid_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._reader_lock = threading.Lock()
        self._reader = None

    def _get_reader(self):
        if RapidOCRLib is None:
            raise RuntimeError("rapidocr library is not installed")
        with self._reader_lock:
            if self._reader is None:
                self._reader = _new_reader_instance()
            return self._reader

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        text = ""
        empty_reason: str | None = None
        empty_details: str | None = None
        raw_result: Any = None
        try:
            reader = self._get_reader()
        except Exception as e:
            self.logger.exception(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR init error: {e}"
            )
            reader = None
            empty_reason = "reader_init_failed"
            empty_details = f"{type(e).__name__}: {e}"

        if reader is not None:
            try:
                frame = _prepare_frame_for_reader(frame)
                raw_result = _normalise_reader_output(_run_reader(reader, frame))
                text = _extract_text(raw_result)
            except Exception as e:
                self.logger.exception(
                    f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {e}"
                )
                empty_reason = "runtime_exception"
                empty_details = f"{type(e).__name__}: {e}"

        if text:
            self.logger.info(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {text}"
                if roi_id is not None
                else f"{self.MODULE_NAME} OCR result: {text}"
            )
        elif empty_reason is None and not _result_structure_is_empty(raw_result):
            empty_reason = "unexpected_payload"
            empty_details = f"raw_type={type(raw_result).__name__}"

        if save:
            self._save_image(frame, roi_id, source)

        if not text and empty_reason:
            _log_empty_text_warning(
                self.logger, self.MODULE_NAME, roi_id, empty_reason, empty_details
            )

        return text

    def _cleanup_extra(self) -> None:
        global _reader
        with _reader_lock:
            _reader = None
        with _last_ocr_lock:
            last_ocr_times.clear()
            last_ocr_results.clear()


# ===========
# Public API
# ===========
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