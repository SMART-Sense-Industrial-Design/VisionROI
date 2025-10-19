from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from queue import Empty, LifoQueue
from types import SimpleNamespace
from typing import Any
from PIL import Image
import cv2
import importlib.util
import logging
import os
from datetime import datetime
import threading
from pathlib import Path
import gc
from numbers import Number

from omegaconf import OmegaConf

from src.utils.logger import get_logger
from src.utils.image import save_image_async

from inference_modules.base_ocr import BaseOCR, np, Image, cv2

try:
    from rapidocr import TextRecognizer as RapidOCRLib  # type: ignore
    from rapidocr.utils.typings import (  # type: ignore
        EngineType,
        LangRec,
        ModelType,
        OCRVersion,
        TaskType,
    )
except Exception:
    RapidOCRLib = None  # type: ignore[assignment]
    EngineType = None  # type: ignore[assignment]
    LangRec = None  # type: ignore[assignment]
    ModelType = None  # type: ignore[assignment]
    OCRVersion = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]

MODULE_NAME = "rapid_ocr"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

_DEFAULT_TEXT_RECOGNIZER_CONFIG: dict[str, Any] | None = None

def _resolve_max_reader_workers() -> int:
    env_value = (os.getenv("RAPIDOCR_MAX_READERS") or "").strip()
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
            logger.warning(
                "[RapidOCR] Ignoring non-positive RAPIDOCR_MAX_READERS=%s", env_value
            )
        except ValueError:
            logger.warning(
                "[RapidOCR] Ignoring invalid RAPIDOCR_MAX_READERS=%s", env_value
            )

    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


_MAX_READER_WORKERS = _resolve_max_reader_workers()
_reader_pool: LifoQueue[Any] = LifoQueue()
_reader_pool_lock = threading.Lock()
_reader_creation_executor: ThreadPoolExecutor | None = None
_reader_created = 0

logger.info(
    "[RapidOCR] Reader pool max workers configured to %d", _MAX_READER_WORKERS
)

last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


def _ensure_reader_executor_locked() -> ThreadPoolExecutor:
    global _reader_creation_executor
    if _reader_creation_executor is None:
        _reader_creation_executor = ThreadPoolExecutor(
            max_workers=_MAX_READER_WORKERS,
            thread_name_prefix="rapidocr-reader",
        )
    return _reader_creation_executor


def _acquire_reader_from_pool() -> Any:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr library is not installed")

    try:
        return _reader_pool.get_nowait()
    except Empty:
        pass

    create_executor: ThreadPoolExecutor | None = None
    should_create = False
    global _reader_created
    with _reader_pool_lock:
        if _reader_created < _MAX_READER_WORKERS:
            _reader_created += 1
            should_create = True
            create_executor = _ensure_reader_executor_locked()

    if should_create and create_executor is not None:
        future = create_executor.submit(_new_reader_instance)
        try:
            reader = future.result()
        except Exception:
            with _reader_pool_lock:
                _reader_created -= 1
            raise
        return reader

    # หากไม่มี reader ว่าง ให้รอจนกว่าจะถูกคืนกลับมา
    return _reader_pool.get()


def _release_reader_to_pool(reader: Any) -> None:
    if reader is None:
        return
    _reader_pool.put(reader)


def _reset_reader_pool() -> None:
    global _reader_created, _reader_creation_executor
    while True:
        try:
            reader = _reader_pool.get_nowait()
        except Empty:
            break
        # ปล่อยอ้างอิงเพื่อให้ GC จัดการต่อ
        del reader
    _reader_created = 0
    if _reader_creation_executor is not None:
        _reader_creation_executor.shutdown(wait=False, cancel_futures=True)
        _reader_creation_executor = None


def _load_default_text_recognizer_config() -> dict[str, Any]:
    global _DEFAULT_TEXT_RECOGNIZER_CONFIG

    if _DEFAULT_TEXT_RECOGNIZER_CONFIG is not None:
        return _DEFAULT_TEXT_RECOGNIZER_CONFIG

    missing = [
        name
        for name, value in (
            ("EngineType", EngineType),
            ("LangRec", LangRec),
            ("ModelType", ModelType),
            ("OCRVersion", OCRVersion),
            ("TaskType", TaskType),
        )
        if value is None
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"rapidocr enums unavailable: {missing_str}")

    spec = importlib.util.find_spec("rapidocr")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("rapidocr package metadata is unavailable")

    config_path = Path(spec.submodule_search_locations[0]) / "config.yaml"

    try:
        cfg = OmegaConf.load(str(config_path))
    except FileNotFoundError as exc:
        raise RuntimeError(f"rapidocr config not found at {config_path}") from exc
    except Exception as exc:  # pragma: no cover - defensive against OmegaConf errors
        raise RuntimeError(f"failed to load rapidocr config: {exc}") from exc

    rec_cfg = cfg.Rec
    try:
        engine_type = EngineType(rec_cfg.engine_type)
        model_type = ModelType(rec_cfg.model_type)
        ocr_version = OCRVersion(rec_cfg.ocr_version)
        task_type = TaskType(rec_cfg.task_type)
        lang_type = LangRec(rec_cfg.lang_type)
    except Exception as exc:
        raise RuntimeError(f"invalid enum values in rapidocr config: {exc}") from exc

    rec_cfg.engine_type = engine_type
    rec_cfg.model_type = model_type
    rec_cfg.ocr_version = ocr_version
    rec_cfg.task_type = task_type
    rec_cfg.lang_type = lang_type
    rec_cfg.engine_cfg = cfg.EngineConfig[engine_type.value]
    rec_cfg.font_path = cfg.Global.font_path

    _DEFAULT_TEXT_RECOGNIZER_CONFIG = OmegaConf.to_container(
        rec_cfg, resolve=True
    )
    return _DEFAULT_TEXT_RECOGNIZER_CONFIG


def _build_text_recognizer_params() -> Any:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr library is not installed")

    base_config = _load_default_text_recognizer_config()
    params = deepcopy(base_config)
    engine_cfg = params.get("engine_cfg")
    if engine_cfg is not None:
        params["engine_cfg"] = deepcopy(engine_cfg)

    model_path = (os.getenv("RAPIDOCR_REC_MODEL_PATH") or "").strip()
    if model_path:
        params["model_path"] = model_path

    keys_path = (os.getenv("RAPIDOCR_REC_KEYS_PATH") or "").strip()
    if keys_path:
        params["rec_keys_path"] = keys_path

    return OmegaConf.create(params)


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

    params = _build_text_recognizer_params()
    reader = RapidOCRLib(params=params)  # type: ignore[call-arg]
    _warmup_reader(reader)
    return reader


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
    """เรียก TextRecognizer ของ RapidOCR สำหรับงาน Rec เท่านั้น."""

    request = SimpleNamespace(img=frame, return_word_box=False)

    try:
        return reader(request)
    except TypeError as exc:
        message = str(exc)
        if "img" not in message and "TextRecInput" not in message:
            raise
        return reader(frame)


def _run_ocr_async(frame, roi_id, save, source) -> str:
    empty_reason: str | None = None
    empty_details: str | None = None
    raw_result: Any = None
    text = ""
    reader = None
    try:
        reader = _acquire_reader_from_pool()
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
        finally:
            _release_reader_to_pool(reader)

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

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        text = ""
        empty_reason: str | None = None
        empty_details: str | None = None
        raw_result: Any = None
        reader = None
        try:
            reader = _acquire_reader_from_pool()
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
            finally:
                _release_reader_to_pool(reader)

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
        _reset_reader_pool()
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
    _reset_reader_pool()
    with _last_ocr_lock:
        last_ocr_times.clear()
        last_ocr_results.clear()
    gc.collect()