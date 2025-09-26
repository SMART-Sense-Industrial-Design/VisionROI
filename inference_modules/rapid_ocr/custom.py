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
from inference_modules.base_ocr import BaseOCR, np, Image, cv2  # reuse base types/utilities

# ===============================
# RapidOCR (Paddle Inference) backend
# ===============================
MODULE_NAME = "rapid_ocr"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)

_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

# Try import rapidocr_paddle + paddle
try:
    from rapidocr_paddle import RapidOCR as RapidOCRLib  # type: ignore
    import rapidocr_paddle as rapidocr_pkg  # for model path probing
except Exception:
    RapidOCRLib = None  # type: ignore[assignment]
    rapidocr_pkg = None  # type: ignore[assignment]

try:
    import paddle
except Exception:
    paddle = None  # type: ignore

_reader = None
_reader_lock = threading.Lock()

last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


# =========================================
# Paddle / TensorRT environment & utilities
# =========================================
def _ensure_default_paddle_envs() -> None:
    """
    Set sensible defaults for Paddle Inference on Jetson.
    These can be overridden by environment variables from the service unit.
    """
    # GPU / TensorRT toggles (used by RapidOCR-Paddle kwargs too)
    os.environ.setdefault("RAPIDOCR_USE_GPU", "1")          # 1 to try GPU if available
    os.environ.setdefault("RAPIDOCR_USE_TRT", "0")          # 1 to try TensorRT if available
    os.environ.setdefault("RAPIDOCR_TRT_PRECISION", "fp16") # fp32|fp16|int8
    os.environ.setdefault("RAPIDOCR_DEVICE_ID", "0")        # GPU id
    os.environ.setdefault("RAPIDOCR_MIN_SUBGRAPH", "3")     # minimal subgraph size for TRT

    # Paddle runtime knobs (optional)
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")  # stable allocator
    os.environ.setdefault("FLAGS_eager_delete_tensor_gb", "0.0")         # do not eager free
    os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.8")
    # Allow TensorRT dynamic shape cache directory (optional)
    os.environ.setdefault(
        "RAPIDOCR_TRT_CACHE_DIR",
        str((_data_sources_root / "trt_cache_paddle").resolve()),
    )


def _paddle_gpu_available() -> bool:
    try:
        return paddle is not None and paddle.device.is_compiled_with_cuda()
    except Exception:
        return False


def _set_paddle_device() -> str:
    """
    Set Paddle device according to availability and env.
    Returns device string used: 'gpu:x' or 'cpu'.
    """
    want_gpu = os.getenv("RAPIDOCR_USE_GPU", "1") == "1"
    dev_id = int(os.getenv("RAPIDOCR_DEVICE_ID", "0"))

    if want_gpu and _paddle_gpu_available():
        try:
            paddle.set_device(f"gpu:{dev_id}")
            logger.info(f"[RapidOCR-Paddle] paddle.set_device(gpu:{dev_id})")
            return f"gpu:{dev_id}"
        except Exception as e:
            logger.warning(f"[RapidOCR-Paddle] set_device gpu failed: {e}")

    try:
        paddle.set_device("cpu")
        logger.info("[RapidOCR-Paddle] paddle.set_device(cpu)")
    except Exception:
        pass
    return "cpu"


# =======================
# Model path discovery 🔎
# =======================
def _discover_rapidocr_paddle_model_dirs() -> dict[str, str]:
    """
    Return dict keys det/rec/cls -> directory path containing Paddle models (.pdmodel/.pdiparams).
    Search order:
      1) ENV override: RAPIDOCR_DET_DIR / RAPIDOCR_REC_DIR / RAPIDOCR_CLS_DIR
      2) ~/.rapidocr/** (common auto-downloaded location)
      3) package folder rapidocr_paddle/** (if bundled)
    Selection heuristic: choose directory that contains '*.pdmodel' AND '*.pdiparams'.
    """
    found: dict[str, str] = {}

    # 1) ENV override
    env_keys = [("det", "RAPIDOCR_DET_DIR"), ("rec", "RAPIDOCR_REC_DIR"), ("cls", "RAPIDOCR_CLS_DIR")]
    for key, envk in env_keys:
        p = os.getenv(envk)
        if p and Path(p).exists() and Path(p).is_dir():
            if list(Path(p).glob("*.pdmodel")) and list(Path(p).glob("*.pdiparams")):
                found[key] = str(Path(p).resolve())

    def _pick_best_dir(dirs: list[Path], prefer_hint: str | None = None) -> Path | None:
        if not dirs:
            return None
        # prefer directory name containing hint
        if prefer_hint:
            pri = [d for d in dirs if prefer_hint in d.name.lower()]
            if pri:
                # choose largest (sum of file sizes) as heuristic
                def dir_size(pp: Path) -> int:
                    return sum(f.stat().st_size for f in pp.glob("**/*") if f.is_file())
                return max(pri, key=dir_size)
        # fallback: largest size
        def dir_size(pp: Path) -> int:
            return sum(f.stat().st_size for f in pp.glob("**/*") if f.is_file())
        return max(dirs, key=dir_size)

    def _scan_for_model_dirs(base: Path) -> dict[str, Path]:
        candidates = {"det": [], "rec": [], "cls": []}
        try:
            for pd in base.rglob("*.pdmodel"):
                d = pd.parent
                if list(d.glob("*.pdiparams")):
                    name = d.name.lower()
                    if "det" in name:
                        candidates["det"].append(d)
                    elif "rec" in name or "crnn" in name:
                        candidates["rec"].append(d)
                    elif "cls" in name:
                        candidates["cls"].append(d)
        except Exception:
            pass
        return {
            "det": _pick_best_dir(candidates["det"], "det"),
            "rec": _pick_best_dir(candidates["rec"], "rec"),
            "cls": _pick_best_dir(candidates["cls"], "cls"),
        }

    # 2) ~/.rapidocr
    try:
        home_cache = Path.home() / ".rapidocr"
        if home_cache.exists():
            res = _scan_for_model_dirs(home_cache)
            for k, v in res.items():
                if k not in found and v is not None:
                    found[k] = str(v.resolve())
    except Exception as e:
        logger.debug(f"[RapidOCR-Paddle] scan ~/.rapidocr failed: {e}")

    # 3) rapidocr_paddle package
    try:
        if rapidocr_pkg is not None:
            base = Path(rapidocr_pkg.__file__).resolve().parent
            res = _scan_for_model_dirs(base)
            for k, v in res.items():
                if k not in found and v is not None:
                    found[k] = str(v.resolve())
    except Exception as e:
        logger.debug(f"[RapidOCR-Paddle] scan rapidocr_paddle package failed: {e}")

    logger.info(f"[RapidOCR-Paddle] discovered model dirs: {found}")
    return found


# ==========================================
# RapidOCR-Paddle reader factory & warm-up
# ==========================================
def _new_reader_instance() -> Any:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_paddle is not installed")

    _ensure_default_paddle_envs()
    device = _set_paddle_device()
    use_gpu = device.startswith("gpu")
    use_tensorrt = os.getenv("RAPIDOCR_USE_TRT", "0") == "1" and use_gpu
    trt_precision = os.getenv("RAPIDOCR_TRT_PRECISION", "fp16")
    min_subgraph = int(os.getenv("RAPIDOCR_MIN_SUBGRAPH", "3"))
    trt_cache_dir = Path(os.getenv("RAPIDOCR_TRT_CACHE_DIR", str(_data_sources_root / "trt_cache_paddle")))
    trt_cache_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = _discover_rapidocr_paddle_model_dirs()

    # Try most-capable kwargs first; fall back gracefully if not supported by installed version.
    reader = None
    last_err: Exception | None = None

    # Candidate kwargs variants (from richer to basic)
    candidates: list[dict[str, Any]] = [
        {
            # Full config (many RapidOCR-Paddle builds accept these)
            "use_gpu": use_gpu,
            "use_tensorrt": use_tensorrt,
            "precision": trt_precision,          # "fp32" / "fp16" / "int8"
            "min_subgraph_size": min_subgraph,   # TRT subgraph size
            "gpu_mem": 4000,                     # MB (adjust if needed)
            "det_model_dir": model_dirs.get("det"),
            "rec_model_dir": model_dirs.get("rec"),
            "cls_model_dir": model_dirs.get("cls"),
            "trt_cache_dir": str(trt_cache_dir),
        },
        {
            # Without TRT options
            "use_gpu": use_gpu,
            "det_model_dir": model_dirs.get("det"),
            "rec_model_dir": model_dirs.get("rec"),
            "cls_model_dir": model_dirs.get("cls"),
        },
        {
            # Minimal (let package choose bundled models)
            "use_gpu": use_gpu,
        },
        {
            # Absolute minimal
        },
    ]

    for idx, kwargs in enumerate(candidates, 1):
        try:
            reader = RapidOCRLib(**{k: v for k, v in kwargs.items() if v is not None})  # type: ignore
            logger.info(f"[RapidOCR-Paddle] RapidOCRLib created with kwargs set #{idx} ✅ -> keys={list(kwargs.keys())}")
            break
        except TypeError as e:
            last_err = e
            logger.warning(f"[RapidOCR-Paddle] kwargs set #{idx} not supported by this RapidOCR version: {e}")
        except Exception as e:
            last_err = e
            logger.exception(f"[RapidOCR-Paddle] init failed with kwargs set #{idx}: {e}")

    if reader is None:
        raise RuntimeError(f"[RapidOCR-Paddle] failed to create RapidOCR instance: {last_err}")

    # Warm-up to initialize runtime (load models, build engines if TRT)
    try:
        import numpy as _np
        dummy = _np.zeros((8, 8, 3), dtype=_np.uint8)
        _ = reader(dummy)
        logger.info("[RapidOCR-Paddle] warmup call executed to initialize models")
    except Exception as e:
        logger.warning(f"[RapidOCR-Paddle] warmup call failed: {e}")

    # Log summary of device choices
    logger.info(
        f"[RapidOCR-Paddle] device={device} use_gpu={use_gpu} "
        f"use_tensorrt={use_tensorrt} precision={trt_precision}"
    )

    # Try to log which model dirs are actually used (if attributes exist)
    for name in ("det_model_dir", "rec_model_dir", "cls_model_dir"):
        try:
            if hasattr(reader, name):
                logger.info(f"[RapidOCR-Paddle] {name}={getattr(reader, name)}")
        except Exception:
            pass

    return reader


def _get_global_reader():
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_paddle is not installed")
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = _new_reader_instance()
        return _reader


# ======================
# Normalise/Extract text
# ======================
def _normalise_reader_output(result: Any) -> Any:
    """
    RapidOCR (both ONNX/Paddle) typically returns:
      - list of [ [box, text, score], ... ]  or
      - tuple (list, elapsed)
    This keeps the OCR list only.
    """
    if (
        isinstance(result, (list, tuple))
        and len(result) >= 1
        and isinstance(result[0], (list, tuple))
    ):
        # If it's (list, something) return list
        return result[0]
    return result


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
                # Common pattern: [box, "text", score]
                if len(current) > 1 and not isinstance(current[1], (list, tuple, dict)):
                    text_candidate = current[1]
                    if text_candidate is not None:
                        pieces.append(str(text_candidate))
                else:
                    queue.extend(current)
            else:
                pieces.append(str(current))
    return " ".join(pieces)


# ===========
# OCR runners
# ===========
def _run_ocr_async(frame, roi_id, save, source) -> str:
    try:
        reader = _get_global_reader()

        # Ensure frame is numpy BGR if PIL.Image provided
        if isinstance(frame, Image.Image) and np is not None:
            f_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        else:
            f_bgr = frame

        result = _normalise_reader_output(reader(f_bgr))
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
        save_image_async(str(path), f_bgr)

    return text


class RapidOCR(BaseOCR):
    MODULE_NAME = "rapid_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._reader_lock = threading.Lock()
        self._reader = None

    def _get_reader(self):
        if RapidOCRLib is None:
            raise RuntimeError("rapidocr_paddle is not installed")
        with self._reader_lock:
            if self._reader is None:
                self._reader = _new_reader_instance()
            return self._reader

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        text = ""
        if RapidOCRLib is not None:
            try:
                reader = self._get_reader()

                # Ensure frame is numpy BGR if PIL.Image provided
                if isinstance(frame, Image.Image) and np is not None:
                    f_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                else:
                    f_bgr = frame

                result = _normalise_reader_output(reader(f_bgr))
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
