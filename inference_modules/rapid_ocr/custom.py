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
import sys

from src.utils.logger import get_logger
from src.utils.image import save_image_async
from inference_modules.base_ocr import BaseOCR, np, Image, cv2  # reuse base types/utilities

MODULE_NAME = "rapid_ocr"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)

_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

# ======================================================
# Backend imports (Paddle preferred, ONNXRuntime fallback)
# ======================================================
# Flags
_BACKEND_FORCED = os.getenv("RAPID_OCR_BACKEND", "paddle").strip().lower()  # "paddle" | "onnx"
_USE_PADDLE = False
_USE_ORT = False

# Try Paddle first (as requested)
RapidOCRPaddle = None
rapidocr_paddle_pkg = None
paddle = None
_paddle_import_err: Exception | None = None
try:
    from rapidocr_paddle import RapidOCR as RapidOCRPaddle  # type: ignore
    import rapidocr_paddle as rapidocr_paddle_pkg  # for model path probing
    try:
        import paddle  # optional but used for device set
    except Exception as e:
        paddle = None  # paddle runtime may be absent on Jetson
        _paddle_import_err = e
except Exception as e:
    _paddle_import_err = e
    RapidOCRPaddle = None
    rapidocr_paddle_pkg = None

# Try ONNXRuntime backend (for fallback)
RapidOCROrt = None
rapidocr_ort_pkg = None
ort = None
_ort_import_err: Exception | None = None
try:
    from rapidocr_onnxruntime import RapidOCR as RapidOCROrt  # type: ignore
    import rapidocr_onnxruntime as rapidocr_ort_pkg
    try:
        import onnxruntime as ort
    except Exception as e:
        _ort_import_err = e
        ort = None
except Exception as e:
    _ort_import_err = e
    RapidOCROrt = None
    rapidocr_ort_pkg = None


# =========================
# Globals / Locks / Caches
# =========================
_reader = None
_reader_lock = threading.Lock()

last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


# ====================================================
# Common utilities (text extraction, PIL->BGR, saving)
# ====================================================
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


# ===========================================================
# Paddle backend helpers (prefer if RAPID_OCR_BACKEND=paddle)
# ===========================================================
def _ensure_default_paddle_envs() -> None:
    os.environ.setdefault("RAPIDOCR_USE_GPU", "1")
    os.environ.setdefault("RAPIDOCR_USE_TRT", "0")
    os.environ.setdefault("RAPIDOCR_TRT_PRECISION", "fp16")
    os.environ.setdefault("RAPIDOCR_DEVICE_ID", "0")
    os.environ.setdefault("RAPIDOCR_MIN_SUBGRAPH", "3")
    os.environ.setdefault("RAPIDOCR_TRT_CACHE_DIR", str((_data_sources_root / "trt_cache_paddle").resolve()))
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
    os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.8")


def _paddle_gpu_available() -> bool:
    try:
        return paddle is not None and paddle.device.is_compiled_with_cuda()
    except Exception:
        return False


def _set_paddle_device() -> str:
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
        if paddle is not None:
            paddle.set_device("cpu")
        logger.info("[RapidOCR-Paddle] paddle.set_device(cpu)")
    except Exception:
        pass
    return "cpu"


def _discover_rapidocr_paddle_model_dirs() -> dict[str, str]:
    found: dict[str, str] = {}
    env_keys = [("det", "RAPIDOCR_DET_DIR"), ("rec", "RAPIDOCR_REC_DIR"), ("cls", "RAPIDOCR_CLS_DIR")]
    for key, envk in env_keys:
        p = os.getenv(envk)
        if p and Path(p).is_dir() and list(Path(p).glob("*.pdmodel")) and list(Path(p).glob("*.pdiparams")):
            found[key] = str(Path(p).resolve())

    def _pick_best_dir(dirs: list[Path], prefer_hint: str | None = None) -> Path | None:
        if not dirs:
            return None
        if prefer_hint:
            pri = [d for d in dirs if prefer_hint in d.name.lower()]
            if pri:
                def dir_size(pp: Path) -> int:
                    return sum(f.stat().st_size for f in pp.glob("**/*") if f.is_file())
                return max(pri, key=dir_size)
        def dir_size(pp: Path) -> int:
            return sum(f.stat().st_size for f in pp.glob("**/*") if f.is_file())
        return max(dirs, key=dir_size)

    def _scan_for_model_dirs(base: Path) -> dict[str, Path]:
        c = {"det": [], "rec": [], "cls": []}
        try:
            for pd in base.rglob("*.pdmodel"):
                d = pd.parent
                if list(d.glob("*.pdiparams")):
                    name = d.name.lower()
                    if "det" in name:
                        c["det"].append(d)
                    elif "rec" in name or "crnn" in name:
                        c["rec"].append(d)
                    elif "cls" in name:
                        c["cls"].append(d)
        except Exception:
            pass
        return {"det": _pick_best_dir(c["det"], "det"),
                "rec": _pick_best_dir(c["rec"], "rec"),
                "cls": _pick_best_dir(c["cls"], "cls")}

    try:
        home = Path.home() / ".rapidocr"
        if home.exists():
            res = _scan_for_model_dirs(home)
            for k, v in res.items():
                if k not in found and v is not None:
                    found[k] = str(v.resolve())
    except Exception:
        pass

    try:
        if rapidocr_paddle_pkg is not None:
            base = Path(rapidocr_paddle_pkg.__file__).resolve().parent
            res = _scan_for_model_dirs(base)
            for k, v in res.items():
                if k not in found and v is not None:
                    found[k] = str(v.resolve())
    except Exception:
        pass

    logger.info(f"[RapidOCR-Paddle] discovered model dirs: {found}")
    return found


def _new_reader_paddle() -> Any:
    if RapidOCRPaddle is None:
        raise RuntimeError(f"rapidocr_paddle import failed: {_paddle_import_err}")

    _ensure_default_paddle_envs()
    device = _set_paddle_device()
    use_gpu = device.startswith("gpu")
    use_tensorrt = (os.getenv("RAPIDOCR_USE_TRT", "0") == "1") and use_gpu
    trt_precision = os.getenv("RAPIDOCR_TRT_PRECISION", "fp16")
    min_subgraph = int(os.getenv("RAPIDOCR_MIN_SUBGRAPH", "3"))
    trt_cache_dir = Path(os.getenv("RAPIDOCR_TRT_CACHE_DIR", str(_data_sources_root / "trt_cache_paddle")))
    trt_cache_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = _discover_rapidocr_paddle_model_dirs()

    candidates: list[dict[str, Any]] = [
        {
            "use_gpu": use_gpu,
            "use_tensorrt": use_tensorrt,
            "precision": trt_precision,
            "min_subgraph_size": min_subgraph,
            "gpu_mem": 4000,
            "det_model_dir": model_dirs.get("det"),
            "rec_model_dir": model_dirs.get("rec"),
            "cls_model_dir": model_dirs.get("cls"),
            "trt_cache_dir": str(trt_cache_dir),
        },
        {
            "use_gpu": use_gpu,
            "det_model_dir": model_dirs.get("det"),
            "rec_model_dir": model_dirs.get("rec"),
            "cls_model_dir": model_dirs.get("cls"),
        },
        {"use_gpu": use_gpu},
        {},
    ]

    last_err: Exception | None = None
    for i, kw in enumerate(candidates, 1):
        try:
            reader = RapidOCRPaddle(**{k: v for k, v in kw.items() if v is not None})  # type: ignore
            logger.info(f"[RapidOCR-Paddle] created with kwargs set #{i} -> keys={list(kw.keys())}")
            try:
                import numpy as _np
                _ = reader(_np.zeros((8, 8, 3), dtype=_np.uint8))
            except Exception as e:
                logger.warning(f"[RapidOCR-Paddle] warmup failed (ignored): {e}")
            logger.info(f"[RapidOCR-Paddle] device={device} use_tensorrt={use_tensorrt} precision={trt_precision}")
            return reader
        except TypeError as e:
            last_err = e
            logger.warning(f"[RapidOCR-Paddle] kwargs set #{i} not supported: {e}")
        except Exception as e:
            last_err = e
            logger.exception(f"[RapidOCR-Paddle] init failed with kwargs set #{i}: {e}")

    raise RuntimeError(f"RapidOCR-Paddle could not be initialized: {last_err}")


# =========================================================
# ONNXRuntime backend helpers (fallback when Paddle fails)
# =========================================================
def _ensure_default_ort_envs() -> None:
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_PATH", str((_data_sources_root / "trt_cache").resolve()))
    os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    os.environ.setdefault("ORT_CUDA_GRAPH_ENABLE", "1")
    os.environ.setdefault("ORT_CUDA_DEVICE_ID", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _available_providers() -> list[str]:
    if ort is None:
        return []
    try:
        av = ort.get_available_providers()
        logger.info(f"[RapidOCR-ORT] onnxruntime available providers: {av}")
        return av
    except Exception as e:
        logger.warning(f"[RapidOCR-ORT] get_available_providers() failed: {e}")
        return []


def _build_provider_priority() -> list[str]:
    env_val = os.getenv("RAPIDOCR_ORT_PROVIDERS")
    if env_val:
        providers = [p.strip() for p in env_val.split(",") if p.strip()]
        return providers
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    av = _available_providers()
    if not av:
        return preferred
    chosen = [p for p in preferred if p in av]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _discover_rapidocr_onnx_models() -> dict[str, str]:
    found: dict[str, str] = {}
    env_keys = [("det", "RAPIDOCR_DET_PATH"), ("rec", "RAPIDOCR_REC_PATH"), ("cls", "RAPIDOCR_CLS_PATH")]
    for k, envk in env_keys:
        p = os.getenv(envk)
        if p and Path(p).exists():
            found[k] = str(Path(p).resolve())

    def _pick_best(cands: list[Path], prefer_sub: str | None = None) -> Path | None:
        if not cands:
            return None
        if prefer_sub:
            pri = [c for c in cands if prefer_sub in c.name.lower()]
            if pri:
                return max(pri, key=lambda x: x.stat().st_size)
        return max(cands, key=lambda x: x.stat().st_size)

    # ~/.rapidocr
    try:
        home_cache = Path.home() / ".rapidocr"
        if home_cache.exists():
            onnx_files = list(home_cache.rglob("*.onnx"))
            if "det" not in found:
                det = _pick_best([p for p in onnx_files if "det" in p.name.lower()], "det")
                if det:
                    found["det"] = str(det.resolve())
            if "rec" not in found:
                rec = _pick_best([p for p in onnx_files if "rec" in p.name.lower() or "crnn" in p.name.lower()], "rec")
                if rec:
                    found["rec"] = str(rec.resolve())
            if "cls" not in found:
                cls = _pick_best([p for p in onnx_files if "cls" in p.name.lower()], "cls")
                if cls:
                    found["cls"] = str(cls.resolve())
    except Exception:
        pass

    # package folder
    try:
        if rapidocr_ort_pkg is not None:
            base = Path(rapidocr_ort_pkg.__file__).resolve().parent
            onnx_files = list(base.rglob("*.onnx"))
            if "det" not in found:
                det = _pick_best([p for p in onnx_files if "det" in p.name.lower()], "det")
                if det:
                    found["det"] = str(det.resolve())
            if "rec" not in found:
                rec = _pick_best([p for p in onnx_files if "rec" in p.name.lower() or "crnn" in p.name.lower()], "rec")
                if rec:
                    found["rec"] = str(rec.resolve())
            if "cls" not in found:
                cls = _pick_best([p for p in onnx_files if "cls" in p.name.lower()], "cls")
                if cls:
                    found["cls"] = str(cls.resolve())
    except Exception:
        pass

    logger.info(f"[RapidOCR-ORT] discovered model paths: {found}")
    return found


def _new_reader_ort() -> Any:
    if RapidOCROrt is None:
        raise RuntimeError(f"rapidocr_onnxruntime import failed: {_ort_import_err}")

    _ensure_default_ort_envs()
    providers = _build_provider_priority()
    model_paths = _discover_rapidocr_onnx_models()

    # Try pass providers + paths
    try:
        kwargs: dict[str, Any] = {"providers": providers}
        if "det" in model_paths:
            kwargs.setdefault("det_path", model_paths["det"])
        if "rec" in model_paths:
            kwargs.setdefault("rec_path", model_paths["rec"])
        if "cls" in model_paths:
            kwargs.setdefault("cls_path", model_paths["cls"])
        reader = RapidOCROrt(**kwargs)  # type: ignore
        logger.info(f"[RapidOCR-ORT] created with providers={providers} (+paths if supported)")
    except TypeError:
        reader = RapidOCROrt()  # type: ignore
        logger.warning("[RapidOCR-ORT] providers/paths not supported by this version; using defaults")
    except Exception as e:
        logger.exception(f"[RapidOCR-ORT] init failed: {e}")
        raise

    # Warmup (and force sessions init internally)
    try:
        import numpy as _np
        _ = reader(_np.zeros((8, 8, 3), dtype=_np.uint8))
        logger.info("[RapidOCR-ORT] warmup executed")
    except Exception as e:
        logger.warning(f"[RapidOCR-ORT] warmup failed (ignored): {e}")

    return reader


# ====================================
# Reader factory (choose actual backend)
# ====================================
def _new_reader_instance() -> Any:
    global _USE_PADDLE, _USE_ORT  # <-- ต้องประกาศก่อน assign เสมอ

    logger.info(
        f"[rapid_ocr] sys.executable={sys.executable} backend_pref={_BACKEND_FORCED} "
        f"(paddle_err={_paddle_import_err}, ort_err={_ort_import_err})"
    )

    # Decide order
    order = ["paddle", "onnx"] if _BACKEND_FORCED != "onnx" else ["onnx", "paddle"]

    last_err: Exception | None = None
    for be in order:
        if be == "paddle":
            if RapidOCRPaddle is None:
                last_err = RuntimeError(f"Paddle backend unavailable: { _paddle_import_err }")
                logger.warning(f"[rapid_ocr] skip Paddle backend: {last_err}")
            else:
                try:
                    reader = _new_reader_paddle()
                    _USE_PADDLE, _USE_ORT = True, False
                    logger.info("[rapid_ocr] using Paddle backend ✅")
                    return reader
                except Exception as e:
                    last_err = e
                    logger.exception(f"[rapid_ocr] Paddle backend failed, will try next: {e}")

        if be == "onnx":
            if RapidOCROrt is None:
                last_err = RuntimeError(f"ONNXRuntime backend unavailable: { _ort_import_err }")
                logger.warning(f"[rapid_ocr] skip ONNX backend: {last_err}")
            else:
                try:
                    reader = _new_reader_ort()
                    _USE_PADDLE, _USE_ORT = False, True
                    logger.info("[rapid_ocr] using ONNXRuntime backend ✅")
                    return reader
                except Exception as e:
                    last_err = e
                    logger.exception(f"[rapid_ocr] ONNX backend failed: {e}")

    raise RuntimeError(f"[rapid_ocr] no available OCR backend. Last error: {last_err}")


def _get_global_reader():
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = _new_reader_instance()
        return _reader


# ===========
# OCR runners
# ===========
def _run_ocr_async(frame, roi_id, save, source) -> str:
    try:
        reader = _get_global_reader()

        # Ensure numpy BGR if PIL.Image provided
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
        with self._reader_lock:
            if self._reader is None:
                self._reader = _new_reader_instance()
            return self._reader

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        text = ""
        try:
            reader = self._get_reader()

            if isinstance(frame, Image.Image) and np is not None:
                f_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            else:
                f_bgr = frame

            result = _normalise_reader_output(reader(f_bgr))
            text = _extract_text(result)
        except Exception as e:
            self.logger.exception(f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {e}")

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
