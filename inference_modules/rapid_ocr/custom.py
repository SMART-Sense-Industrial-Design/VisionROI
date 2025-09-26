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
    from rapidocr_onnxruntime import RapidOCR as RapidOCRLib  # type: ignore
    import rapidocr_onnxruntime as rapidocr_pkg  # for model path probing
except Exception:
    RapidOCRLib = None  # type: ignore[assignment]
    rapidocr_pkg = None  # type: ignore[assignment]

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

last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


# ==============================
# Utilities สำหรับ ORT/TensorRT
# ==============================
def _ensure_default_ort_envs() -> None:
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_PATH", str((_data_sources_root / "trt_cache").resolve()))
    os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")
    os.environ.setdefault("ORT_CUDA_DEVICE_ID", "0")
    os.environ.setdefault("ORT_LOGGING_LEVEL", "2")  # INFO
    os.environ.setdefault("ORT_CUDA_GRAPH_ENABLE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _available_providers() -> list[str]:
    if ort is None:
        return []
    try:
        av = ort.get_available_providers()
        logger.warning(f"[RapidOCR-ORT] onnxruntime available providers: {av}")
        return av
    except Exception as e:
        logger.warning(f"[RapidOCR-ORT] get_available_providers() failed: {e}")
        return []


def _build_provider_priority() -> list[str]:
    env_val = os.getenv("RAPIDOCR_ORT_PROVIDERS")
    if env_val:
        providers = [p.strip() for p in env_val.split(",") if p.strip()]
        return providers
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    av = _available_providers()
    if not av:
        return preferred
    chosen = [p for p in preferred if p in av]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _build_provider_options(providers: list[str]) -> list[dict] | None:
    trt_opts = {
        "trt_max_workspace_size": str(1 << 30),  # 1GB
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": os.environ.get("ORT_TENSORRT_ENGINE_CACHE_PATH", "trt_cache"),
        "trt_fp16_enable": os.environ.get("ORT_TENSORRT_FP16_ENABLE", "1"),
        "trt_cuda_graph_enable": "1",
    }
    cuda_opts = {
        "device_id": os.environ.get("ORT_CUDA_DEVICE_ID", "0"),
        "arena_extend_strategy": "kSameAsRequested",
    }
    opts: list[dict] = []
    for p in providers:
        if p == "TensorrtExecutionProvider":
            opts.append(trt_opts)
        elif p == "CUDAExecutionProvider":
            opts.append(cuda_opts)
        else:
            opts.append({})
    return opts


def _make_sess_options() -> "ort.SessionOptions | None":
    if ort is None:
        return None
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    try:
        import multiprocessing
        so.intra_op_num_threads = max(1, multiprocessing.cpu_count() // 2)
    except Exception:
        pass
    return so


def _session_uses_gpu(sess: Any) -> bool:
    try:
        ps = sess.get_providers() if hasattr(sess, "get_providers") else []
        return any(p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") for p in ps)
    except Exception:
        return False


# =======================
# Model path discovery 🔎
# =======================
def _discover_rapidocr_model_paths() -> dict[str, str]:
    """
    คืน dict ที่อาจมีคีย์ det/rec/cls -> absolute path ของ .onnx
    ลำดับค้นหา:
      1) ENV: RAPIDOCR_DET_PATH / RAPIDOCR_REC_PATH / RAPIDOCR_CLS_PATH
      2) ~/.rapidocr/**/*.onnx (ดาวน์โหลดอัตโนมัติบ่อยสุด)
      3) โฟลเดอร์ในแพ็กเกจ rapidocr_onnxruntime (เผื่อฝังมา)
    """
    found: dict[str, str] = {}

    # 1) ENV override
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

    # 2) ~/.rapidocr scan
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
    except Exception as e:
        logger.debug(f"[RapidOCR-ORT] scan ~/.rapidocr failed: {e}")

    # 3) package folder scan
    try:
        if rapidocr_pkg is not None:
            base = Path(rapidocr_pkg.__file__).resolve().parent
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
    except Exception as e:
        logger.debug(f"[RapidOCR-ORT] scan rapidocr package failed: {e}")

    logger.warning(f"[RapidOCR-ORT] discovered model paths: {found}")
    return found


def _rebind_internal_sessions_if_cpu_only(reader: Any, providers: list[str], model_paths: dict[str, str]) -> None:
    """
    รีบิลด์ใหม่ด้วย GPU providers แม้ sess เดิมจะ None:
      - ถ้ามีพาธโมเดล -> สร้าง InferenceSession ใหม่ แล้ว set กลับเข้า reader
    """
    if ort is None:
        return

    so = _make_sess_options()

    def _rebuild_one(name: str, sess_attr: str, key_in_paths: str):
        mp = model_paths.get(key_in_paths)
        if not mp:
            logger.warning(f"[RapidOCR-ORT] cannot rebuild {name}_sess: model path not found")
            return

        try:
            logger.warning(f"[RapidOCR-ORT] rebuilding {name}_sess with {providers} from model: {mp}")
            try:
                new_sess = ort.InferenceSession(
                    mp,
                    sess_options=so,
                    providers=providers,
                    provider_options=_build_provider_options(providers),
                )
            except TypeError:
                new_sess = ort.InferenceSession(mp, sess_options=so, providers=providers)
            setattr(reader, sess_attr, new_sess)
            if hasattr(new_sess, "get_providers"):
                logger.warning(f"[RapidOCR-ORT] {name}_sess providers now = {new_sess.get_providers()}")
        except Exception as e:
            logger.exception(f"[RapidOCR-ORT] failed to rebuild {name}_sess: {e}")

    _rebuild_one("det", "det_sess", "det")
    _rebuild_one("rec", "rec_sess", "rec")
    _rebuild_one("cls", "cls_sess", "cls")


def _prime_reader_sessions(reader: Any) -> None:
    """
    บังคับให้ RapidOCR สร้าง det/rec/cls sessions (แก้เคส lazy-init)
    """
    try:
        import numpy as _np
        dummy = _np.zeros((8, 8, 3), dtype=_np.uint8)
        reader(dummy)
        logger.warning("[RapidOCR-ORT] warmup call executed to initialize sessions")
    except Exception as e:
        logger.warning(f"[RapidOCR-ORT] warmup call failed: {e}")


def _new_reader_instance() -> Any:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime is not installed")

    _ensure_default_ort_envs()
    providers = _build_provider_priority()
    provider_options = _build_provider_options(providers)
    model_paths = _discover_rapidocr_model_paths()
    logger.warning(f"[RapidOCR-ORT] requested providers={providers}")

    reader = None

    # 1) พยายามส่ง providers + model paths เข้า RapidOCRLib โดยตรง
    try:
        kwargs = {"providers": providers}
        if provider_options is not None:
            kwargs["provider_options"] = provider_options  # type: ignore[assignment]
        for k in ("det_path", "det_model_path"):
            if k not in kwargs and "det" in model_paths:
                kwargs[k] = model_paths["det"]  # type: ignore[index]
                break
        for k in ("rec_path", "rec_model_path"):
            if k not in kwargs and "rec" in model_paths:
                kwargs[k] = model_paths["rec"]  # type: ignore[index]
                break
        for k in ("cls_path", "cls_model_path"):
            if k not in kwargs and "cls" in model_paths:
                kwargs[k] = model_paths["cls"]  # type: ignore[index]
                break

        reader = RapidOCRLib(**kwargs)  # type: ignore
        logger.warning("[RapidOCR-ORT] RapidOCRLib created with providers(+paths if supported) ✅")
    except TypeError:
        reader = RapidOCRLib()  # type: ignore
        logger.warning("[RapidOCR-ORT] RapidOCRLib ignored providers/paths (fallback) ❌")
    except Exception as e:
        logger.exception(f"[RapidOCR-ORT] RapidOCRLib init failed: {e}")
        raise

    # อุ่นเครื่องให้สร้าง sessions
    _prime_reader_sessions(reader)

    # Log providers ของแต่ละ session (รอบแรกหลัง warm-up)
    try:
        if hasattr(reader, "det_sess"):
            logger.warning(f"[RapidOCR-ORT] det_sess providers={reader.det_sess.get_providers()}")
        if hasattr(reader, "rec_sess"):
            logger.warning(f"[RapidOCR-ORT] rec_sess providers={reader.rec_sess.get_providers()}")
        if hasattr(reader, "cls_sess"):
            logger.warning(f"[RapidOCR-ORT] cls_sess providers={reader.cls_sess.get_providers()}")
    except Exception as e:
        logger.debug(f"[RapidOCR-ORT] cannot inspect actual providers: {e}")

    # 2) ถ้ายัง CPU-only ให้รีบิลด์ด้วย ORT + model_paths
    try:
        det_ok = hasattr(reader, "det_sess") and _session_uses_gpu(reader.det_sess)
        rec_ok = hasattr(reader, "rec_sess") and _session_uses_gpu(reader.rec_sess)
        cls_ok = hasattr(reader, "cls_sess") and _session_uses_gpu(reader.cls_sess)
        if not (det_ok or rec_ok or cls_ok):
            logger.warning("[RapidOCR-ORT] sessions appear CPU-only; attempting to rebuild with GPU providers…")
            _rebind_internal_sessions_if_cpu_only(reader, providers, model_paths)

            # log ซ้ำหลังรีบิลด์
            if hasattr(reader, "det_sess"):
                logger.warning(f"[RapidOCR-ORT] det_sess providers(after)={reader.det_sess.get_providers()}")
            if hasattr(reader, "rec_sess"):
                logger.warning(f"[RapidOCR-ORT] rec_sess providers(after)={reader.rec_sess.get_providers()}")
            if hasattr(reader, "cls_sess"):
                logger.warning(f"[RapidOCR-ORT] cls_sess providers(after)={reader.cls_sess.get_providers()}")
    except Exception as e:
        logger.exception(f"[RapidOCR-ORT] post-init GPU self-check failed: {e}")

    return reader


def _get_global_reader():
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime library is not installed")
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
