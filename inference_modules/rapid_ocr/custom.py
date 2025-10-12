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
from numbers import Number
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

_reader_executor_lock = threading.Lock()
_reader_executor: ThreadPoolExecutor | None = None
_reader_thread_locals = threading.local()

_single_reader_init_lock = threading.Lock()
_single_reader_run_lock = threading.Lock()
_single_reader: Any | None = None


def _resolve_max_readers() -> int:
    env_val = os.getenv("RAPIDOCR_MAX_READERS")
    if env_val:
        try:
            parsed = int(env_val)
            if parsed >= 1:
                return parsed
            logger.warning(
                "[RapidOCR] RAPIDOCR_MAX_READERS must be >= 1; falling back to auto"
            )
        except ValueError:
            logger.warning(
                "[RapidOCR] RAPIDOCR_MAX_READERS is not an int; falling back to auto"
            )
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 4))


def _resolve_acquire_timeout() -> float | None:
    env_val = os.getenv("RAPIDOCR_ACQUIRE_TIMEOUT")
    if env_val:
        try:
            parsed = float(env_val)
            if parsed <= 0:
                return None
            return parsed
        except ValueError:
            logger.warning(
                "[RapidOCR] RAPIDOCR_ACQUIRE_TIMEOUT is not a float; using default"
            )
    return 10.0


def _resolve_worker_mode() -> str:
    env_val = os.getenv("RAPIDOCR_WORKER_MODE", "executor").strip().lower()
    if env_val in {"executor", "threadpool", "pool"}:
        return "executor"
    if env_val in {"lock", "single", "sequential"}:
        return "lock"
    if env_val:
        logger.warning(
            "[RapidOCR] unknown RAPIDOCR_WORKER_MODE=%s, falling back to executor",
            env_val,
        )
    return "executor"


_worker_mode = _resolve_worker_mode()
_max_reader_pool = _resolve_max_readers()
_reader_acquire_timeout = _resolve_acquire_timeout()
logger.info(
    "[RapidOCR] worker mode=%s, workers=%s, acquire_timeout=%s",
    _worker_mode,
    _max_reader_pool,
    "infinite" if _reader_acquire_timeout is None else _reader_acquire_timeout,
)

last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


# ==============================
# Utilities สำหรับ ORT/TensorRT
# ==============================
def _ensure_default_ort_envs() -> None:
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault(
        "ORT_TENSORRT_ENGINE_CACHE_PATH",
        str((_data_sources_root / "trt_cache").resolve()),
    )
    os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_VERBOSE_LOGGING", "1")  # เพิ่ม log ของ TRT EP
    os.environ.setdefault("ORT_CUDA_DEVICE_ID", "0")
    # ลดความดังของ ORT ใน prod (0=VERBOSE, 3=WARNING)
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
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
    # ให้สามารถ override ผ่าน env ได้
    env_val = os.getenv("RAPIDOCR_ORT_PROVIDERS")
    if env_val:
        providers = [p.strip() for p in env_val.split(",") if p.strip()]
        return providers

    # ค่าตั้งต้น (จะ reorder ตาม availability ภายหลัง)
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    av = _available_providers()
    if not av:
        # หาก onnxruntime ไม่รายงาน provider ใดเลย (เช่นไม่ได้ติดตั้ง GPU EP)
        # ให้กลับไปใช้ CPU ล้วน เพื่อหลีกเลี่ยง warning จาก ORT ที่พยายามโหลด EP ที่ไม่มี
        return ["CPUExecutionProvider"]

    chosen = [p for p in preferred if p in av]
    if not chosen:
        # กรณีที่รายการ preferred ไม่มีอยู่ใน av (เช่น ใช้ CoreML บน macOS)
        # ยังต้องแน่ใจว่า CPU อยู่ในลิสต์เพื่อให้ RapidOCR ทำงานได้
        if "CPUExecutionProvider" in av:
            chosen.append("CPUExecutionProvider")
    elif "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _build_provider_options_map() -> dict[str, dict]:
    trt_opts = {
        "trt_max_workspace_size": 1 << 30,  # 1GB
        "trt_engine_cache_enable": 1,
        "trt_engine_cache_path": os.environ.get("ORT_TENSORRT_ENGINE_CACHE_PATH", "trt_cache"),
        "trt_fp16_enable": int(os.environ.get("ORT_TENSORRT_FP16_ENABLE", "1")),
        "trt_cuda_graph_enable": 1,
        "trt_timing_cache_enable": 1,
        "trt_timing_cache_path": os.environ.get("ORT_TENSORRT_ENGINE_CACHE_PATH", "trt_cache"),
    }
    cuda_opts = {
        "device_id": int(os.environ.get("ORT_CUDA_DEVICE_ID", "0")),
        "arena_extend_strategy": "kSameAsRequested",
    }
    return {
        "TensorrtExecutionProvider": trt_opts,
        "CUDAExecutionProvider": cuda_opts,
        "CPUExecutionProvider": {},
    }


def _select_provider_options(providers: list[str]) -> list[dict]:
    m = _build_provider_options_map()
    return [m.get(p, {}) for p in providers]


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
      2) ~/.rapidocr/**/*.onnx
      3) โฟลเดอร์ในแพ็กเกจ rapidocr_onnxruntime
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


# ================
# Strict build 🔧
# ================
def _try_create_session_strict(
    model_path: str,
    order: list[str],
    so: "ort.SessionOptions",
) -> tuple[Any | None, list[str]]:
    """
    สร้าง session แบบ strict:
      - ขอ EP ทีละตัว (ไม่พ่วง CPU) เพื่อให้เห็น error จริง
      - ถ้า session ที่ได้ไม่ได้ใช้ EP ที่ขอ (เช็ค s.get_providers()[0]) => ถือว่า fail
      - ถ้า GPU พังทุกตัว ค่อย fallback CPU ตอนท้าย
    """
    errors: list[str] = []
    for ep in order:
        try_providers = [ep]
        try_opts = _select_provider_options(try_providers)
        try:
            s = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=try_providers,
                provider_options=try_opts,
            )
            actual = s.get_providers() if hasattr(s, "get_providers") else []
            if not actual or actual[0] != ep:
                raise RuntimeError(f"requested {ep} but actual providers={actual}")
            logger.warning(f"[RapidOCR-ORT] built session ✔️ requested={ep} actual={actual}")
            return s, errors
        except Exception as e:
            msg = f"{ep} failed or not actually used: {e}"
            errors.append(msg)
            logger.warning(f"[RapidOCR-ORT] {_short_model(model_path)} -> {msg}")

    # Fallback CPU (สุดท้ายจริง ๆ)
    try:
        s = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
        )
        logger.warning(f"[RapidOCR-ORT] fallback CPUExecutionProvider for {_short_model(model_path)}")
        return s, errors
    except Exception as e:
        errors.append(f"CPUExecutionProvider failed too: {e}")
        logger.exception(f"[RapidOCR-ORT] even CPU failed for {_short_model(model_path)}: {e}")
        return None, errors


def _short_model(p: str) -> str:
    try:
        return str(Path(p).name)
    except Exception:
        return p


def _rebind_internal_sessions_if_cpu_only(reader: Any, providers: list[str], model_paths: dict[str, str]) -> None:
    """
    รีบิลด์ใหม่ด้วยลำดับ strict ตาม providers ที่เลือกไว้ (ยึดตาม env ถ้ามี):
      - CUDA -> TRT (หรือเฉพาะตัวที่มีใน providers)
      - ถ้า GPU ใช้ไม่ได้ทั้งหมด ค่อย fallback CPU
    """
    if ort is None:
        return

    so = _make_sess_options()

    # ลำดับความพยายาม: ยึดตาม providers ที่ถูกเลือกไว้ (เช่นจาก env)
    gpu_candidates = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    available = set(_available_providers())

    try_order_env = [ep for ep in providers if ep in gpu_candidates]
    try_order_env = [ep for ep in try_order_env if ep in available]

    default_try_order = [ep for ep in gpu_candidates if ep in available]

    try_order = try_order_env if try_order_env else default_try_order

    if not try_order:
        logger.warning(
            "[RapidOCR-ORT] no GPU execution providers available; skip strict rebuild"
        )
        return

    def _rebuild_one(name: str, sess_attr: str, key_in_paths: str):
        mp = model_paths.get(key_in_paths)
        if not mp:
            logger.warning(f"[RapidOCR-ORT] cannot rebuild {name}_sess: model path not found")
            return
        sess, errs = _try_create_session_strict(mp, try_order, so)
        if errs:
            logger.warning(f"[RapidOCR-ORT] {_short_model(mp)} GPU init errors: {errs}")
        if sess is not None:
            setattr(reader, sess_attr, sess)
            if hasattr(sess, "get_providers"):
                logger.warning(f"[RapidOCR-ORT] {sess_attr} providers now = {sess.get_providers()}")

    _rebuild_one("det", "text_det_sess", "det")
    _rebuild_one("rec", "text_rec_sess", "rec")
    _rebuild_one("cls", "text_cls_sess", "cls")


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
    model_paths = _discover_rapidocr_model_paths()
    logger.warning(f"[RapidOCR-ORT] requested providers={providers}")

    # 1) พยายามส่ง providers + model paths เข้า RapidOCRLib โดยตรง (ถ้ารองรับ)
    reader = None
    try:
        kwargs: dict[str, Any] = {"providers": providers}
        # ใส่ path ถ้าจับได้ (ชื่อพารามิเตอร์ที่ถูกต้อง)
        if "det" in model_paths:
            kwargs.setdefault("det_model_path", model_paths["det"])
        if "rec" in model_paths:
            kwargs.setdefault("rec_model_path", model_paths["rec"])
        if "cls" in model_paths:
            kwargs.setdefault("cls_model_path", model_paths["cls"])
        kwargs.setdefault("use_det", False)
        kwargs.setdefault("use_cls", False)
        reader = RapidOCRLib(**kwargs)  # type: ignore
        logger.warning("[RapidOCR-ORT] RapidOCRLib created with providers(+model_paths) ✅")
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
        if hasattr(reader, "text_det_sess"):
            logger.warning(f"[RapidOCR-ORT] text_det_sess providers={reader.text_det_sess.get_providers()}")
        if hasattr(reader, "text_rec_sess"):
            logger.warning(f"[RapidOCR-ORT] text_rec_sess providers={reader.text_rec_sess.get_providers()}")
        if hasattr(reader, "text_cls_sess"):
            logger.warning(f"[RapidOCR-ORT] text_cls_sess providers={reader.text_cls_sess.get_providers()}")
    except Exception as e:
        logger.debug(f"[RapidOCR-ORT] cannot inspect actual providers: {e}")

    # 2) ถ้ายัง CPU-only ให้รีบิลด์ด้วย strict ตามลำดับ (ยึด env ถ้ามี)
    try:
        det_ok = hasattr(reader, "text_det_sess") and _session_uses_gpu(reader.text_det_sess)
        rec_ok = hasattr(reader, "text_rec_sess") and _session_uses_gpu(reader.text_rec_sess)
        cls_ok = hasattr(reader, "text_cls_sess") and _session_uses_gpu(reader.text_cls_sess)
        if not (det_ok or rec_ok or cls_ok):
            logger.warning("[RapidOCR-ORT] sessions appear CPU-only; attempting to rebuild with GPU providers…")
            _rebind_internal_sessions_if_cpu_only(reader, providers, model_paths)

            # log ซ้ำหลังรีบิลด์
            if hasattr(reader, "text_det_sess"):
                logger.warning(f"[RapidOCR-ORT] text_det_sess providers(after)={reader.text_det_sess.get_providers()}")
            if hasattr(reader, "text_rec_sess"):
                logger.warning(f"[RapidOCR-ORT] text_rec_sess providers(after)={reader.text_rec_sess.get_providers()}")
            if hasattr(reader, "text_cls_sess"):
                logger.warning(f"[RapidOCR-ORT] text_cls_sess providers(after)={reader.text_cls_sess.get_providers()}")
    except Exception as e:
        logger.exception(f"[RapidOCR-ORT] post-init GPU self-check failed: {e}")

    return reader


def _ensure_reader_executor() -> ThreadPoolExecutor:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime library is not installed")

    global _reader_executor
    if _reader_executor is None:
        with _reader_executor_lock:
            if _reader_executor is None:
                _reader_executor = ThreadPoolExecutor(
                    max_workers=_max_reader_pool,
                    thread_name_prefix="rapidocr",
                )
                logger.debug(
                    "[RapidOCR] ThreadPoolExecutor created with %s workers",
                    _max_reader_pool,
                )
    return _reader_executor


def _get_thread_reader() -> Any:
    reader = getattr(_reader_thread_locals, "reader", None)
    if reader is None:
        reader = _new_reader_instance()
        _reader_thread_locals.reader = reader
        logger.debug("[RapidOCR] created thread-local reader instance")
    return reader


def _invoke_reader(frame) -> Any:
    reader = _get_thread_reader()
    return reader(frame)


def _get_single_reader() -> Any:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime library is not installed")
    global _single_reader
    if _single_reader is None:
        with _single_reader_init_lock:
            if _single_reader is None:
                _single_reader = _new_reader_instance()
                logger.debug("[RapidOCR] created singleton reader instance")
    return _single_reader


def _reset_reader_pool() -> None:
    global _reader_executor, _single_reader
    if _worker_mode == "executor":
        with _reader_executor_lock:
            if _reader_executor is not None:
                _reader_executor.shutdown(wait=True, cancel_futures=True)
                logger.debug("[RapidOCR] ThreadPoolExecutor shutdown completed")
                _reader_executor = None
        # thread-local readers จะถูกเก็บกวาดเมื่อ thread pool ถูก shutdown
        vars(_reader_thread_locals).clear()
    else:
        with _single_reader_init_lock:
            _single_reader = None


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
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
        except AttributeError:
            pass
    return frame


def _run_reader(frame):
    if _worker_mode == "executor":
        executor = _ensure_reader_executor()
        future = executor.submit(_invoke_reader, frame)
        try:
            if _reader_acquire_timeout is None:
                return future.result()
            return future.result(timeout=_reader_acquire_timeout)
        except FuturesTimeoutError as exc:
            cancelled = future.cancel()
            if not cancelled:
                logger.warning(
                    "[RapidOCR] OCR task exceeded timeout but could not be cancelled"
                )
            raise TimeoutError(
                "Timed out waiting for available RapidOCR worker thread"
            ) from exc
    reader = _get_single_reader()
    with _single_reader_run_lock:
        return reader(frame)


def _run_ocr_async(frame, roi_id, save, source) -> str:
    try:
        frame = _prepare_frame_for_reader(frame)
        result = _normalise_reader_output(_run_reader(frame))
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

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        text = ""
        try:
            frame = _prepare_frame_for_reader(frame)
            result = _normalise_reader_output(_run_reader(frame))
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