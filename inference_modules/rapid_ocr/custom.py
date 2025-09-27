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
from inference_modules.base_ocr import BaseOCR, np, Image as PILImage, cv2 as _cv2  # keep compat

# ------------------------------
# RapidOCR (ONNXRuntime) - REC ONLY
# ------------------------------
try:
    from rapidocr_onnxruntime import RapidOCR as RapidOCRLib  # type: ignore
    import rapidocr_onnxruntime as rapidocr_pkg  # for model path probing
except Exception:
    RapidOCRLib = None  # type: ignore[assignment]
    rapidocr_pkg = None  # type: ignore[assignment]

# onnxruntime สำหรับตรวจสอบ/เลือก EP
try:
    import onnxruntime as ort
except Exception:
    ort = None  # type: ignore

MODULE_NAME = "rapid_ocr_rec_only"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)

_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"

# instance แบบ global (ลด overhead)
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
    os.environ.setdefault("ORT_TENSORRT_VERBOSE_LOGGING", "0")
    os.environ.setdefault("ORT_CUDA_DEVICE_ID", "0")
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    os.environ.setdefault("ORT_CUDA_GRAPH_ENABLE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _available_providers() -> list[str]:
    if ort is None:
        return []
    try:
        av = ort.get_available_providers()
        logger.info(f"[REC] onnxruntime available providers: {av}")
        return av
    except Exception as e:
        logger.warning(f"[REC] get_available_providers() failed: {e}")
        return []


def _build_provider_priority() -> list[str]:
    # อนุญาต override ผ่าน env: RAPIDOCR_ORT_PROVIDERS="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
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


def _build_provider_options_map() -> dict[str, dict]:
    trt_cache = os.environ.get("ORT_TENSORRT_ENGINE_CACHE_PATH", "trt_cache")
    trt_opts = {
        "trt_max_workspace_size": 1 << 30,  # 1GB
        "trt_engine_cache_enable": 1,
        "trt_engine_cache_path": trt_cache,
        "trt_fp16_enable": int(os.environ.get("ORT_TENSORRT_FP16_ENABLE", "1")),
        "trt_cuda_graph_enable": 1,
        "trt_timing_cache_enable": 1,
        "trt_timing_cache_path": trt_cache,
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
# Model path discovery 🔎 (REC ONLY)
# =======================
def _discover_rec_model_path() -> str | None:
    """
    ลำดับค้นหา:
      1) ENV: RAPIDOCR_REC_PATH
      2) ~/.rapidocr/**/*.onnx (เลือกไฟล์ที่มีคำว่า 'rec' หรือ 'crnn')
      3) โฟลเดอร์ในแพ็กเกจ rapidocr_onnxruntime
    """
    # 1) ENV override
    env_p = os.getenv("RAPIDOCR_REC_PATH")
    if env_p and Path(env_p).exists():
        p = str(Path(env_p).resolve())
        logger.info(f"[REC] Using rec model from env: {p}")
        return p

    def _pick_best(cands: list[Path]) -> Path | None:
        if not cands:
            return None
        return max(cands, key=lambda x: x.stat().st_size)

    # 2) ~/.rapidocr scan
    try:
        home_cache = Path.home() / ".rapidocr"
        if home_cache.exists():
            onnx_files = list(home_cache.rglob("*.onnx"))
            rec = _pick_best([p for p in onnx_files if "rec" in p.name.lower() or "crnn" in p.name.lower()])
            if rec:
                p = str(rec.resolve())
                logger.info(f"[REC] Found rec model in ~/.rapidocr: {p}")
                return p
    except Exception as e:
        logger.debug(f"[REC] scan ~/.rapidocr failed: {e}")

    # 3) package folder scan
    try:
        if rapidocr_pkg is not None:
            base = Path(rapidocr_pkg.__file__).resolve().parent
            onnx_files = list(base.rglob("*.onnx"))
            rec = _pick_best([p for p in onnx_files if "rec" in p.name.lower() or "crnn" in p.name.lower()])
            if rec:
                p = str(rec.resolve())
                logger.info(f"[REC] Found rec model in package: {p}")
                return p
    except Exception as e:
        logger.debug(f"[REC] scan rapidocr package failed: {e}")

    logger.warning("[REC] rec model path not found; RapidOCRLib may fallback to built-ins (if any)")
    return None


# ================
# Strict build 🔧 (REC ONLY)
# ================
def _try_create_session_strict(model_path: str, order: list[str], so: "ort.SessionOptions") -> tuple[Any | None, list[str]]:
    """
    ขอ EP ทีละตัว (ไม่พ่วง CPU) เพื่อให้ได้ error ที่แท้จริง
    ถ้า GPU พังทุกตัว ค่อย fallback CPU ตอนท้าย
    """
    errors: list[str] = []
    for ep in order:
        try_providers = [ep]
        try_opts = _select_provider_options(try_providers)
        try:
            s = ort.InferenceSession(model_path, sess_options=so, providers=try_providers, provider_options=try_opts)
            actual = s.get_providers() if hasattr(s, "get_providers") else []
            if not actual or actual[0] != ep:
                raise RuntimeError(f"requested {ep} but actual providers={actual}")
            logger.info(f"[REC] built session ✔️ requested={ep} actual={actual}")
            return s, errors
        except Exception as e:
            msg = f"{ep} failed or not actually used: {e}"
            errors.append(msg)
            logger.warning(f"[REC] {Path(model_path).name} -> {msg}")

    # Fallback CPU
    try:
        s = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"], provider_options=[{}])
        logger.info(f"[REC] fallback CPUExecutionProvider for {Path(model_path).name}")
        return s, errors
    except Exception as e:
        errors.append(f"CPUExecutionProvider failed too: {e}")
        logger.exception(f"[REC] even CPU failed for {Path(model_path).name}: {e}")
        return None, errors


def _rebind_rec_session(reader: Any, providers: list[str], rec_model_path: str | None) -> None:
    """
    รีบิลด์เฉพาะ text_rec_sess ให้ใช้ GPU ตามลำดับ providers
    """
    if ort is None or rec_model_path is None:
        return

    so = _make_sess_options()
    # ยึดลำดับจาก env ถ้ามี (พยายาม CUDA/TRT ตามจริง)
    try_order_env = [ep for ep in providers if ep in ("CUDAExecutionProvider", "TensorrtExecutionProvider")]
    try_order = try_order_env if try_order_env else ["CUDAExecutionProvider", "TensorrtExecutionProvider"]

    sess, errs = _try_create_session_strict(rec_model_path, try_order, so)
    if errs:
        logger.warning(f"[REC] GPU init errors (rec): {errs}")
    if sess is not None:
        setattr(reader, "text_rec_sess", sess)
        try:
            logger.info(f"[REC] text_rec_sess providers = {reader.text_rec_sess.get_providers()}")
        except Exception:
            pass


# ============
# Reader init (REC ONLY)
# ============
def _new_reader_instance() -> Any:
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime is not installed")

    _ensure_default_ort_envs()
    providers = _build_provider_priority()
    rec_model_path = _discover_rec_model_path()

    # สร้าง RapidOCR ให้ใช้เฉพาะ rec (ไม่ใส่ det/cls)
    reader = None
    try:
        kwargs: dict[str, Any] = {"providers": providers}
        if rec_model_path:
            kwargs["rec_model_path"] = rec_model_path
        # อย่า warmup ด้วยการ call ทั้ง pipeline; เราจะใช้ rec() โดยตรง
        reader = RapidOCRLib(**kwargs)  # type: ignore
        logger.info("[REC] RapidOCRLib created with providers(+rec_model_path) ✅")
    except TypeError:
        # บางเวอร์ชันอาจไม่รองรับ kwargs; ลอง init แบบว่างๆ แล้วค่อย rebind rec_sess
        reader = RapidOCRLib()  # type: ignore
        logger.info("[REC] RapidOCRLib ignored providers/paths (fallback) ❌")
    except Exception as e:
        logger.exception(f"[REC] RapidOCRLib init failed: {e}")
        raise

    # รีบิลด์ session เฉพาะ rec ให้ใช้ GPU (ถ้า init แล้วเป็น CPU-only)
    try:
        rec_ok = hasattr(reader, "text_rec_sess") and _session_uses_gpu(reader.text_rec_sess)
        if not rec_ok:
            logger.info("[REC] rec session appears CPU-only; attempting to rebuild with GPU providers…")
            _rebind_rec_session(reader, providers, rec_model_path)
    except Exception as e:
        logger.exception(f"[REC] post-init GPU self-check failed: {e}")

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
# Normalize/Extract text
# ======================
def _normalise_reader_output(result: Any) -> Any:
    """
    รองรับรูปแบบผลลัพธ์หลายทรง แต่ในโหมด rec() ปกติมักคืนเป็น list/tuple ของ [[text, score]] หรือใกล้เคียง
    """
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
# OCR runner (REC ONLY)
# ===========
def _run_rec_only_async(frame, roi_id, save, source) -> str:
    """
    รับ frame ที่ถูก crop แล้ว และทำเฉพาะ rec()
    """
    try:
        reader = _get_global_reader()
        # เรียกเฉพาะ rec — ไม่วิ่ง det/cls
        result = reader.rec(frame)  # type: ignore[attr-defined]
        result = _normalise_reader_output(result)
        text = _extract_text(result)

        logger.info(
            f"roi_id={roi_id} rec OCR result: {text}"
            if roi_id is not None
            else f"rec OCR result: {text}"
        )
    except Exception as e:
        logger.exception(f"roi_id={roi_id} rec OCR error: {e}")
        text = ""

    with _last_ocr_lock:
        last_ocr_results[roi_id] = text

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
    """
    เวอร์ชันนี้ทำเฉพาะ 'Recognition' จากภาพที่ถูก crop แล้ว
    - ไม่โหลด/ไม่เรียก det/cls
    - พยายามใช้ GPU EP (TensorRT/CUDA) สำหรับ rec model ถ้ามี
    """
    MODULE_NAME = "rapid_ocr_rec_only"

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
                # เรียกเฉพาะ rec()
                result = reader.rec(frame)  # type: ignore[attr-defined]
                result = _normalise_reader_output(result)
                text = _extract_text(result)
            except Exception as e:
                self.logger.exception(f"roi_id={roi_id} {self.MODULE_NAME} rec OCR error: {e}")

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
    """
    ส่ง frame ที่ถูก crop มาจาก upstream (เช่น VisionROI) เพื่อทำ 'rec เท่านั้น'
    """
    logger = get_logger(MODULE_NAME, source)

    if cam_id is not None:
        try:
            import app  # type: ignore
            app.save_roi_flags[cam_id] = False
        except Exception:
            pass

    if isinstance(frame, PILImage) and np is not None:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    current_time = time.monotonic()
    with _last_ocr_lock:
        last_time = last_ocr_times.get(roi_id)

    diff_time = 0 if last_time is None else current_time - last_time
    should_ocr = last_time is None or diff_time >= interval

    if should_ocr:
        with _last_ocr_lock:
            last_ocr_times[roi_id] = current_time
        return _run_rec_only_async(frame, roi_id, save, source)

    with _last_ocr_lock:
        return last_ocr_results.get(roi_id)


def cleanup() -> None:
    global _reader
    with _reader_lock:
        _reader = None
    with _last_ocr_lock:
        last_ocr_times.clear()
        last_ocr_results.clear()
    gc.collect()
