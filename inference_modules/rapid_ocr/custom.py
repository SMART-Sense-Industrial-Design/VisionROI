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

# เก็บผลลัพธ์/เวลาเรียกครั้งล่าสุด เพื่อลดความถี่ OCR
last_ocr_times: dict = {}
last_ocr_results: dict = {}
_last_ocr_lock = threading.Lock()


# ==============================
# Utilities สำหรับ ORT/TensorRT
# ==============================
def _ensure_default_ort_envs() -> None:
    """
    ตั้งค่า env ที่เหมาะกับ Jetson/TensorRT ถ้ายังไม่ได้ตั้งค่าจากภายนอก
    """
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_PATH", str((_data_sources_root / "trt_cache").resolve()))
    os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")  # เปิด FP16 บน Jetson
    os.environ.setdefault("ORT_CUDA_DEVICE_ID", "0")        # ใช้ GPU 0
    # เปิด verbose ถ้าต้องการ debug ลึก (ตั้งเป็น 3=WARNING, 0=VERBOSE)
    os.environ.setdefault("ORT_LOGGING_LEVEL", "2")  # INFO
    # เปิด CUDA graph (บ้างรุ่นช่วยความเสถียร/ความเร็ว)
    os.environ.setdefault("ORT_CUDA_GRAPH_ENABLE", "1")


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
    """
    เรียงลำดับ EP ตามความเร็วโดยทั่วไป: TensorRT > CUDA > CPU
    อนุญาต override ผ่าน env RAPIDOCR_ORT_PROVIDERS
    """
    env_val = os.getenv("RAPIDOCR_ORT_PROVIDERS")
    if env_val:
        providers = [p.strip() for p in env_val.split(",") if p.strip()]
        return providers

    preferred = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    av = _available_providers()
    if not av:
        return preferred
    chosen = [p for p in preferred if p in av]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _build_provider_options(providers: list[str]) -> list[dict] | None:
    """
    สร้าง provider_options คู่กับ providers (ถ้า ORT เวอร์ชันรองรับ)
    - สำหรับ TensorRT: เปิด FP16/แคช/ตั้ง workspace
    - สำหรับ CUDA: เลือก device_id
    """
    trt_opts = {
        "trt_max_workspace_size": str(1 << 30),  # 1GB; ปรับได้ตาม RAM/GPU
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": os.environ.get("ORT_TENSORRT_ENGINE_CACHE_PATH", "trt_cache"),
        "trt_fp16_enable": os.environ.get("ORT_TENSORRT_FP16_ENABLE", "1"),
        # บางรุ่นช่วยให้เสถียรขึ้น
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
    # เปิด optimize สูงสุด
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # เปิด memory pattern ช่วย speed ในบางเคส
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    try:
        # Jetson มัก benefit จากการกำหนด intra_op ให้เท่ากับจำนวน CPU cores
        import multiprocessing
        so.intra_op_num_threads = max(1, multiprocessing.cpu_count() // 2)
    except Exception:
        pass
    return so


def _session_uses_gpu(sess: Any) -> bool:
    """
    เช็คว่า session นั้นมี EP ที่เป็น GPU (TRT/CUDA) อยู่ในลิสต์ providers
    หมายเหตุ: InferenceSession.get_providers() คืนลำดับความสำคัญของ EPs สำหรับ session นั้น
    """
    try:
        ps = sess.get_providers() if hasattr(sess, "get_providers") else []
        return any(p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") for p in ps)
    except Exception:
        return False


def _rebind_internal_sessions_if_cpu_only(reader: Any, providers: list[str]) -> None:
    """
    ถ้า RapidOCRLib รับ 'providers' ไม่ได้หรือยังเป็น CPU-only,
    พยายาม “รีบิลด์” det/rec/cls session ด้วย onnxruntime โดยตรง
    จากพาธโมเดลภายในอ็อบเจ็กต์ (ชื่อแอททริบิวต์อาจต่างกันในแต่ละเวอร์ชัน เลยลองหลายชื่อ)
    """
    if ort is None:
        return

    so = _make_sess_options()
    provider_opts = _build_provider_options(providers)

    def _maybe_get_path(obj: Any, candidates: list[str]) -> str | None:
        for name in candidates:
            if hasattr(obj, name):
                v = getattr(obj, name)
                if isinstance(v, (str, Path)) and Path(v).exists():
                    return str(v)
        return None

    # เก็บคู่ (ชื่อเซสชัน, พาธโมเดลที่หาได้, แอตทริบิวต์เซสชัน)
    targets: list[tuple[str, str | None, str]] = [
        ("det", _maybe_get_path(reader, ["det_path", "det_model_path", "det_model"]), "det_sess"),
        ("rec", _maybe_get_path(reader, ["rec_path", "rec_model_path", "rec_model"]), "rec_sess"),
        ("cls", _maybe_get_path(reader, ["cls_path", "cls_model_path", "cls_model"]), "cls_sess"),
    ]

    for name, model_path, sess_attr in targets:
        sess = getattr(reader, sess_attr, None)
        if sess is None:
            continue
        already_gpu = _session_uses_gpu(sess)
        if already_gpu:
            logger.info(f"[RapidOCR-ORT] {name}_sess already has GPU provider(s): {sess.get_providers()}")
            continue
        if not model_path:
            logger.warning(f"[RapidOCR-ORT] cannot rebind {name}_sess: model path not found")
            continue

        try:
            logger.info(f"[RapidOCR-ORT] rebuilding {name}_sess with providers={providers}")
            try:
                # ORT >= 1.17 รองรับ provider_options
                new_sess = ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=providers,
                    provider_options=_build_provider_options(providers),
                )
            except TypeError:
                # ถ้าเวอร์ชันเก่า ไม่รองรับ provider_options
                new_sess = ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=providers,
                )
            setattr(reader, sess_attr, new_sess)
            logger.info(f"[RapidOCR-ORT] {name}_sess providers now = {new_sess.get_providers()}")
        except Exception as e:
            logger.exception(f"[RapidOCR-ORT] failed to rebuild {name}_sess with GPU providers: {e}")


def _new_reader_instance() -> Any:
    """สร้างอินสแตนซ์ RapidOCRLib (onnxruntime backend) พร้อม providers + self-check"""
    if RapidOCRLib is None:
        raise RuntimeError("rapidocr_onnxruntime is not installed")

    _ensure_default_ort_envs()
    providers = _build_provider_priority()
    logger.info(f"[RapidOCR-ORT] requested providers={providers}")

    reader = None
    accepted_providers_arg = False

    # ลองส่ง providers / provider_options เข้าไปที่ RapidOCRLib โดยตรง
    try:
        try:
            provider_options = _build_provider_options(providers)
            reader = RapidOCRLib(providers=providers, provider_options=provider_options)  # type: ignore[call-arg]
            accepted_providers_arg = True
            logger.info("[RapidOCR-ORT] RapidOCRLib accepted 'providers' + 'provider_options' ✅")
        except TypeError:
            reader = RapidOCRLib(providers=providers)  # type: ignore[call-arg]
            accepted_providers_arg = True
            logger.info("[RapidOCR-ORT] RapidOCRLib accepted 'providers' ✅")
    except TypeError:
        # ไลบรารีรุ่นที่ไม่รองรับ พารามิเตอร์ providers
        reader = RapidOCRLib()  # type: ignore[call-arg]
        logger.warning("[RapidOCR-ORT] RapidOCRLib did NOT accept 'providers' ❌ (fallback to default)")

    # แสดง provider ของแต่ละ session
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

    # ถ้าทั้งสาม session ยังเป็น CPU-only ให้ลอง rebuild ด้วย ORT โดยตรง
    try:
        det_ok = hasattr(reader, "det_sess") and _session_uses_gpu(reader.det_sess)
        rec_ok = hasattr(reader, "rec_sess") and _session_uses_gpu(reader.rec_sess)
        cls_ok = hasattr(reader, "cls_sess") and _session_uses_gpu(reader.cls_sess)
        if not (det_ok or rec_ok or cls_ok):
            logger.warning("[RapidOCR-ORT] sessions appear CPU-only; attempting to rebuild with GPU providers…")
            _rebind_internal_sessions_if_cpu_only(reader, providers)

            # log ซ้ำหลังรีบิลด์
            if hasattr(reader, "det_sess"):
                logger.info(f"[RapidOCR-ORT] det_sess providers(after)={reader.det_sess.get_providers()}")
            if hasattr(reader, "rec_sess"):
                logger.info(f"[RapidOCR-ORT] rec_sess providers(after)={reader.rec_sess.get_providers()}")
            if hasattr(reader, "cls_sess"):
                logger.info(f"[RapidOCR-ORT] cls_sess providers(after)={reader.cls_sess.get_providers()}")
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
