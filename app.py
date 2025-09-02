from quart import Quart, render_template, websocket, request, jsonify, send_file, redirect, Response
import asyncio
import os
import sys
import platform
import argparse
import cv2
from camera_worker import CameraWorker
try:
    import numpy as np
except Exception:
    np = None
import json
import base64
import shutil
import importlib.util
from types import ModuleType
from pathlib import Path
import contextlib
import inspect
import gc
from typing import Callable, Awaitable, Any
try:
    from websockets.exceptions import ConnectionClosed
except Exception:
    ConnectionClosed = Exception

import signal
import faulthandler


# ---------- stability: limit threads (helps memory & races) ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ---------- crash backtrace (won't prevent crash, just logs) ----------
try:
    faulthandler.enable()
    faulthandler.register(signal.SIGSEGV, all_threads=True, chain=True)
    faulthandler.register(signal.SIGABRT, all_threads=True, chain=True)
except Exception:
    pass

# ===== เพิ่ม import ไว้ด้านบนไฟล์ (ใกล้ ๆ imports อื่น) =====
# ฟังก์ชันหยุดเฉพาะ source (เรียกตอนกด Stop)
try:
    from inference_modules.rapid_ocr.custom import stop_workers_for_source as rapidocr_stop
except Exception:
    rapidocr_stop = None  # rapidocr ไม่ได้ใช้/ไม่ติดตั้งใน env นี้

try:
    from inference_modules.easy_ocr.custom import stop_workers_for_source as easyocr_stop
except Exception:
    easyocr_stop = None  # easyocr ไม่ได้ใช้/ไม่ติดตั้งใน env นี้

# ฟังก์ชันหยุดทั้งระบบ (เรียกตอน SIGINT/SIGTERM และ after_serving)
try:
    from inference_modules.rapid_ocr.custom import stop_all_workers as rapidocr_stop_all
except Exception:
    rapidocr_stop_all = None

try:
    from inference_modules.easy_ocr.custom import stop_all_workers as easyocr_stop_all
except Exception:
    easyocr_stop_all = None


# === Runtime state ===
camera_workers: dict[str, CameraWorker | None] = {}
camera_sources: dict[str, int | str] = {}
camera_resolutions: dict[str, tuple[int | None, int | None]] = {}
camera_locks: dict[str, asyncio.Lock] = {}
frame_queues: dict[str, asyncio.Queue[bytes]] = {}
inference_tasks: dict[str, asyncio.Task | None] = {}
roi_frame_queues: dict[str, asyncio.Queue[bytes | None]] = {}
roi_result_queues: dict[str, asyncio.Queue[str | None]] = {}
roi_tasks: dict[str, asyncio.Task | None] = {}

# note: เดิม inference_rois เป็น list[dict] ต่อกล้อง (เก็บ rois.json ที่โหลดมา)
inference_rois: dict[str, list[dict]] = {}
active_sources: dict[str, str] = {}
save_roi_flags: dict[str, bool] = {}

# บังคับ group ในโหมด Inference Group (ถ้า None = ไม่บังคับ ใช้ผล page/roi)
inference_groups: dict[str, str | None] = {}

# โหมดงานล่าสุดของกล้องนั้นๆ: "inference" หรือ "roi_stream"
inference_mode: dict[str, str | None] = {}

# สัญญาณหยุดระดับกล้อง (ตั้ง True เมื่อสั่ง Stop)
stop_flags: dict[str, bool] = {}

STATE_FILE = "service_state.json"
PAGE_SCORE_THRESHOLD = 0.4

app = Quart(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_ROI_DIR = os.path.realpath("data_sources")


# =========================
# Memory helpers
# =========================
def _is_idle() -> bool:
    """ไม่มี worker/task เหลือ และคิวว่างทั้งหมด"""
    if any((t and not t.done()) for t in inference_tasks.values()):
        return False
    if any((t and not t.done()) for t in roi_tasks.values()):
        return False
    if camera_workers:
        return False
    # queues must be gone or empty
    for q in list(frame_queues.values()) + list(roi_frame_queues.values()) + list(roi_result_queues.values()):
        try:
            if q is not None and (not q.empty()):
                return False
        except Exception:
            pass
    return True


def _supports_trim() -> bool:
    return sys.platform.startswith("linux")


def _malloc_trim_now() -> None:
    """เรียก malloc_trim เฉพาะ Linux และเมื่อ idle จริง ๆ เท่านั้น"""
    if not _supports_trim():
        return
    if not _is_idle():
        return
    try:
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


async def _deferred_trim(delay: float = 1.0):
    """หน่วงเวลาเล็กน้อย รอทุกอย่างสงบก่อน แล้วค่อย trim (Linux เท่านั้น)"""
    await asyncio.sleep(delay)
    gc.collect()
    _malloc_trim_now()


def _free_cam_state(cam_id: str):
    """ตัด references ทั้งหมดของกล้องนี้"""
    camera_sources.pop(cam_id, None)
    camera_resolutions.pop(cam_id, None)
    active_sources.pop(cam_id, None)
    save_roi_flags.pop(cam_id, None)
    inference_groups.pop(cam_id, None)
    inference_rois.pop(cam_id, None)
    inference_mode.pop(cam_id, None)
    stop_flags.pop(cam_id, None)

    q = frame_queues.pop(cam_id, None)
    if q is not None:
        with contextlib.suppress(Exception):
            while not q.empty():
                _ = q.get_nowait()

    rq = roi_frame_queues.pop(cam_id, None)
    if rq is not None:
        with contextlib.suppress(Exception):
            while not rq.empty():
                _ = rq.get_nowait()

    rres = roi_result_queues.pop(cam_id, None)
    if rres is not None:
        with contextlib.suppress(Exception):
            while not rres.empty():
                _ = rres.get_nowait()

    camera_locks.pop(cam_id, None)
    gc.collect()


# =========================
# Utils / Security helpers
# =========================
def _safe_in_base(base: str, target: str) -> bool:
    base = os.path.realpath(base)
    target = os.path.realpath(target)
    try:
        return os.path.commonpath([base]) == os.path.commonpath([base, target])
    except Exception:
        return False


def get_cam_lock(cam_id: str) -> asyncio.Lock:
    return camera_locks.setdefault(cam_id, asyncio.Lock())


def set_stop_flag(cam_id: str, value: bool) -> None:
    stop_flags[cam_id] = bool(value)


def is_stopping(cam_id: str) -> bool:
    return bool(stop_flags.get(cam_id))


def _is_inference_running(cam_id: str) -> bool:
    t = inference_tasks.get(cam_id)
    return bool(t and not t.done())


def _is_roi_running(cam_id: str) -> bool:
    t = roi_tasks.get(cam_id)
    return bool(t and not t.done())


def save_service_state() -> None:
    # รวม cam_ids จากทั้ง inference, roi, sources เพื่อ cover ทุกเคส
    cam_ids = set(camera_sources) | set(active_sources) | set(inference_tasks) | set(roi_tasks)
    data = {
        str(cam_id): {
            "source": camera_sources.get(cam_id),
            "resolution": list(camera_resolutions.get(cam_id, (None, None))),
            "active_source": active_sources.get(cam_id, ""),
            "inference_running": _is_inference_running(cam_id),
            "roi_running": _is_roi_running(cam_id),
            "mode": inference_mode.get(cam_id),        # "inference" | "roi_stream" | None
            "group": inference_groups.get(cam_id),     # บังคับ group (ถ้ามี)
        }
        for cam_id in cam_ids
    }
    with contextlib.suppress(Exception):
        with open(STATE_FILE, "w") as f:
            json.dump({"cameras": data}, f)


def load_service_state() -> dict[str, dict[str, object]]:
    path = Path(STATE_FILE)
    if not path.exists():
        return {}
    with contextlib.suppress(Exception):
        with path.open("r") as f:
            data = json.load(f)
            cams = data.get("cameras")
            if isinstance(cams, dict):
                return cams
    return {}


def ensure_roi_ids(rois):
    if isinstance(rois, list):
        for idx, r in enumerate(rois):
            if isinstance(r, dict) and "id" not in r:
                r["id"] = str(idx + 1)
    return rois


async def restore_service_state() -> None:
    cams = load_service_state()

    for cam_id, cfg in cams.items():
        camera_sources[cam_id] = cfg.get("source")
        res = cfg.get("resolution") or [None, None]
        camera_resolutions[cam_id] = (res[0], res[1])
        active_sources[cam_id] = cfg.get("active_source", "")

        # คืนค่า group/โหมดล่าสุด
        grp = (cfg.get("group") or "").strip() or None
        if grp:
            inference_groups[cam_id] = grp
        mode = (cfg.get("mode") or "").strip() or None
        if mode in ("inference", "roi_stream"):
            inference_mode[cam_id] = mode

        # Autostart เฉพาะเมื่อ state ล่าสุดบันทึกว่า "กำลังรัน" และมี active_source
        has_source = bool(active_sources.get(cam_id))
        if has_source:
            if cfg.get("inference_running") and mode == "inference":
                await perform_start_inference(cam_id, group=inference_groups.get(cam_id), save_state=False)
            elif cfg.get("roi_running") and mode == "roi_stream":
                # ROI stream mode (หน้า inference_page / preview ROI)
                _, _, _ = await start_camera_task(cam_id, roi_tasks, run_roi_loop)

    save_service_state()


# ใช้ decorator ถ้า framework รองรับ
if hasattr(app, "before_serving"):
    @app.before_serving
    async def _on_startup():
        # ติดตั้ง signal handlers ตั้งแต่ต้น
        loop = asyncio.get_event_loop()
        _install_signal_handlers(loop)
        await restore_service_state()
else:
    async def _on_startup():
        loop = asyncio.get_event_loop()
        _install_signal_handlers(loop)
        await restore_service_state()


# ========== stop helpers ==========
async def _safe_stop(cam_id: str,
                     task_dict: dict[str, asyncio.Task | None],
                     queue_dict: dict[str, asyncio.Queue[bytes | None]] | None):
    try:
        await asyncio.wait_for(
            stop_camera_task(cam_id, task_dict, queue_dict),
            timeout=1.2
        )
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass


# ========== Graceful shutdown ==========
@app.post("/_quit")
async def _quit():
    asyncio.get_event_loop().call_soon(lambda: os.kill(os.getpid(), signal.SIGTERM))
    return Response("shutting down", status=202, mimetype="text/plain")

if hasattr(app, "after_serving"):
    @app.after_serving
    async def _shutdown_cleanup():
        # หยุด OCR modules ทั้งระบบก่อน
        try:
            if rapidocr_stop_all:
                rapidocr_stop_all()
        except Exception:
            pass
        try:
            if easyocr_stop_all:
                easyocr_stop_all()
        except Exception:
            pass

        # หยุด tasks ของกล้องทั้งหมด
        await asyncio.gather(
            *[_safe_stop(cid, inference_tasks, frame_queues) for cid in list(inference_tasks.keys())],
            *[_safe_stop(cid, roi_tasks, roi_frame_queues) for cid in list(roi_tasks.keys())],
            return_exceptions=True
        )
        # ปิดคิวผลลัพธ์ + เคลียร์ state
        for cam_id, q in list(roi_result_queues.items()):
            with contextlib.suppress(Exception):
                await q.put(None)
            roi_result_queues.pop(cam_id, None)

        inference_rois.clear()
        inference_groups.clear()
        save_roi_flags.clear()
        inference_mode.clear()
        camera_workers.clear()

        gc.collect()
        # schedule trim after event loop settles
        asyncio.create_task(_deferred_trim(1.0))
else:
    async def _shutdown_cleanup():
        try:
            if rapidocr_stop_all:
                rapidocr_stop_all()
        except Exception:
            pass
        try:
            if easyocr_stop_all:
                easyocr_stop_all()
        except Exception:
            pass
        await asyncio.gather(
            *[_safe_stop(cid, inference_tasks, frame_queues) for cid in list(inference_tasks.keys())],
            *[_safe_stop(cid, roi_tasks, roi_frame_queues) for cid in list(roi_tasks.keys())],
            return_exceptions=True
        )
        for cam_id, q in list(roi_result_queues.items()):
            with contextlib.suppress(Exception):
                await q.put(None)
            roi_result_queues.pop(cam_id, None)
        inference_rois.clear()
        inference_groups.clear()
        save_roi_flags.clear()
        inference_mode.clear()
        camera_workers.clear()
        gc.collect()
        asyncio.create_task(_deferred_trim(1.0))


# =========================
# Signal handlers (SIGINT/SIGTERM)
# =========================
def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    async def _graceful_exit(sig_name: str):
        # 1) broadcast stop ให้ OCR ทั้งระบบ (ถ้ามี)
        try:
            if rapidocr_stop_all:
                rapidocr_stop_all()
        except Exception:
            pass
        try:
            if easyocr_stop_all:
                easyocr_stop_all()
        except Exception:
            pass

        # 2) ยกเลิก tasks ของกล้องทั้งหมดอย่างสุภาพ
        try:
            for cid in list(inference_tasks.keys()):
                await _safe_stop(cid, inference_tasks, frame_queues)
            for cid in list(roi_tasks.keys()):
                await _safe_stop(cid, roi_tasks, roi_frame_queues)
        except Exception:
            pass

        # 3) ปิดคิวผลลัพธ์ + เคลียร์ state
        try:
            for cam_id, q in list(roi_result_queues.items()):
                with contextlib.suppress(Exception):
                    await q.put(None)
                roi_result_queues.pop(cam_id, None)
            inference_rois.clear()
            inference_groups.clear()
            save_roi_flags.clear()
            inference_mode.clear()
            camera_workers.clear()
        except Exception:
            pass

        gc.collect()
        # บน Linux trim memory ได้
        asyncio.create_task(_deferred_trim(0.5))
        # ออกจากโปรเซสอย่างสะอาด (เลี่ยง traceback จาก threading._shutdown)
        os._exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(_graceful_exit(s.name)))
        except NotImplementedError:
            # บางแพลตฟอร์ม (เช่น Windows) ใช้ add_signal_handler ไม่ได้: ข้ามไป
            pass


# =========================
# Routes (pages)
# =========================
@app.route("/")
async def index():
    return redirect("/home")


@app.route("/home")
async def home():
    return await render_template("home.html")


@app.route("/roi")
async def roi_page():
    return await render_template("roi_selection.html")


@app.route("/inference")
async def inference() -> str:
    return await render_template("inference.html")


@app.route("/inference_page")
async def inference_page() -> str:
    return await render_template("inference_page.html")


@app.get("/_healthz")
async def _healthz():
    return "ok", 200


# =========================
# Inference module loader
# =========================
def load_custom_module(name: str) -> ModuleType | None:
    path = Path("inference_modules") / name / "custom.py"
    if not path.exists():
        return None
    module_name = f"custom_{name}"
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =========================
# Queue getters
# =========================
def get_frame_queue(cam_id: str) -> asyncio.Queue[bytes]:
    return frame_queues.setdefault(cam_id, asyncio.Queue(maxsize=1))


def get_roi_frame_queue(cam_id: str) -> asyncio.Queue[bytes | None]:
    return roi_frame_queues.setdefault(cam_id, asyncio.Queue(maxsize=1))


def get_roi_result_queue(cam_id: str) -> asyncio.Queue[str | None]:
    return roi_result_queues.setdefault(cam_id, asyncio.Queue(maxsize=10))


# =========================
# Frame readers
# =========================
async def read_and_queue_frame(
    cam_id: str, queue: asyncio.Queue[bytes], frame_processor=None
) -> None:
    # ถ้ากำลังหยุด ไม่อ่าน/ไม่ส่งเฟรม
    if is_stopping(cam_id):
        await asyncio.sleep(0.02)
        return

    worker = camera_workers.get(cam_id)
    if worker is None:
        await asyncio.sleep(0.05)
        return
    frame = await worker.read()
    if frame is None:
        await asyncio.sleep(0.05)
        return
    if np is not None and hasattr(frame, "size") and frame.size == 0:
        await asyncio.sleep(0.05)
        return
    if frame_processor:
        frame = await frame_processor(frame)
    if frame is None:
        return
    try:
        ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok or buffer is None:
            await asyncio.sleep(0.05)
            return
        frame_bytes = buffer.tobytes() if hasattr(buffer, "tobytes") else bytes(buffer)
    except Exception:
        await asyncio.sleep(0.05)
        return
    finally:
        # drop heavy arrays ASAP
        try:
            del frame
        except Exception:
            pass
        try:
            del buffer
        except Exception:
            pass
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    await queue.put(frame_bytes)
    # drop bytes ref too
    frame_bytes = None
    await asyncio.sleep(0.04)


# =========================
# Loops
# =========================
async def run_inference_loop(cam_id: str):
    module_cache: dict[str, tuple[ModuleType | None, bool, bool]] = {}

    async def process_frame(frame):
        # ออกเร็วถ้ากำลังหยุด
        if is_stopping(cam_id):
            try:
                del frame
            except Exception:
                pass
            return None

        rois = inference_rois.get(cam_id, [])
        forced_group = inference_groups.get(cam_id)
        if not rois:
            out = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            del frame
            return out

        save_flag = bool(save_roi_flags.get(cam_id))
        output = None
        best_score = -1.0
        scores: list[dict[str, float | str]] = []
        has_page = False

        # pass 1: evaluate page ROIs
        for i, r in enumerate(rois):
            if is_stopping(cam_id):
                return None
            if np is None or r.get('type') != 'page':
                continue
            has_page = True
            pts = r.get('points', [])
            if len(pts) != 4:
                continue
            src = np.array([[p['x'], p['y']] for p in pts], dtype=np.float32)
            width_a = float(np.linalg.norm(src[0] - src[1]))
            width_b = float(np.linalg.norm(src[2] - src[3]))
            max_w = int(max(width_a, width_b))
            height_a = float(np.linalg.norm(src[0] - src[3]))
            height_b = float(np.linalg.norm(src[1] - src[2]))
            max_h = int(max(height_a, height_b))
            dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            roi = cv2.warpPerspective(frame, matrix, (max_w, max_h))
            template = r.get('_template')
            if template is not None:
                try:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    if roi_gray.shape != template.shape:
                        roi_gray = cv2.resize(roi_gray, (template.shape[1], template.shape[0]))
                    res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
                    score = float(res[0][0])
                    scores.append({'page': r.get('page', ''), 'score': score})
                    if score > best_score:
                        best_score = score
                        output = r.get('page', '')
                except Exception:
                    pass
                finally:
                    try:
                        del roi_gray
                    except Exception:
                        pass
            # draw
            cv2.polylines(frame, [src.astype(int)], True, (0, 255, 0), 2)
            label_pt = src[0].astype(int)
            cv2.putText(frame, str(r.get('id', i + 1)),
                        (int(label_pt[0]), max(0, int(label_pt[1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # drop temps
            del src, dst, matrix, roi

        if is_stopping(cam_id):
            return None

        if not has_page:
            output = forced_group or ''
        elif best_score <= PAGE_SCORE_THRESHOLD:
            output = ''

        scores.sort(key=lambda x: x['score'], reverse=True)

        if (output or scores) and not is_stopping(cam_id):
            try:
                q = get_roi_result_queue(cam_id)
                payload = json.dumps({'group': output, 'scores': scores})
                if q.full():
                    q.get_nowait()
                await q.put(payload)
            except Exception:
                pass

        for i, r in enumerate(rois):
            if is_stopping(cam_id):
                return None
            if r.get('type') != 'roi':
                continue
            if forced_group != 'all':
                if not output or r.get('group') != output:
                    continue
            if np is None:
                continue
            pts = r.get('points', [])
            if len(pts) != 4:
                continue
            src = np.array([[p['x'], p['y']] for p in pts], dtype=np.float32)
            try:
                if r.get('module'):
                    mod_name = r.get('module')
                    module_entry = module_cache.get(mod_name)
                    if module_entry is None:
                        module = load_custom_module(mod_name)
                        takes_source = False
                        takes_cam_id = False
                        if module:
                            proc = getattr(module, 'process', None)
                            if callable(proc):
                                params = inspect.signature(proc).parameters
                                takes_source = 'source' in params
                                takes_cam_id = 'cam_id' in params
                        module_cache[mod_name] = (module, takes_source, takes_cam_id)
                    else:
                        module, takes_source, takes_cam_id = module_entry
                    if module is not None:
                        width_a = float(np.linalg.norm(src[0] - src[1]))
                        width_b = float(np.linalg.norm(src[2] - src[3]))
                        max_w = int(max(width_a, width_b))
                        height_a = float(np.linalg.norm(src[0] - src[3]))
                        height_b = float(np.linalg.norm(src[1] - src[2]))
                        max_h = int(max(height_a, height_b))
                        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
                        matrix = cv2.getPerspectiveTransform(src, dst)
                        roi = cv2.warpPerspective(frame, matrix, (max_w, max_h))
                        result_text = ''
                        try:
                            if is_stopping(cam_id):
                                raise RuntimeError("stopping")
                            args = [roi, r.get('id', str(i)), bool(save_flag)]
                            if takes_source:
                                args.append(active_sources.get(cam_id, ''))
                            if takes_cam_id:
                                args.append(cam_id)
                            result = getattr(module, 'process')(*args)
                            if inspect.isawaitable(result):
                                result = await result
                            if is_stopping(cam_id):
                                raise RuntimeError("stopping")
                            if isinstance(result, str):
                                result_text = result
                            elif isinstance(result, dict) and 'text' in result:
                                result_text = str(result['text'])
                        except Exception:
                            pass
                        try:
                            ok, roi_buf = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok and roi_buf is not None:
                                roi_b64 = base64.b64encode(roi_buf).decode('ascii')
                                if not is_stopping(cam_id):
                                    q = get_roi_result_queue(cam_id)
                                    payload = json.dumps({'id': r.get('id', i), 'image': roi_b64, 'text': result_text})
                                    if q.full():
                                        q.get_nowait()
                                    await q.put(payload)
                        except Exception:
                            pass
                        finally:
                            try:
                                del roi_buf
                            except Exception:
                                pass
                            try:
                                del roi
                            except Exception:
                                pass
                    else:
                        print(f"module '{mod_name}' not found for ROI {r.get('id', i)}")
                else:
                    print(f"module missing for ROI {r.get('id', i)}")
            finally:
                src_int = src.astype(int)
                cv2.polylines(frame, [src_int], True, (255, 0, 0), 2)
                label_pt = src_int[0]
                cv2.putText(frame, str(r.get('id', i + 1)),
                            (int(label_pt[0]), max(0, int(label_pt[1]) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                del src_int, src

        out = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        del frame
        return out

    queue = get_frame_queue(cam_id)
    try:
        while True:
            if is_stopping(cam_id):
                break
            await read_and_queue_frame(cam_id, queue, process_frame)
    except asyncio.CancelledError:
        pass
    finally:
        for mod_name, (module, _, _) in module_cache.items():
            if module is not None:
                sys.modules.pop(f"custom_{mod_name}", None)
        module_cache.clear()
        gc.collect()


async def run_roi_loop(cam_id: str):
    queue = get_roi_frame_queue(cam_id)
    try:
        while True:
            if is_stopping(cam_id):
                break
            await read_and_queue_frame(cam_id, queue)
    except asyncio.CancelledError:
        pass


# =========================
# Start/Stop helpers
# =========================
async def start_camera_task(
    cam_id: str,
    task_dict: dict[str, asyncio.Task | None],
    loop_func: Callable[[str], Awaitable[Any]],
):
    async with get_cam_lock(cam_id):
        task = task_dict.get(cam_id)
        if task is not None and not task.done():
            return task, {"status": "already_running", "cam_id": cam_id}, 200

        worker = camera_workers.get(cam_id)
        if worker is None:
            src = camera_sources.get(cam_id, 0)
            width, height = camera_resolutions.get(cam_id, (None, None))
            worker = CameraWorker(src, asyncio.get_running_loop(), width, height)
            if not worker.start():
                await worker.stop()
                camera_workers.pop(cam_id, None)
                return (
                    task,
                    {"status": "error", "message": "open_failed", "cam_id": cam_id},
                    400,
                )
            camera_workers[cam_id] = worker

        task = asyncio.create_task(loop_func(cam_id))
        task_dict[cam_id] = task
        return task, {"status": "started", "cam_id": cam_id}, 200


async def stop_camera_task(
    cam_id: str,
    task_dict: dict[str, asyncio.Task | None],
    queue_dict: dict[str, asyncio.Queue[bytes | None]] | None = None,
):
    async with get_cam_lock(cam_id):
        task = task_dict.pop(cam_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except asyncio.CancelledError:
                pass
            status = "stopped"
        else:
            status = "no_task"

        if queue_dict is not None:
            queue = queue_dict.pop(cam_id, None)
            if queue is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(queue.put(None), timeout=0.2)
                for _ in range(3):
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()

        inf_task = inference_tasks.get(cam_id)
        roi_task = roi_tasks.get(cam_id)
        worker = camera_workers.get(cam_id)
        if (
            (inf_task is None or inf_task.done())
            and (roi_task is None or roi_task.done())
            and worker
        ):
            await worker.stop()
            camera_workers.pop(cam_id, None)
            # fully free per-camera state
            _free_cam_state(cam_id)
            try:
                del worker
            except Exception:
                pass
            gc.collect()
            # schedule safe trim a bit later
            asyncio.create_task(_deferred_trim(1.0))

        return task, {"status": status, "cam_id": cam_id}, 200


# =========================
# WebSockets
# =========================
@app.websocket('/ws/<string:cam_id>')
async def ws(cam_id: str):
    queue = get_frame_queue(cam_id)
    try:
        while True:
            frame_bytes = await queue.get()
            if frame_bytes is None:
                await websocket.close(code=1000)
                break
            await websocket.send(frame_bytes)
            # drop ref immediately
            frame_bytes = None
    except ConnectionClosed:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


@app.websocket('/ws_roi/<string:cam_id>')
async def ws_roi(cam_id: str):
    queue = get_roi_frame_queue(cam_id)
    try:
        while True:
            frame_bytes = await queue.get()
            if frame_bytes is None:
                await websocket.close(code=1000)
                break
            await websocket.send(frame_bytes)
            frame_bytes = None
    except ConnectionClosed:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


@app.websocket('/ws_roi_result/<string:cam_id>')
async def ws_roi_result(cam_id: str):
    queue = get_roi_result_queue(cam_id)
    try:
        while True:
            data = await queue.get()
            if data is None:
                await websocket.close(code=1000)
                break
            await websocket.send(data)
            data = None
    except ConnectionClosed:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


# =========================
# Camera config
# =========================
async def apply_camera_settings(cam_id: str, data: dict) -> bool:
    async with get_cam_lock(cam_id):
        active_sources[cam_id] = data.get("name", "")
        source_val = data.get("source", "")
        width_val = data.get("width")
        height_val = data.get("height")
        try:
            w = int(width_val) if width_val not in (None, "") else None
        except ValueError:
            w = None
        try:
            h = int(height_val) if height_val not in (None, "") else None
        except ValueError:
            h = None
        camera_resolutions[cam_id] = (w, h)

        worker = camera_workers.pop(cam_id, None)
        if worker:
            await worker.stop()

        try:
            camera_sources[cam_id] = int(source_val)
        except ValueError:
            camera_sources[cam_id] = source_val

        if (
            (inference_tasks.get(cam_id) and not inference_tasks[cam_id].done())
            or (roi_tasks.get(cam_id) and not roi_tasks[cam_id].done())
        ):
            width, height = camera_resolutions.get(cam_id, (None, None))
            worker = CameraWorker(
                camera_sources[cam_id], asyncio.get_running_loop(), width, height
            )
            if not worker.start():
                save_service_state()
                return False
            camera_workers[cam_id] = worker

        save_service_state()
        return True


# =========================
# Source / Module listing
# =========================
@app.route("/data_sources")
async def list_sources():
    base_dir = Path(ALLOWED_ROI_DIR)
    try:
        names = [d.name for d in base_dir.iterdir() if d.is_dir()]
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.route("/inference_modules")
async def list_inference_modules():
    base_dir = Path(__file__).resolve().parent / "inference_modules"
    try:
        names = [d.name for d in base_dir.iterdir() if d.is_dir()]
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.route("/groups")
async def list_groups():
    base_dir = Path(ALLOWED_ROI_DIR)
    groups: set[str] = set()
    try:
        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            roi_path = d / "rois.json"
            if not roi_path.exists():
                continue
            try:
                data = json.loads(roi_path.read_text())
                for r in data:
                    p = r.get("group") or r.get("page") or r.get("name")
                    if p:
                        groups.add(str(p))
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return jsonify(sorted(groups))


@app.route("/source_list", methods=["GET"])
async def source_list():
    base_dir = Path(ALLOWED_ROI_DIR)
    result = []
    try:
        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            cfg_path = d / "config.json"
            if not cfg_path.exists():
                continue
            try:
                with cfg_path.open("r") as f:
                    cfg = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            result.append(
                {
                    "name": cfg.get("name", d.name),
                    "source": cfg.get("source", ""),
                    "width": cfg.get("width"),
                    "height": cfg.get("height"),
                }
            )
    except FileNotFoundError:
        pass
    return jsonify(result)


@app.route("/source_config")
async def source_config():
    name = os.path.basename(request.args.get("name", "").strip())
    path = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name, "config.json"))
    if not _safe_in_base(ALLOWED_ROI_DIR, path) or not os.path.exists(path):
        return jsonify({"status": "error", "message": "not found"}), 404
    with open(path, "r") as f:
        cfg = json.load(f)
    return jsonify(cfg)


# =========================
# Source CRUD (secure)
# =========================
@app.route("/create_source", methods=["GET"], strict_slashes=False)
async def create_source_get():
    return await render_template("create_source.html")


@app.route("/create_source", methods=["POST"], strict_slashes=False)
async def create_source_post():
    form = await request.form
    name = os.path.basename(form.get("name", "").strip())
    source = form.get("source", "").strip()
    width = form.get("width")
    height = form.get("height")
    if not name or not source:
        return jsonify({"status": "error", "message": "missing data"}), 400

    source_dir = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, source_dir):
        return jsonify({"status": "error", "message": "invalid name"}), 400

    try:
        os.makedirs(source_dir, exist_ok=False)
    except FileExistsError:
        return jsonify({"status": "error", "message": "name exists"}), 400

    try:
        config = {
            "name": name,
            "source": source,
            "rois": "rois.json",
        }
        if width:
            try:
                config["width"] = int(width)
            except ValueError:
                pass
        if height:
            try:
                config["height"] = int(height)
            except ValueError:
                pass
        rois_path = os.path.join(source_dir, "rois.json")
        with open(rois_path, "w") as f:
            f.write("[]")
        with open(os.path.join(source_dir, "config.json"), "w") as f:
            json.dump(config, f)
    except Exception:
        shutil.rmtree(source_dir, ignore_errors=True)
        return jsonify({"status": "error", "message": "save failed"}), 500

    return jsonify({"status": "created"})


@app.route("/delete_source/<name>", methods=["DELETE"])
async def delete_source(name: str):
    name = os.path.basename(name.strip())
    directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, directory) or not os.path.exists(directory):
        return jsonify({"status": "error", "message": "not found"}), 404
    shutil.rmtree(directory)
    json_path = "sources.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = [s for s in data if s != name]
            elif isinstance(data, dict):
                data.pop(name, None)
                if "sources" in data and isinstance(data["sources"], list):
                    data["sources"] = [s for s in data["sources"] if s != name]
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    return jsonify({"status": "deleted"})


# =========================
# ROI save/load (secure)
# =========================
@app.route("/save_roi", methods=["POST"])
async def save_roi():
    data = await request.get_json()
    rois = ensure_roi_ids(data.get("rois", []))
    path = request.args.get("path", "")
    base_dir = ALLOWED_ROI_DIR

    if path:
        full_path = os.path.realpath(os.path.join(base_dir, path))
        if not _safe_in_base(base_dir, full_path):
            return jsonify({"status": "error", "message": "path outside allowed directory"}), 400
        dir_path = os.path.dirname(full_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(rois, f, indent=2)
        return jsonify({"status": "saved", "filename": full_path})

    name = os.path.basename(data.get("source", "").strip())
    if not name:
        return jsonify({"status": "error", "message": "missing source"}), 400
    directory = os.path.realpath(os.path.join(base_dir, name))
    if not _safe_in_base(base_dir, directory):
        return jsonify({"status": "error", "message": "invalid source path"}), 400

    config_path = os.path.join(directory, "config.json")
    if not os.path.exists(config_path):
        return jsonify({"status": "error", "message": "config not found"}), 404
    with open(config_path, "r") as f:
        cfg = json.load(f)
    roi_file = cfg.get("rois", "rois.json")
    roi_path = os.path.realpath(os.path.join(directory, roi_file))
    if not _safe_in_base(directory, roi_path):
        return jsonify({"status": "error", "message": "invalid ROI filename"}), 400
    with open(roi_path, "w") as f:
        json.dump(rois, f, indent=2)
    return jsonify({"status": "saved", "filename": roi_path})


@app.route("/load_roi/<name>")
async def load_roi(name: str):
    name = os.path.basename(name.strip())
    directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, directory):
        return jsonify({"rois": [], "filename": "None"})
    config_path = os.path.join(directory, "config.json")
    if not os.path.exists(config_path):
        return jsonify({"rois": [], "filename": "None"})
    with open(config_path, "r") as f:
        cfg = json.load(f)
    roi_file = cfg.get("rois", "rois.json")
    roi_path = os.path.realpath(os.path.join(directory, roi_file))
    if not _safe_in_base(directory, roi_path) or not os.path.exists(roi_path):
        return jsonify({"rois": [], "filename": roi_file})
    with open(roi_path, "r") as f:
        rois = ensure_roi_ids(json.load(f))
    return jsonify({"rois": rois, "filename": roi_file})


@app.route("/load_roi_file")
async def load_roi_file():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"rois": [], "filename": "None"})
    full_path = os.path.realpath(path)
    if not _safe_in_base(ALLOWED_ROI_DIR, full_path) or not os.path.exists(full_path):
        return jsonify({"rois": [], "filename": "None"})
    with open(full_path, "r") as f:
        rois = ensure_roi_ids(json.load(f))
    return jsonify({"rois": rois, "filename": os.path.basename(full_path)})


# =========================
# Inference controls
# =========================
async def perform_start_inference(cam_id: str, rois=None, group: str | None = None, save_state: bool = True):
    # โหมด inference (หน้า Inference Group)
    inference_mode[cam_id] = "inference"
    # เคลียร์สัญญาณหยุดเมื่อเริ่มใหม่
    set_stop_flag(cam_id, False)

    if roi_tasks.get(cam_id) and not roi_tasks[cam_id].done():
        return False
    if rois is None:
        source = active_sources.get(cam_id, "")
        source_dir = os.path.join(ALLOWED_ROI_DIR, source)
        rois_path = os.path.join(source_dir, "rois.json")
        try:
            with open(rois_path) as f:
                rois = json.load(f)
        except FileNotFoundError:
            rois = []
    elif isinstance(rois, str):
        try:
            rois = json.loads(rois)
        except json.JSONDecodeError:
            rois = []
    if not isinstance(rois, list):
        rois = []
    rois = ensure_roi_ids(rois)

    if np is not None:
        for r in rois:
            if isinstance(r, dict) and r.get("type") == "page":
                img_b64 = r.get("image")
                if img_b64:
                    try:
                        arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
                        tmpl = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        tmpl = None
                    r["_template"] = tmpl

    inference_rois[cam_id] = rois
    if group is None:
        inference_groups.pop(cam_id, None)
    else:
        inference_groups[cam_id] = group

    q = get_roi_result_queue(cam_id)
    while not q.empty():
        with contextlib.suppress(asyncio.QueueEmpty):
            q.get_nowait()

    await start_camera_task(cam_id, inference_tasks, run_inference_loop)
    if save_state:
        save_service_state()
    return True


@app.route('/start_inference/<string:cam_id>', methods=["POST"])
async def start_inference(cam_id: str):
    if roi_tasks.get(cam_id) and not roi_tasks[cam_id].done():
        return jsonify({"status": "roi_running", "cam_id": cam_id}), 400
    data = await request.get_json() or {}
    cfg = dict(data)
    rois = cfg.pop("rois", None)
    group = cfg.pop("group", None)
    if cfg:
        cfg_ok = await apply_camera_settings(cam_id, cfg)
        if not cfg_ok:
            return jsonify({"status": "error", "cam_id": cam_id}), 400
    ok = await perform_start_inference(cam_id, rois, group)
    if ok:
        return jsonify({"status": "started", "cam_id": cam_id}), 200
    return jsonify({"status": "error", "cam_id": cam_id}), 400


@app.route('/stop_inference/<string:cam_id>', methods=["POST"])
async def stop_inference(cam_id: str):
    # ตั้งสัญญาณหยุดก่อน เพื่อให้ลูป/โมดูลไม่ส่งผลเพิ่ม
    set_stop_flag(cam_id, True)

    # หยุด OCR workers ของ source นี้ (rapid_ocr และ easy_ocr)
    source = (active_sources.get(cam_id, "") or "")
    for mod_name, stop_func in (("rapid_ocr", rapidocr_stop), ("easy_ocr", easyocr_stop)):
        try:
            if stop_func:
                stop_func(source)
            else:
                mod = load_custom_module(mod_name)
                if mod and hasattr(mod, "stop_workers_for_source"):
                    mod.stop_workers_for_source(source)
        except Exception:
            pass

    _, resp, status = await stop_camera_task(cam_id, inference_tasks, frame_queues)
    queue = roi_result_queues.get(cam_id)
    if queue is not None:
        await queue.put(None)
        roi_result_queues.pop(cam_id, None)
    # clear cached ROI data and flags
    inference_rois.pop(cam_id, None)
    inference_groups.pop(cam_id, None)
    save_roi_flags.pop(cam_id, None)
    inference_mode.pop(cam_id, None)

    _free_cam_state(cam_id)
    save_service_state()
    gc.collect()
    _malloc_trim_now()
    # schedule safe trim shortly after stop
    asyncio.create_task(_deferred_trim(1.0))
    return jsonify(resp), status


@app.route('/start_roi_stream/<string:cam_id>', methods=["POST"])
async def start_roi_stream(cam_id: str):
    # โหมด ROI stream (หน้า Inference Page / preview ROI)
    inference_mode[cam_id] = "roi_stream"
    # เคลียร์สัญญาณหยุดเมื่อเริ่มใหม่
    set_stop_flag(cam_id, False)

    if inference_tasks.get(cam_id) and not inference_tasks[cam_id].done():
        return jsonify({"status": "inference_running", "cam_id": cam_id}), 400
    data = await request.get_json() or {}
    if data:
        cfg_ok = await apply_camera_settings(cam_id, data)
        if not cfg_ok:
            return jsonify({"status": "error", "cam_id": cam_id}), 400
    queue = get_roi_frame_queue(cam_id)
    while not queue.empty():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    _, resp, status = await start_camera_task(cam_id, roi_tasks, run_roi_loop)
    save_service_state()
    return jsonify(resp), status


@app.route('/stop_roi_stream/<string:cam_id>', methods=["POST"])
async def stop_roi_stream(cam_id: str):
    # ตั้งสัญญาณหยุดก่อน
    set_stop_flag(cam_id, True)

    # หยุด OCR workers ของ source นี้ (rapid_ocr และ easy_ocr)
    source = (active_sources.get(cam_id, "") or "")
    for mod_name, stop_func in (("rapid_ocr", rapidocr_stop), ("easy_ocr", easyocr_stop)):
        try:
            if stop_func:
                stop_func(source)
            else:
                mod = load_custom_module(mod_name)
                if mod and hasattr(mod, "stop_workers_for_source"):
                    mod.stop_workers_for_source(source)
        except Exception:
            pass

    _, resp, status = await stop_camera_task(cam_id, roi_tasks, roi_frame_queues)
    inference_mode.pop(cam_id, None)
    _free_cam_state(cam_id)
    save_service_state()
    gc.collect()
    _malloc_trim_now()
    asyncio.create_task(_deferred_trim(1.0))
    return jsonify(resp), status


@app.route('/inference_status/<string:cam_id>', methods=["GET"])
async def inference_status(cam_id: str):
    running = _is_inference_running(cam_id)
    return jsonify({"running": running, "cam_id": cam_id})


@app.route('/roi_stream_status/<string:cam_id>', methods=["GET"])
async def roi_stream_status(cam_id: str):
    running = _is_roi_running(cam_id)
    source = active_sources.get(cam_id, "")
    return jsonify({"running": running, "cam_id": cam_id, "source": source})


# =========================
# Snapshot (safer Response)
# =========================
@app.route("/ws_snapshot/<string:cam_id>")
async def ws_snapshot(cam_id: str):
    worker = camera_workers.get(cam_id)
    if worker is None:
        return "Camera not initialized", 400
    frame = await worker.read()
    if frame is None:
        return "Camera error", 500
    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return "Encode error", 500
    try:
        data = buffer.tobytes()
    finally:
        try:
            del frame
            del buffer
        except Exception:
            pass
    return Response(data, mimetype="image/jpeg")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VisionROI server")
    parser.add_argument("--port", type=int, default=5000, help="Port for the web server")
    parser.add_argument("--use-uvicorn", action="store_true", help="Run with uvicorn instead of built-in server")
    args = parser.parse_args()

    # default to uvicorn
    use_uvicorn = True if args.use_uvicorn or True else False

    if use_uvicorn:
        import uvicorn
        uvicorn_kwargs = dict(
            host="0.0.0.0",
            port=args.port,
            reload=False,
            lifespan="on",
            workers=1,
            timeout_keep_alive=2,
            timeout_graceful_shutdown=2,
        )
        # On macOS, use pure-Python backends to avoid C-extension races
        if sys.platform == "darwin":
            uvicorn_kwargs.update({"loop": "asyncio", "http": "h11"})
        uvicorn.run("app:app", **uvicorn_kwargs)
    else:
        app.run(port=args.port)
