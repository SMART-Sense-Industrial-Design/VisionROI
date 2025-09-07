import faulthandler
faulthandler.enable()

from quart import Quart, render_template, websocket, request, jsonify, send_file, redirect, Response
import asyncio
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
import os
import sys
import argparse
from types import ModuleType
from pathlib import Path
import contextlib
import inspect
import gc
import time
from typing import Callable, Awaitable, Any
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
try:
    from websockets.exceptions import ConnectionClosed
except Exception:
    ConnectionClosed = Exception

import signal  # signals

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
inference_rois: dict[str, list[dict]] = {}
active_sources: dict[str, str] = {}
save_roi_flags: dict[str, bool] = {}
inference_groups: dict[str, str | None] = {}
inference_intervals: dict[str, float] = {}

STATE_FILE = "service_state.json"
PAGE_SCORE_THRESHOLD = 0.4

app = Quart(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_ROI_DIR = os.path.realpath("data_sources")

# =========================
# Global inference queue & thread pool
# =========================
MAX_WORKERS = os.cpu_count() or 1
_INFERENCE_QUEUE = Queue(maxsize=MAX_WORKERS * 10)
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def _inference_worker():
    while True:
        item = _INFERENCE_QUEUE.get()
        if item is None:
            _INFERENCE_QUEUE.task_done()
            break
        func, args, fut = item
        try:
            res = func(*args)
            if inspect.isawaitable(res):
                res = asyncio.run(res)
            fut.set_result(res)
        except Exception as e:
            fut.set_exception(e)
        finally:
            _INFERENCE_QUEUE.task_done()

for _ in range(MAX_WORKERS):
    _EXECUTOR.submit(_inference_worker)


# =========================
# Memory helpers (free & trim)
# =========================
def _malloc_trim():
    """Try to return free heap pages back to the OS (Linux/glibc)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _free_cam_state(cam_id: str):
    """Drop all references for this camera to allow GC to reclaim big buffers."""
    # Drop conf/state that may hold refs
    camera_sources.pop(cam_id, None)
    camera_resolutions.pop(cam_id, None)
    active_sources.pop(cam_id, None)
    save_roi_flags.pop(cam_id, None)
    inference_groups.pop(cam_id, None)
    inference_rois.pop(cam_id, None)
    inference_intervals.pop(cam_id, None)

    # Queues: drain & drop
    q = frame_queues.pop(cam_id, None)
    if q is not None:
        with contextlib.suppress(Exception):
            while not q.empty():
                q.get_nowait()

    rq = roi_frame_queues.pop(cam_id, None)
    if rq is not None:
        with contextlib.suppress(Exception):
            while not rq.empty():
                rq.get_nowait()

    rres = roi_result_queues.pop(cam_id, None)
    if rres is not None:
        with contextlib.suppress(Exception):
            while not rres.empty():
                rres.get_nowait()

    # Locks
    camera_locks.pop(cam_id, None)

    # GC & trim
    gc.collect()
    _malloc_trim()


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


def save_service_state() -> None:
    cam_ids = set(camera_sources) | set(active_sources) | set(inference_tasks)
    data = {
        str(cam_id): {
            "source": camera_sources.get(cam_id),
            "resolution": list(camera_resolutions.get(cam_id, (None, None))),
            "active_source": active_sources.get(cam_id, ""),
            "inference_running": bool(
                inference_tasks.get(cam_id) and not inference_tasks[cam_id].done()
            ),
            "inference_group": inference_groups.get(cam_id),
            "interval": inference_intervals.get(cam_id, 1.0),
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
        group = cfg.get("inference_group")
        inference_intervals[cam_id] = cfg.get("interval", 1.0)
        if cfg.get("inference_running"):
            await perform_start_inference(cam_id, group=group, save_state=False)
        elif group is not None:
            inference_groups[cam_id] = group
    save_service_state()


if hasattr(app, "before_serving"):
    app.before_serving(restore_service_state)


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


# ========== Graceful shutdown (fast & robust) ==========
_SHUTTING_DOWN = False  # prevent reentry

async def _shutdown_cleanup_concurrent():
    # run after_serving cleanup but cap time
    if "_shutdown_cleanup" in globals():
        with contextlib.suppress(Exception, asyncio.TimeoutError):
            await asyncio.wait_for(_shutdown_cleanup(), timeout=2.0)

async def _graceful_exit():
    """Cancel tasks, unblock queues, release cameras, then exit fast."""
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return
    _SHUTTING_DOWN = True
    with contextlib.suppress(Exception):
        await _shutdown_cleanup_concurrent()
    # allow 202 response to flush
    await asyncio.sleep(0.05)
    os._exit(0)

def _sync_signal_handler(signum, frame):
    """Ensure SIGTERM/SIGINT triggers the same fast path."""
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(lambda: asyncio.create_task(_graceful_exit()))
    except RuntimeError:
        os._exit(0)

# install sync handlers early (more reliable than loop.add_signal_handler for our case)
signal.signal(signal.SIGTERM, _sync_signal_handler)
signal.signal(signal.SIGINT,  _sync_signal_handler)


# Close all tasks/cameras concurrently at shutdown (Quart lifespan)
if hasattr(app, "after_serving"):
    @app.after_serving
    async def _shutdown_cleanup():
        await asyncio.gather(
            *[_safe_stop(cid, inference_tasks, frame_queues) for cid in list(inference_tasks.keys())],
            *[_safe_stop(cid, roi_tasks, roi_frame_queues) for cid in list(roi_tasks.keys())],
            return_exceptions=True
        )
        for cam_id, q in list(roi_result_queues.items()):
            with contextlib.suppress(Exception):
                await q.put(None)
            roi_result_queues.pop(cam_id, None)
        gc.collect()
        _malloc_trim()  # trim heap after shutdown cleanup


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


# Health & quit endpoints (used by systemd)
@app.get("/_healthz")
async def _healthz():
    return "ok", 200


@app.post("/_quit")
async def _quit():
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return Response("already shutting down", status=202, mimetype="text/plain")
    asyncio.create_task(_graceful_exit())
    return Response("shutting down", status=202, mimetype="text/plain")


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
# Queues getters
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
        encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    except Exception:
        await asyncio.sleep(0.05)
        return
    if not encoded or buffer is None:
        await asyncio.sleep(0.05)
        return
    frame_bytes = buffer.tobytes() if hasattr(buffer, "tobytes") else buffer
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(frame_bytes)
    await asyncio.sleep(0.04)


# =========================
# Loops
# =========================
async def run_inference_loop(cam_id: str):
    module_cache: dict[str, tuple[ModuleType | None, bool, bool, bool]] = {}

    async def process_frame(frame):
        rois = inference_rois.get(cam_id, [])
        forced_group = inference_groups.get(cam_id)
        if not rois:
            return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        save_flag = bool(save_roi_flags.get(cam_id))
        output = None
        frame_time = time.time()
        best_score = -1.0
        scores: list[dict[str, float | str]] = []
        has_page = False

        # pass 1: evaluate page ROIs
        for i, r in enumerate(rois):
            if np is None or r.get('type') != 'page':
                continue
            has_page = True
            pts = r.get('points', [])
            if len(pts) != 4:
                continue
            src = np.array([[p['x'], p['y']] for p in pts], dtype=np.float32)
            color = (0, 255, 0)
            width_a = np.linalg.norm(src[0] - src[1])
            width_b = np.linalg.norm(src[2] - src[3])
            max_w = int(max(width_a, width_b))
            height_a = np.linalg.norm(src[0] - src[3])
            height_b = np.linalg.norm(src[1] - src[2])
            max_h = int(max(height_a, height_b))
            dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            roi = cv2.warpPerspective(frame, matrix, (max_w, max_h))
            template = r.get('_template')
            if template is not None:
                try:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                except Exception:
                    roi_gray = None
                if roi_gray is not None:
                    if roi_gray.shape != template.shape:
                        roi_gray = cv2.resize(roi_gray, (template.shape[1], template.shape[0]))
                    try:
                        res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
                        score = float(res[0][0])
                        scores.append({'page': r.get('page', ''), 'score': score})
                        if score > best_score:
                            best_score = score
                            output = r.get('page', '')
                    except Exception:
                        pass
            cv2.polylines(frame, [src.astype(int)], True, color, 2)
            label_pt = src[0].astype(int)
            cv2.putText(
                frame,
                str(r.get('id', i + 1)),
                (int(label_pt[0]), max(0, int(label_pt[1]) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if not has_page:
            output = forced_group or ''
        elif best_score <= PAGE_SCORE_THRESHOLD:
            output = ''

        scores.sort(key=lambda x: x['score'], reverse=True)

        if output or scores:
            try:
                q = get_roi_result_queue(cam_id)
                payload = json.dumps({'group': output, 'scores': scores})
                if q.full():
                    q.get_nowait()
                await q.put(payload)
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        for i, r in enumerate(rois):
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

            if r.get('module'):
                mod_name = r.get('module')
                module_entry = module_cache.get(mod_name)
                if module_entry is None:
                    module = load_custom_module(mod_name)
                    takes_source = False
                    takes_cam_id = False
                    takes_interval = False
                    if module:
                        proc = getattr(module, 'process', None)
                        if callable(proc):
                            params = inspect.signature(proc).parameters
                            takes_source = 'source' in params
                            takes_cam_id = 'cam_id' in params
                            takes_interval = 'interval' in params
                    module_cache[mod_name] = (module, takes_source, takes_cam_id, takes_interval)
                else:
                    module, takes_source, takes_cam_id, takes_interval = module_entry
                if module is None:
                    print(f"module '{mod_name}' not found for ROI {r.get('id', i)}")
                else:
                    process_fn = getattr(module, 'process', None)
                    if not callable(process_fn):
                        print(f"process function not found in module '{mod_name}' for ROI {r.get('id', i)}")
                    else:
                        width_a = np.linalg.norm(src[0] - src[1])
                        width_b = np.linalg.norm(src[2] - src[3])
                        max_w = int(max(width_a, width_b))
                        height_a = np.linalg.norm(src[0] - src[3])
                        height_b = np.linalg.norm(src[1] - src[2])
                        max_h = int(max(height_a, height_b))
                        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
                        matrix = cv2.getPerspectiveTransform(src, dst)
                        roi = cv2.warpPerspective(frame, matrix, (max_w, max_h))
                        args = [roi, r.get('id', str(i)), save_flag]
                        if takes_source:
                            args.append(active_sources.get(cam_id, ''))
                        if takes_cam_id:
                            args.append(cam_id)
                        if takes_interval:
                            args.append(inference_intervals.get(cam_id, 1.0))
                        fut = loop.create_future()
                        try:
                            if _INFERENCE_QUEUE.full():
                                _INFERENCE_QUEUE.get_nowait()
                            _INFERENCE_QUEUE.put_nowait((process_fn, tuple(args), fut))
                        except Exception:
                            continue

                        def _on_done(
                            f: asyncio.Future,
                            roi_img=roi,
                            roi_id=r.get('id', i),
                            cam=cam_id,
                            frame_time=frame_time,
                        ) -> None:
                            result_text = ''
                            try:
                                result = f.result()
                                if isinstance(result, str):
                                    result_text = result
                                elif isinstance(result, dict) and 'text' in result:
                                    result_text = str(result['text'])
                            except Exception:
                                pass
                            try:
                                result_time = time.time()
                                _, roi_buf = cv2.imencode(
                                    '.jpg', roi_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                                )
                                roi_b64 = base64.b64encode(roi_buf).decode('ascii')
                                q = get_roi_result_queue(cam)
                                payload = json.dumps(
                                    {
                                        'id': roi_id,
                                        'image': roi_b64,
                                        'text': result_text,
                                        'frame_time': frame_time,
                                        'result_time': result_time,
                                    }
                                )
                                if q.full():
                                    q.get_nowait()
                                q.put_nowait(payload)
                            except Exception:
                                pass

                        fut.add_done_callback(_on_done)
            else:
                print(f"module missing for ROI {r.get('id', i)}")

            src_int = src.astype(int)
            cv2.polylines(frame, [src_int], True, (255, 0, 0), 2)
            label_pt = src_int[0]
            cv2.putText(
                frame,
                str(r.get('id', i + 1)),
                (int(label_pt[0]), max(0, int(label_pt[1]) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    queue = get_frame_queue(cam_id)
    try:
        while True:
            await read_and_queue_frame(cam_id, queue, process_frame)
    except asyncio.CancelledError:
        pass
    finally:
        for mod_name, (module, _, _, _) in module_cache.items():
            if module is not None:
                with contextlib.suppress(Exception):
                    cleanup_fn = getattr(module, "cleanup", None)
                    if callable(cleanup_fn):
                        cleanup_fn()
                sys.modules.pop(f"custom_{mod_name}", None)
        module_cache.clear()
        gc.collect()


async def run_roi_loop(cam_id: str):
    queue = get_roi_frame_queue(cam_id)
    try:
        while True:
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
                await asyncio.wait_for(task, timeout=0.8)
            except (asyncio.TimeoutError, asyncio.CancelledError):
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
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

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

            # NEW: fully free per-camera state & trim memory
            _free_cam_state(cam_id)

            del worker
            gc.collect()
            _malloc_trim()

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
@app.route("/create_source", methods=["GET", "POST"])
async def create_source():
    if request.method == "GET":
        return await render_template("create_source.html")

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


@app.route("/read_log")
async def read_log():
    source = request.args.get("source", "")
    try:
        lines = int(request.args.get("lines", 20))
    except ValueError:
        lines = 20
    if not source:
        return jsonify({"lines": []})
    source = os.path.basename(source.strip())
    log_path = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, source, "custom.log"))
    if not _safe_in_base(ALLOWED_ROI_DIR, log_path) or not os.path.exists(log_path):
        return jsonify({"lines": []})
    try:
        with open(log_path, "r") as f:
            content = f.readlines()[-lines:]
    except Exception:
        content = []
    return jsonify({"lines": [line.rstrip() for line in content]})


# =========================
# Inference controls
# =========================
async def perform_start_inference(cam_id: str, rois=None, group: str | None = None, save_state: bool = True):
    """Start an inference task for the given camera.

    Returns a tuple ``(ok, resp, status)`` where ``ok`` is a boolean indicating
    success, ``resp`` is the response dictionary from ``start_camera_task`` and
    ``status`` is the HTTP status code. This mirrors the behaviour of
    ``start_roi_stream`` which bubbles up the camera start error message (e.g.
    ``"open_failed"``) to the caller so the frontend can display a more useful
    alert.
    """

    if roi_tasks.get(cam_id) and not roi_tasks[cam_id].done():
        return False, {"status": "roi_running", "cam_id": cam_id}, 400
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
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break

    _task, resp, status = await start_camera_task(cam_id, inference_tasks, run_inference_loop)
    if status != 200:
        inference_rois.pop(cam_id, None)
        inference_groups.pop(cam_id, None)
        return False, resp, status
    if save_state:
        save_service_state()
    return True, resp, status


@app.route('/start_inference/<string:cam_id>', methods=["POST"])
async def start_inference(cam_id: str):
    if roi_tasks.get(cam_id) and not roi_tasks[cam_id].done():
        return jsonify({"status": "roi_running", "cam_id": cam_id}), 400
    data = await request.get_json() or {}
    cfg = dict(data)
    rois = cfg.pop("rois", None)
    group = cfg.pop("group", None)
    interval = cfg.pop("interval", None)
    if interval is not None:
        try:
            inference_intervals[cam_id] = float(interval)
        except (TypeError, ValueError):
            inference_intervals[cam_id] = 1.0
    else:
        inference_intervals.pop(cam_id, None)
    if cfg:
        cfg_ok = await apply_camera_settings(cam_id, cfg)
        if not cfg_ok:
            return jsonify({"status": "error", "cam_id": cam_id}), 400
    ok, resp, status = await perform_start_inference(cam_id, rois, group)
    return jsonify(resp), status


@app.route('/stop_inference/<string:cam_id>', methods=["POST"])
async def stop_inference(cam_id: str):
    _, resp, status = await stop_camera_task(cam_id, inference_tasks, frame_queues)
    queue = roi_result_queues.get(cam_id)
    if queue is not None:
        await queue.put(None)
        roi_result_queues.pop(cam_id, None)
    # clear cached ROI data and flags
    rois = inference_rois.pop(cam_id, [])
    # cleanup any loaded custom modules
    seen: set[str] = set()
    for r in rois:
        mod_name = r.get("module")
        if not mod_name or mod_name in seen:
            continue
        seen.add(mod_name)
        mod_key = f"custom_{mod_name}"
        module = sys.modules.get(mod_key)
        if module is not None:
            with contextlib.suppress(Exception):
                cleanup_fn = getattr(module, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn()
            sys.modules.pop(mod_key, None)
    inference_groups.pop(cam_id, None)
    save_roi_flags.pop(cam_id, None)
    inference_intervals.pop(cam_id, None)

    # NEW: fully free per-camera state & trim memory
    _free_cam_state(cam_id)
    gc.collect()
    _malloc_trim()

    save_service_state()
    return jsonify(resp), status


@app.route('/start_roi_stream/<string:cam_id>', methods=["POST"])
async def start_roi_stream(cam_id: str):
    if inference_tasks.get(cam_id) and not inference_tasks[cam_id].done():
        return jsonify({"status": "inference_running", "cam_id": cam_id}), 400
    data = await request.get_json() or {}
    if data:
        cfg_ok = await apply_camera_settings(cam_id, data)
        if not cfg_ok:
            return jsonify({"status": "error", "cam_id": cam_id}), 400
    queue = get_roi_frame_queue(cam_id)
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    _, resp, status = await start_camera_task(cam_id, roi_tasks, run_roi_loop)
    return jsonify(resp), status


@app.route('/stop_roi_stream/<string:cam_id>', methods=["POST"])
async def stop_roi_stream(cam_id: str):
    _, resp, status = await stop_camera_task(cam_id, roi_tasks, roi_frame_queues)

    # NEW: free state & trim for ROI-only mode too
    _free_cam_state(cam_id)
    gc.collect()
    _malloc_trim()

    return jsonify(resp), status


@app.route('/inference_status/<string:cam_id>', methods=["GET"])
async def inference_status(cam_id: str):
    running = inference_tasks.get(cam_id) is not None and not inference_tasks[cam_id].done()
    return jsonify({"running": running, "cam_id": cam_id})


@app.route('/roi_stream_status/<string:cam_id>', methods=["GET"])
async def roi_stream_status(cam_id: str):
    running = roi_tasks.get(cam_id) is not None and not roi_tasks[cam_id].done()
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
    return Response(buffer.tobytes(), mimetype="image/jpeg")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VisionROI server")
    parser.add_argument("--port", type=int, default=5000, help="Port for the web server")
    parser.add_argument("--use-uvicorn", action="store_true", help="Run with uvicorn instead of built-in server")
    args = parser.parse_args()

    if args.use_uvicorn:
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
            lifespan="on",
            timeout_keep_alive=2,
            timeout_graceful_shutdown=2,
            workers=1,
        )
    else:
        app.run(port=args.port)
