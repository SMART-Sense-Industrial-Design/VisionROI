from quart import Quart, render_template, websocket, request, jsonify, send_file, redirect, Response
import asyncio
import cv2
from camera_worker import CameraWorker
try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy is missing
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
import signal
from typing import Callable, Awaitable, Any
try:  # pragma: no cover
    from websockets.exceptions import ConnectionClosed
except Exception:  # websockets not installed
    ConnectionClosed = Exception

# เก็บ worker ของกล้องแต่ละตัวในรูปแบบ dict
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

# ไฟล์สำหรับเก็บสถานะการทำงาน เพื่อให้สามารถกลับมารันต่อหลังรีสตาร์ท service
STATE_FILE = "service_state.json"
PAGE_SCORE_THRESHOLD = 0.4

app = Quart(__name__)
# กำหนดเพดานขนาดไฟล์ที่เซิร์ฟเวอร์ยอมรับ (100 MB)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_ROI_DIR = os.path.realpath("data_sources")


# =========================
# Utils / Security helpers
# =========================
def _safe_in_base(base: str, target: str) -> bool:
    """ตรวจว่าพาธ target อยู่ภายใต้ base จริง (ป้องกัน path traversal)"""
    base = os.path.realpath(base)
    target = os.path.realpath(target)
    try:
        return os.path.commonpath([base]) == os.path.commonpath([base, target])
    except Exception:
        return False


def get_cam_lock(cam_id: str) -> asyncio.Lock:
    """คืนค่า lock ประจำกล้อง (หนึ่งตัวต่อ cam_id)"""
    return camera_locks.setdefault(cam_id, asyncio.Lock())


def save_service_state() -> None:
    """บันทึกข้อมูลกล้องและสถานะ inference ลงไฟล์"""
    cam_ids = set(camera_sources) | set(active_sources) | set(inference_tasks)
    data = {
        str(cam_id): {
            "source": camera_sources.get(cam_id),
            "resolution": list(camera_resolutions.get(cam_id, (None, None))),
            "active_source": active_sources.get(cam_id, ""),
            "inference_running": bool(
                inference_tasks.get(cam_id)
                and not inference_tasks[cam_id].done()
            ),
        }
        for cam_id in cam_ids
    }
    with contextlib.suppress(Exception):
        with open(STATE_FILE, "w") as f:
            json.dump({"cameras": data}, f)


def load_service_state() -> dict[str, dict[str, object]]:
    """โหลดข้อมูลสถานะจากไฟล์"""
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
    """Ensure each ROI dict has a unique string id."""
    if isinstance(rois, list):
        for idx, r in enumerate(rois):
            if isinstance(r, dict) and "id" not in r:
                r["id"] = str(idx + 1)
    return rois


async def restore_service_state() -> None:
    """โหลดสถานะที่บันทึกไว้และเริ่มงาน inference ที่เคยรัน"""
    cams = load_service_state()
    for cam_id, cfg in cams.items():
        camera_sources[cam_id] = cfg.get("source")
        res = cfg.get("resolution") or [None, None]
        camera_resolutions[cam_id] = (res[0], res[1])
        active_sources[cam_id] = cfg.get("active_source", "")
        if cfg.get("inference_running"):
            await perform_start_inference(cam_id, save_state=False)
    save_service_state()


if hasattr(app, "before_serving"):
    app.before_serving(restore_service_state)


# ปิดทุกงาน/กล้องเมื่อแอปหยุด (กันค้าง)
if hasattr(app, "after_serving"):
    @app.after_serving
    async def _shutdown_cleanup():
        # stop inference tasks
        for cam_id in list(inference_tasks.keys()):
            try:
                await stop_camera_task(cam_id, inference_tasks, frame_queues)
            except Exception:
                pass
        # stop roi tasks
        for cam_id in list(roi_tasks.keys()):
            try:
                await stop_camera_task(cam_id, roi_tasks, roi_frame_queues)
            except Exception:
                pass
        # close result queues
        for cam_id, q in list(roi_result_queues.items()):
            try:
                await q.put(None)
            except Exception:
                pass
            roi_result_queues.pop(cam_id, None)
        gc.collect()


# จัดการสัญญาณเพื่อปิดเซิร์ฟเวอร์และเคลียร์ทรัพยากรให้รวดเร็ว
def _handle_sigterm(*_: object) -> None:
    """Handle SIGTERM/SIGINT by initiating app shutdown."""
    try:
        asyncio.create_task(app.shutdown())
    except Exception:
        # หากปิดแอปไม่สำเร็จ พยายามเรียกทำความสะอาดทันที
        try:
            asyncio.create_task(_shutdown_cleanup())
        except Exception:
            pass


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# =========================
# Routes (pages)
# =========================
# ✅ Redirect root ไปหน้า home
@app.route("/")
async def index():
    return redirect("/home")


# ✅ หน้าแรก (dashboard + menu)
@app.route("/home")
async def home():
    return await render_template("home.html")


# ✅ หน้าเลือก ROI
@app.route("/roi")
async def roi_page():
    return await render_template("roi_selection.html")


# ✅ หน้า inference
@app.route("/inference")
async def inference() -> str:
    """แสดงหน้า Inference สำหรับรันโมดูลตาม ROI ที่บันทึกไว้"""
    return await render_template("inference.html")


# ✅ หน้า inference page detection
@app.route("/inference_page")
async def inference_page() -> str:
    """แสดงหน้า Inference Page สำหรับตรวจจับหน้ากระดาษ"""
    return await render_template("inference_page.html")


# =========================
# Inference module loader
# =========================
def load_custom_module(name: str) -> ModuleType | None:
    path = Path("inference_modules") / name / "custom.py"
    if not path.exists():
        return None
    module_name = f"custom_{name}"
    # ลบโมดูลเก่าที่ค้างอยู่เพื่อให้โหลดใหม่ได้ถูกต้องและไม่กินหน่วยความจำ
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
        await asyncio.sleep(0.1)
        return
    frame = await worker.read()
    if frame is None:
        await asyncio.sleep(0.1)
        return
    if np is not None and hasattr(frame, "size") and frame.size == 0:
        await asyncio.sleep(0.1)
        return
    if frame_processor:
        frame = await frame_processor(frame)
    if frame is None:
        return
    try:
        # ลดขนาด buffer ด้วย JPEG quality หากต้องการ
        encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    except Exception:
        await asyncio.sleep(0.1)
        return
    if not encoded or buffer is None:
        await asyncio.sleep(0.1)
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
    module_cache: dict[str, tuple[ModuleType | None, bool, bool]] = {}

    async def process_frame(frame):
        rois = inference_rois.get(cam_id, [])
        forced_group = inference_groups.get(cam_id)
        if not rois:
            return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        save_flag = bool(save_roi_flags.get(cam_id))
        output = None
        best_score = -1.0
        scores: list[dict[str, float | str]] = []
        has_page = False

        # pass 1: evaluate page ROIs to determine best group and draw them
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

        # pass 2: process ROI items that belong to detected group
        if output or scores:
            try:
                q = get_roi_result_queue(cam_id)
                payload = json.dumps({'group': output, 'scores': scores})
                if q.full():
                    q.get_nowait()
                await q.put(payload)
            except Exception:
                pass

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
                    if module:
                        proc = getattr(module, 'process', None)
                        if callable(proc):
                            params = inspect.signature(proc).parameters
                            takes_source = 'source' in params
                            takes_cam_id = 'cam_id' in params
                    module_cache[mod_name] = (module, takes_source, takes_cam_id)
                else:
                    module, takes_source, takes_cam_id = module_entry
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
                        result_text = ''
                        try:
                            args = [roi, r.get('id', str(i)), save_flag]
                            if takes_source:
                                args.append(active_sources.get(cam_id, ''))
                            if takes_cam_id:
                                args.append(cam_id)
                            result = process_fn(*args)
                            if inspect.isawaitable(result):
                                result = await result
                            if isinstance(result, str):
                                result_text = result
                            elif isinstance(result, dict) and 'text' in result:
                                result_text = str(result['text'])
                        except Exception:
                            pass
                        try:
                            _, roi_buf = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            roi_b64 = base64.b64encode(roi_buf).decode('ascii')
                            q = get_roi_result_queue(cam_id)
                            payload = json.dumps({'id': r.get('id', i), 'image': roi_b64, 'text': result_text})
                            if q.full():
                                q.get_nowait()
                            await q.put(payload)
                        except Exception:
                            pass
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
        # ยอมให้ยกเลิกงานได้อย่างปลอดภัยเพื่อป้องกันการค้างของ thread
        pass
    finally:
        # ล้างโมดูล inference ที่โหลดไว้เพื่อลดการใช้หน่วยความจำหลังหยุดงาน
        for mod_name, (module, _, _) in module_cache.items():
            if module is not None:
                sys.modules.pop(f"custom_{mod_name}", None)
        module_cache.clear()
        gc.collect()


async def run_roi_loop(cam_id: str):
    queue = get_roi_frame_queue(cam_id)
    try:
        while True:
            await read_and_queue_frame(cam_id, queue)
    except asyncio.CancelledError:
        # ปิดงานเมื่อถูกยกเลิกเพื่อไม่ให้ไปอ่านกล้องต่อหลังจากถูกสั่งหยุด
        pass


# =========================
# Start/Stop helpers
# =========================
async def start_camera_task(
    cam_id: str,
    task_dict: dict[str, asyncio.Task | None],
    loop_func: Callable[[str], Awaitable[Any]],
):
    """เริ่มงานแบบ asynchronous สำหรับกล้องที่ระบุ (พร้อม lock กัน race)"""
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
                # ป้องกันไม่ให้กล้องค้างเมื่อเปิดไม่สำเร็จ
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
    """หยุดงานที่ใช้กล้องตาม cam_id (พร้อม lock กัน race)"""
    async with get_cam_lock(cam_id):
        task = task_dict.pop(cam_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            status = "stopped"
        else:
            status = "no_task"

        if queue_dict is not None:
            queue = queue_dict.pop(cam_id, None)
            if queue is not None:
                await queue.put(None)
                # drain queue
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

        # ปล่อยกล้องเมื่อไม่มีงานอื่นใช้งานอยู่
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
            frame_queues.pop(cam_id, None)
            roi_frame_queues.pop(cam_id, None)
            del worker
            gc.collect()

        return task, {"status": status, "cam_id": cam_id}, 200


# =========================
# WebSockets
# =========================
# ✅ WebSocket video stream
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


# ✅ WebSocket ROI stream
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

        # หากมี task ใดกำลังรันอยู่ ให้เปิดกล้องใหม่ด้วยค่าที่ตั้ง
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
                # skip sources with unreadable or invalid config
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
# ✅ บันทึก ROI
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


# ✅ โหลด ROI จากไฟล์ล่าสุดของ source
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


# ✅ โหลด ROI ตามพาธที่ระบุใน config (จำกัดให้อยู่ใน data_sources เท่านั้น)
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
# ฟังก์ชันช่วยเริ่มงาน inference เพื่อใช้ซ้ำได้ทั้งจาก route และตอนรีสตาร์ท
async def perform_start_inference(cam_id: str, rois=None, group: str | None = None, save_state: bool = True):
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
    rois = ensure_roi_ids(rois)  # ← ให้มี id เสมอ

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

    # เก็บ ROI ทั้งหมดไว้เพื่อใช้วาดกรอบและประมวลผลเฉพาะที่จำเป็น
    inference_rois[cam_id] = rois
    if group is None:
        inference_groups.pop(cam_id, None)
    else:
        inference_groups[cam_id] = group

    # เคลียร์ผลค้าง
    q = get_roi_result_queue(cam_id)
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break

    await start_camera_task(cam_id, inference_tasks, run_inference_loop)
    if save_state:
        save_service_state()
    return True


# ✅ เริ่มงาน inference
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


# ✅ หยุดงาน inference
@app.route('/stop_inference/<string:cam_id>', methods=["POST"])
async def stop_inference(cam_id: str):
    _, resp, status = await stop_camera_task(cam_id, inference_tasks, frame_queues)
    queue = roi_result_queues.get(cam_id)
    if queue is not None:
        await queue.put(None)
        roi_result_queues.pop(cam_id, None)
    # Clear cached ROI data and flags to free memory
    inference_rois.pop(cam_id, None)
    inference_groups.pop(cam_id, None)
    save_roi_flags.pop(cam_id, None)
    gc.collect()
    save_service_state()
    return jsonify(resp), status


# ✅ เริ่มงาน ROI stream
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


# ✅ หยุดงาน ROI stream
@app.route('/stop_roi_stream/<string:cam_id>', methods=["POST"])
async def stop_roi_stream(cam_id: str):
    _, resp, status = await stop_camera_task(cam_id, roi_tasks, roi_frame_queues)
    return jsonify(resp), status


# ✅ สถานะงาน inference
@app.route('/inference_status/<string:cam_id>', methods=["GET"])
async def inference_status(cam_id: str):
    """คืนค่าความพร้อมของงาน inference"""
    running = inference_tasks.get(cam_id) is not None and not inference_tasks[cam_id].done()
    return jsonify({"running": running, "cam_id": cam_id})


# ✅ สถานะงาน ROI stream
@app.route('/roi_stream_status/<string:cam_id>', methods=["GET"])
async def roi_stream_status(cam_id: str):
    """เช็คว่างาน ROI stream กำลังทำงานอยู่หรือไม่"""
    running = roi_tasks.get(cam_id) is not None and not roi_tasks[cam_id].done()
    source = active_sources.get(cam_id, "")
    return jsonify({"running": running, "cam_id": cam_id, "source": source})


# =========================
# Snapshot (safer Response)
# =========================
# ✅ ส่ง snapshot 1 เฟรม (ใช้ในหน้า inference)
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
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for the web server"
    )
    args = parser.parse_args()
    app.run(port=args.port)
