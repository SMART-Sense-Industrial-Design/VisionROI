from quart import Quart, render_template, websocket, request, jsonify, send_file, redirect
import asyncio
import cv2
import json
import shutil
import importlib.util
import os, sys
from types import ModuleType
from pathlib import Path
import contextlib
import inspect
from typing import Callable, Awaitable, Any

# เก็บสถานะของกล้องแต่ละตัวในรูปแบบ dict
cameras: dict[int, cv2.VideoCapture | None] = {}
camera_sources: dict[int, int | str] = {}
camera_locks: dict[int, asyncio.Lock] = {}
frame_queues: dict[int, asyncio.Queue[bytes]] = {}
inference_tasks: dict[int, asyncio.Task | None] = {}
roi_frame_queues: dict[int, asyncio.Queue[bytes | None]] = {}
roi_tasks: dict[int, asyncio.Task | None] = {}
inference_rois: dict[int, list[dict]] = {}
active_sources: dict[int, str] = {}

app = Quart(__name__)
# กำหนดเพดานขนาดไฟล์ที่เซิร์ฟเวอร์ยอมรับ (100 MB)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_ROI_DIR = os.path.realpath("data_sources")

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
async def inference():
    return await render_template("inference.html")

# ✅ Fragment endpoints สำหรับโหลดคอนเทนต์แบบไม่ใช้ layout
@app.route("/fragment/home")
async def fragment_home():
    return await render_template("fragments/home.html")

@app.route("/fragment/create_source")
async def fragment_create_source():
    return await render_template("fragments/create_source.html")

@app.route("/fragment/roi")
async def fragment_roi():
    return await render_template("fragments/roi_selection.html")

@app.route("/fragment/inference")
async def fragment_inference():
    return await render_template("fragments/inference.html")


def load_custom_module(name: str) -> ModuleType | None:
    path = os.path.join("data_sources", name, "custom.py")
    if not os.path.exists(path):
        return None
    module_name = f"custom_{name}"
    # ลบโมดูลเก่าที่ค้างอยู่เพื่อให้โหลดใหม่ได้ถูกต้องและไม่กินหน่วยความจำ
    if module_name in sys.modules:
        del sys.modules[module_name]
        importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_frame_queue(cam_id: int) -> asyncio.Queue[bytes]:
    return frame_queues.setdefault(cam_id, asyncio.Queue(maxsize=1))


def get_roi_frame_queue(cam_id: int) -> asyncio.Queue[bytes | None]:
    return roi_frame_queues.setdefault(cam_id, asyncio.Queue(maxsize=1))


async def read_and_queue_frame(
    cam_id: int, queue: asyncio.Queue[bytes], frame_processor=None
) -> None:
    lock = camera_locks.setdefault(cam_id, asyncio.Lock())
    async with lock:
        camera = cameras.get(cam_id)
        if camera is None or not getattr(camera, "isOpened", lambda: True)():
            success = False
        else:
            success, frame = await asyncio.to_thread(camera.read)
    if camera is None or not success:
        await asyncio.sleep(0.1)
        return
    if frame_processor:
        frame = await frame_processor(frame)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes() if hasattr(buffer, "tobytes") else buffer
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(frame_bytes)
    await asyncio.sleep(0.05)


async def run_inference_loop(cam_id: int):
    custom_module = load_custom_module(active_sources.get(cam_id, ""))
    process_fn = getattr(custom_module, "process", None) if custom_module else None
    process_has_id = False
    if process_fn:
        try:
            process_has_id = len(inspect.signature(process_fn).parameters) >= 2
        except (ValueError, TypeError):
            process_has_id = False

    async def process_frame(frame):
        rois = inference_rois.get(cam_id, [])
        if not rois:
            if process_fn:
                try:
                    if process_has_id:
                        await asyncio.to_thread(process_fn, frame, None)
                    else:
                        await asyncio.to_thread(process_fn, frame)
                except Exception:
                    pass
        else:
            for i, r in enumerate(rois):
                x, y, w, h = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])
                roi = frame[y:y + h, x:x + w]
                if process_fn:
                    try:
                        if process_has_id:
                            await asyncio.to_thread(process_fn, roi, r.get("id", str(i)))
                        else:
                            await asyncio.to_thread(process_fn, roi)
                    except Exception:
                        pass
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    str(r.get("id", i + 1)),
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    queue = get_frame_queue(cam_id)
    while True:
        await read_and_queue_frame(cam_id, queue, process_frame)


async def run_roi_loop(cam_id: int):
    queue = get_roi_frame_queue(cam_id)
    while True:
        await read_and_queue_frame(cam_id, queue)


# ฟังก์ชัน generic สำหรับเริ่มและหยุดงานที่ใช้กล้อง
async def start_camera_task(
    cam_id: int,
    task_dict: dict[int, asyncio.Task | None],
    loop_func: Callable[[int], Awaitable[Any]],
):
    """เริ่มงานแบบ asynchronous สำหรับกล้องที่ระบุ"""
    task = task_dict.get(cam_id)
    if task is not None and not task.done():
        return task, {"status": "already_running", "cam_id": cam_id}, 200
    camera = cameras.get(cam_id)
    if camera is None or not camera.isOpened():
        src = camera_sources.get(cam_id, 0)
        camera = cv2.VideoCapture(src)
        if not camera.isOpened():
            cameras[cam_id] = None
            return task, {"status": "error", "message": "open_failed", "cam_id": cam_id}, 400
        cameras[cam_id] = camera
    task = asyncio.create_task(loop_func(cam_id))
    task_dict[cam_id] = task
    return task, {"status": "started", "cam_id": cam_id}, 200


async def stop_camera_task(
    cam_id: int,
    task_dict: dict[int, asyncio.Task | None],
    queue_dict: dict[int, asyncio.Queue[bytes | None]] | None = None,
):
    """หยุดงานที่ใช้กล้องตาม cam_id"""
    task = task_dict.get(cam_id)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        task_dict[cam_id] = None
        status = "stopped"
    else:
        status = "no_task"
    if queue_dict is not None:
        queue = queue_dict.get(cam_id)
        if queue is not None:
            await queue.put(None)
    # ปล่อยกล้องเมื่อไม่มีงานอื่นใช้งานอยู่
    inf_task = inference_tasks.get(cam_id)
    roi_task = roi_tasks.get(cam_id)
    camera = cameras.get(cam_id)
    if (
        (inf_task is None or inf_task.done())
        and (roi_task is None or roi_task.done())
        and camera
        and camera.isOpened()
    ):
        lock = camera_locks.setdefault(cam_id, asyncio.Lock())
        async with lock:
            camera.release()
        cameras[cam_id] = None
    return task_dict.get(cam_id), {"status": status, "cam_id": cam_id}, 200


# ✅ WebSocket video stream
@app.websocket('/ws/<int:cam_id>')
async def ws(cam_id: int):
    queue = get_frame_queue(cam_id)
    while True:
        frame_bytes = await queue.get()
        if frame_bytes is None:
            break
        await websocket.send(frame_bytes)


# ✅ WebSocket ROI stream
@app.websocket('/ws_roi/<int:cam_id>')
async def ws_roi(cam_id: int):
    queue = get_roi_frame_queue(cam_id)
    while True:
        frame_bytes = await queue.get()
        if frame_bytes is None:
            await websocket.close(code=1000)
            break
        await websocket.send(frame_bytes)


@app.route("/set_camera/<int:cam_id>", methods=["POST"])
async def set_camera(cam_id: int):
    data = await request.get_json()
    active_sources[cam_id] = data.get("name", "")
    source_val = data.get("source", "")
    camera = cameras.get(cam_id)
    if camera and camera.isOpened():
        camera.release()
        cameras[cam_id] = None
    try:
        camera_sources[cam_id] = int(source_val)
    except ValueError:
        camera_sources[cam_id] = source_val
    if (
        (inference_tasks.get(cam_id) and not inference_tasks[cam_id].done())
        or (roi_tasks.get(cam_id) and not roi_tasks[cam_id].done())
    ):
        cam = cv2.VideoCapture(camera_sources[cam_id])
        if not cam.isOpened():
            return jsonify({"status": "error", "cam_id": cam_id}), 400
        cameras[cam_id] = cam
    return jsonify({"status": "ok", "cam_id": cam_id})


@app.route("/create_source", methods=["GET", "POST"])
async def create_source():
    if request.method == "GET":
        return await render_template("create_source.html")

    form = await request.form
    files = await request.files
    name = form.get("name", "").strip()
    source = form.get("source", "").strip()
    model = files.get("model")
    label = files.get("label")
    if not name or not source:
        return jsonify({"status": "error", "message": "missing data"}), 400

    source_dir = f"data_sources/{name}"
    try:
        os.makedirs(f"data_sources/{name}", exist_ok=False)
    except FileExistsError:
        return jsonify({"status": "error", "message": "name exists"}), 400

    try:
        config = {
            "name": name,
            "source": source,
            "model": "",
            "label": "",
            "rois": "rois.json",
        }
        if model:
            await model.save(os.path.join(source_dir, "model.onnx"))
            config["model"] = "model.onnx"
        if label:
            await label.save(os.path.join(source_dir, "classes.txt"))
            config["label"] = "classes.txt"
        rois_path = os.path.join(source_dir, "rois.json")
        with open(rois_path, "w") as f:
            f.write("[]")
        custom_path = os.path.join(source_dir, "custom.py")
        with open(custom_path, "w") as f:
            f.write(
                "def process(frame):\n"
                "    \"\"\"รับเฟรมเต็มและภาพ ROI ที่ตัดแล้ว\"\"\"\n"
                "    # เขียนโค้ดประมวลผลตามต้องการ เช่น OCR\n"
                "    return frame\n"
            )
        with open(os.path.join(source_dir, "config.json"), "w") as f:
            json.dump(config, f)
    except Exception:
        shutil.rmtree(source_dir, ignore_errors=True)
        return jsonify({"status": "error", "message": "save failed"}), 500

    return jsonify({"status": "created"})


# ✅ เริ่มงาน inference
@app.route('/start_inference/<int:cam_id>', methods=["POST"])
async def start_inference(cam_id: int):
    if roi_tasks.get(cam_id) and not roi_tasks[cam_id].done():
        return jsonify({"status": "roi_running", "cam_id": cam_id}), 400
    data = await request.get_json() or {}
    inference_rois[cam_id] = data.get("rois", [])
    inference_tasks[cam_id], resp, status = await start_camera_task(
        cam_id, inference_tasks, run_inference_loop
    )
    return jsonify(resp), status


# ✅ หยุดงาน inference
@app.route('/stop_inference/<int:cam_id>', methods=["POST"])
async def stop_inference(cam_id: int):
    inference_tasks[cam_id], resp, status = await stop_camera_task(
        cam_id, inference_tasks
    )
    return jsonify(resp), status


# ✅ เริ่มงาน ROI stream
@app.route('/start_roi_stream/<int:cam_id>', methods=["POST"])
async def start_roi_stream(cam_id: int):
    if inference_tasks.get(cam_id) and not inference_tasks[cam_id].done():
        return jsonify({"status": "inference_running", "cam_id": cam_id}), 400
    queue = get_roi_frame_queue(cam_id)
    while not queue.empty():
        queue.get_nowait()
    roi_tasks[cam_id], resp, status = await start_camera_task(
        cam_id, roi_tasks, run_roi_loop
    )
    return jsonify(resp), status


# ✅ หยุดงาน ROI stream
@app.route('/stop_roi_stream/<int:cam_id>', methods=["POST"])
async def stop_roi_stream(cam_id: int):
    roi_tasks[cam_id], resp, status = await stop_camera_task(
        cam_id, roi_tasks, roi_frame_queues
    )
    return jsonify(resp), status

# ✅ สถานะงาน inference
# เพิ่ม endpoint สำหรับตรวจสอบว่างาน inference กำลังทำงานอยู่หรือไม่
@app.route('/inference_status/<int:cam_id>', methods=["GET"])
async def inference_status(cam_id: int):
    """คืนค่าความพร้อมของงาน inference"""
    running = inference_tasks.get(cam_id) is not None and not inference_tasks[cam_id].done()
    return jsonify({"running": running, "cam_id": cam_id})


@app.route("/data_sources")
async def list_sources():
    base_dir = Path(__file__).resolve().parent / "data_sources"
    try:
        names = [d.name for d in base_dir.iterdir() if d.is_dir()]
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.route("/source_list", methods=["GET"])
async def source_list():
    base_dir = Path(__file__).resolve().parent / "data_sources"
    result = []
    try:
        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            cfg_path = d / "config.json"
            if not cfg_path.exists():
                continue
            with cfg_path.open("r") as f:
                cfg = json.load(f)
            result.append(
                {
                    "name": cfg.get("name", d.name),
                    "source": cfg.get("source", ""),
                    "model": cfg.get("model", ""),
                    "label": cfg.get("label", ""),
                }
            )
    except FileNotFoundError:
        pass
    return jsonify(result)


@app.route("/source_config")
async def source_config():
    name = request.args.get("name", "")
    path = os.path.join("data_sources", name, "config.json")
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": "not found"}), 404
    with open(path, "r") as f:
        cfg = json.load(f)
    return jsonify(cfg)


@app.route("/delete_source/<name>", methods=["DELETE"])
async def delete_source(name: str):
    directory = os.path.join("data_sources", name)
    if not os.path.exists(directory):
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

"""ROI save and load helpers"""

# ✅ บันทึก ROI
@app.route("/save_roi", methods=["POST"])
async def save_roi():
    data = await request.get_json()
    rois = data.get("rois", [])
    for idx, r in enumerate(rois):
        if isinstance(r, dict) and "id" not in r:
            r["id"] = str(idx + 1)
    path = request.args.get("path", "")
    base_dir = ALLOWED_ROI_DIR
    if path:
        full_path = os.path.realpath(os.path.join(base_dir, path))
        if not full_path.startswith(base_dir + os.sep):
            return jsonify({"status": "error", "message": "path outside allowed directory"}), 400
        dir_path = os.path.dirname(full_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(rois, f, indent=2)
        return jsonify({"status": "saved", "filename": full_path})

    name = data.get("source", "")
    if not name:
        return jsonify({"status": "error", "message": "missing source"}), 400
    directory = os.path.realpath(os.path.join(base_dir, name))
    if not directory.startswith(base_dir + os.sep):
        return jsonify({"status": "error", "message": "invalid source path"}), 400
    config_path = os.path.join(directory, "config.json")
    if not os.path.exists(config_path):
        return jsonify({"status": "error", "message": "config not found"}), 404
    with open(config_path, "r") as f:
        cfg = json.load(f)
    roi_file = cfg.get("rois", "rois.json")
    roi_path = os.path.realpath(os.path.join(directory, roi_file))
    if not roi_path.startswith(directory + os.sep):
        return jsonify({"status": "error", "message": "invalid ROI filename"}), 400
    with open(roi_path, "w") as f:
        json.dump(rois, f, indent=2)
    return jsonify({"status": "saved", "filename": roi_path})

# ✅ โหลด ROI จากไฟล์ล่าสุดของ source
@app.route("/load_roi/<name>")
async def load_roi(name: str):
    directory = os.path.join("data_sources", name)
    config_path = os.path.join(directory, "config.json")
    if not os.path.exists(config_path):
        return jsonify({"rois": [], "filename": "None"})
    with open(config_path, "r") as f:
        cfg = json.load(f)
    roi_file = cfg.get("rois", "rois.json")
    roi_path = os.path.join(directory, roi_file)
    if not os.path.exists(roi_path):
        return jsonify({"rois": [], "filename": roi_file})
    with open(roi_path, "r") as f:
        rois = json.load(f)
    if isinstance(rois, list):
        for idx, r in enumerate(rois):
            if isinstance(r, dict) and "id" not in r:
                r["id"] = str(idx + 1)
    return jsonify({"rois": rois, "filename": roi_file})

# ✅ โหลด ROI ตามพาธที่ระบุใน config
@app.route("/load_roi_file")
async def load_roi_file():
    path = request.args.get("path", "")
    if not path or not os.path.exists(path):
        return jsonify({"rois": [], "filename": "None"})
    with open(path, "r") as f:
        rois = json.load(f)
    if isinstance(rois, list):
        for idx, r in enumerate(rois):
            if isinstance(r, dict) and "id" not in r:
                r["id"] = str(idx + 1)
    return jsonify({"rois": rois, "filename": os.path.basename(path)})

# ✅ ส่ง snapshot 1 เฟรม (ใช้ในหน้า inference)
@app.route("/ws_snapshot/<int:cam_id>")
async def ws_snapshot(cam_id: int):
    camera = cameras.get(cam_id)
    if camera is None or not camera.isOpened():
        return "Camera not initialized", 400
    success, frame = camera.read()
    if not success:
        return "Camera error", 500
    _, buffer = cv2.imencode('.jpg', frame)
    return await send_file(
        bytes(buffer),
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="snapshot.jpg"
    )

if __name__ == "__main__":
    app.run()
