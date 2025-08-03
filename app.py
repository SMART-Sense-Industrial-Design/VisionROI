from quart import Quart, render_template, websocket, request, jsonify, send_file, redirect
import asyncio
import cv2
import base64
import json
import shutil
import importlib.util
import os, sys
from types import ModuleType

frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
inference_task: asyncio.Task | None = None
roi_frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
roi_task: asyncio.Task | None = None
inference_rois: list[dict] = []
active_source: str = ""

app = Quart(__name__)
# กำหนดเพดานขนาดไฟล์ที่เซิร์ฟเวอร์ยอมรับ (100 MB)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
camera: cv2.VideoCapture | None = None
camera_source: int | str = 0
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
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

async def run_inference_loop():
    custom_module = load_custom_module(active_source)
    while True:
        if camera is None:
            await asyncio.sleep(0.1)
            continue
        success, frame = camera.read()
        if not success:
            await asyncio.sleep(0.1)
            continue
        if not inference_rois:
            if custom_module and hasattr(custom_module, "process"):
                try:
                    await asyncio.to_thread(custom_module.process, frame)

                except Exception:
                    pass
        else:
            for i, r in enumerate(inference_rois):
                x, y, w, h = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])
                roi = frame[y:y + h, x:x + w]
                if custom_module and hasattr(custom_module, "process"):
                    try:
                        await asyncio.to_thread(custom_module.process, roi)
                    except Exception:
                        pass
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ROI {i + 1}",
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await frame_queue.put(frame_b64)
        await asyncio.sleep(0.05)


async def run_roi_loop():
    while True:
        if camera is None:
            await asyncio.sleep(0.1)
            continue
        success, frame = camera.read()
        if not success:
            await asyncio.sleep(0.1)
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        if roi_frame_queue.full():
            try:
                roi_frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await roi_frame_queue.put(frame_b64)
        await asyncio.sleep(0.05)


# ✅ WebSocket video stream
@app.websocket('/ws')
async def ws():
    while True:
        frame_b64 = await frame_queue.get()
        await websocket.send(frame_b64)


# ✅ WebSocket ROI stream
@app.websocket('/ws_roi')
async def ws_roi():
    while True:
        frame_b64 = await roi_frame_queue.get()
        await websocket.send(frame_b64)

@app.route("/set_camera", methods=["POST"])
async def set_camera():
    global camera, camera_source, inference_task, roi_task, active_source
    data = await request.get_json()
    active_source = data.get("name", "")
    source_val = data.get("source", "")
    if camera and camera.isOpened():
        camera.release()
        camera = None
    try:
        camera_source = int(source_val)
    except ValueError:
        camera_source = source_val
    if (
        inference_task is not None and not inference_task.done()
    ) or (
        roi_task is not None and not roi_task.done()
    ):
        camera = cv2.VideoCapture(camera_source)
        if not camera.isOpened():
            return jsonify({"status": "error"}), 400
    return jsonify({"status": "ok"})


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
@app.route('/start_inference', methods=["POST"])
async def start_inference():
    global inference_task, camera, roi_task, inference_rois
    if roi_task is not None and not roi_task.done():
        return jsonify({"status": "roi_running"}), 400
    data = await request.get_json() or {}
    inference_rois = data.get("rois", [])
    if inference_task is None or inference_task.done():
        camera = cv2.VideoCapture(camera_source)
        if not camera.isOpened():
            camera = None
            return jsonify({"status": "error", "message": "open_failed"}), 400
        inference_task = asyncio.create_task(run_inference_loop())
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


# ✅ หยุดงาน inference
@app.route('/stop_inference', methods=["POST"])
async def stop_inference():
    global inference_task, camera
    if inference_task is not None and not inference_task.done():
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass
        inference_task = None
        if camera and camera.isOpened():
            camera.release()
            camera = None
        return jsonify({"status": "stopped"})
    if camera and camera.isOpened():
        camera.release()
        camera = None
    return jsonify({"status": "no_task"})


# ✅ เริ่มงาน ROI stream
@app.route('/start_roi_stream', methods=["POST"])
async def start_roi_stream():
    global roi_task, camera, inference_task
    if inference_task is not None and not inference_task.done():
        return jsonify({"status": "inference_running"}), 400
    if roi_task is None or roi_task.done():
        camera = cv2.VideoCapture(camera_source)
        if not camera.isOpened():
            camera = None
            return jsonify({"status": "error", "message": "open_failed"}), 400
        roi_task = asyncio.create_task(run_roi_loop())
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


# ✅ หยุดงาน ROI stream
@app.route('/stop_roi_stream', methods=["POST"])
async def stop_roi_stream():
    global roi_task, camera
    if roi_task is not None and not roi_task.done():
        roi_task.cancel()
        try:
            await roi_task
        except asyncio.CancelledError:
            pass
        roi_task = None
        if camera and camera.isOpened():
            camera.release()
            camera = None
        return jsonify({"status": "stopped"})
    if camera and camera.isOpened():
        camera.release()
        camera = None
    return jsonify({"status": "no_task"})

# ✅ สถานะงาน inference
# เพิ่ม endpoint สำหรับตรวจสอบว่างาน inference กำลังทำงานอยู่หรือไม่
@app.route('/inference_status', methods=["GET"])
async def inference_status():
    """คืนค่าความพร้อมของงาน inference"""
    running = inference_task is not None and not inference_task.done()
    return jsonify({"running": running})


@app.route("/data_sources")
async def list_sources():
    try:
        names = [d for d in os.listdir("data_sources") if os.path.isdir(os.path.join("data_sources", d))]
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.route("/source_list", methods=["GET"])
async def source_list():
    result = []
    try:
        for d in os.listdir("data_sources"):
            path = os.path.join("data_sources", d)
            if not os.path.isdir(path):
                continue
            cfg_path = os.path.join(path, "config.json")
            if not os.path.exists(cfg_path):
                continue
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            result.append(
                {
                    "name": cfg.get("name", d),
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
    return jsonify({"rois": rois, "filename": roi_file})

# ✅ โหลด ROI ตามพาธที่ระบุใน config
@app.route("/load_roi_file")
async def load_roi_file():
    path = request.args.get("path", "")
    if not path or not os.path.exists(path):
        return jsonify({"rois": [], "filename": "None"})
    with open(path, "r") as f:
        rois = json.load(f)
    return jsonify({"rois": rois, "filename": os.path.basename(path)})

# ✅ ส่ง snapshot 1 เฟรม (ใช้ในหน้า inference)
@app.route("/ws_snapshot")
async def ws_snapshot():
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
