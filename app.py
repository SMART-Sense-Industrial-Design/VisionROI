from quart import Quart, render_template, websocket, request, jsonify, send_file, redirect
import asyncio
import cv2
import base64
import json
from datetime import datetime
import os
import shutil

frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
inference_task: asyncio.Task | None = None

app = Quart(__name__)
camera = cv2.VideoCapture(0)

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

async def run_inference_loop():
    while True:
        success, frame = camera.read()
        if not success:
            await asyncio.sleep(0.1)
            continue
        # TODO: perform inference and save results
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await frame_queue.put(frame_b64)
        await asyncio.sleep(0.05)


# ✅ WebSocket video stream
@app.websocket('/ws')
async def ws():
    while True:
        frame_b64 = await frame_queue.get()
        await websocket.send(frame_b64)

@app.route("/set_camera", methods=["POST"])
async def set_camera():
    global camera
    data = await request.get_json()
    source_val = data.get("source", "")
    if camera and camera.isOpened():
        camera.release()
    try:
        source = int(source_val)
    except ValueError:
        source = source_val
    camera = cv2.VideoCapture(source)
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
    if not name or not source or model is None or label is None:
        return jsonify({"status": "error", "message": "missing data"}), 400

    source_dir = os.path.join("sources", name)
    try:
        os.makedirs(source_dir, exist_ok=False)
    except FileExistsError:
        return jsonify({"status": "error", "message": "name exists"}), 400

    try:
        await model.save(os.path.join(source_dir, "model.onnx"))
        await label.save(os.path.join(source_dir, "classes.txt"))
        config = {
            "name": name,
            "source": source,
            "model": "model.onnx",
            "label": "classes.txt",
        }
        with open(os.path.join(source_dir, "config.json"), "w") as f:
            json.dump(config, f)
    except Exception:
        shutil.rmtree(source_dir, ignore_errors=True)
        return jsonify({"status": "error", "message": "save failed"}), 500

    return jsonify({"status": "created"})


# ✅ เริ่มงาน inference
@app.route('/start_inference', methods=["POST"])
async def start_inference():
    global inference_task
    if inference_task is None or inference_task.done():
        inference_task = asyncio.create_task(run_inference_loop())
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


# ✅ หยุดงาน inference
@app.route('/stop_inference', methods=["POST"])
async def stop_inference():
    global inference_task
    if inference_task is not None and not inference_task.done():
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass
        inference_task = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "no_task"})

# ✅ สถานะงาน inference
# เพิ่ม endpoint สำหรับตรวจสอบว่างาน inference กำลังทำงานอยู่หรือไม่
@app.route('/inference_status', methods=["GET"])
async def inference_status():
    """คืนค่าความพร้อมของงาน inference"""
    running = inference_task is not None and not inference_task.done()
    return jsonify({"running": running})

# ✅ บันทึก ROI
@app.route("/save_roi", methods=["POST"])
async def save_roi():
    data = await request.get_json()
    rois = data.get("rois", [])
    filename = f"rois_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(rois, f, indent=2)
    return jsonify({"status": "saved", "filename": filename})

# ✅ โหลด ROI จากไฟล์ล่าสุด
@app.route("/load_roi")
async def load_roi():
    roi_files = [f for f in os.listdir() if f.startswith("rois_") and f.endswith(".json")]
    if not roi_files:
        return jsonify({"rois": [], "filename": "None"})
    latest_file = sorted(roi_files)[-1]
    with open(latest_file, "r") as f:
        rois = json.load(f)
    return jsonify({"rois": rois, "filename": latest_file})

# ✅ ส่ง snapshot 1 เฟรม (ใช้ในหน้า inference)
@app.route("/ws_snapshot")
async def ws_snapshot():
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
