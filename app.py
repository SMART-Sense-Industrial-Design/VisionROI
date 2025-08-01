from quart import Quart, render_template, websocket, request, jsonify, send_file, redirect
import asyncio
import cv2
import base64
import json
from datetime import datetime
import os

app = Quart(__name__)
camera = cv2.VideoCapture(0)
streaming = False  # ควบคุมการเริ่ม/หยุด stream

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

# ✅ WebSocket video stream
@app.websocket('/ws')
async def ws():
    global streaming
    streaming = True
    while streaming:
        success, frame = camera.read()
        if not success:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(frame_b64)
        await asyncio.sleep(0.05)

# ✅ หยุด stream
@app.route('/stop_stream', methods=["POST"])
async def stop_stream():
    global streaming
    streaming = False
    return jsonify({"status": "stopped"})

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
