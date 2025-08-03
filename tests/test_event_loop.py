import asyncio
import contextlib
import time

import sys
import types

# สร้างโมดูล quart จำลองเพื่อหลีกเลี่ยงการติดตั้งจริง
quart_stub = types.ModuleType("quart")


class DummyQuart:
    def __init__(self, name):
        self.config = {}

    def route(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator

    def websocket(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator


quart_stub.Quart = DummyQuart
quart_stub.render_template = lambda *a, **k: None
quart_stub.websocket = lambda *a, **k: None
quart_stub.request = None
quart_stub.jsonify = lambda *a, **k: None
quart_stub.send_file = lambda *a, **k: None
quart_stub.redirect = lambda *a, **k: None

sys.modules["quart"] = quart_stub

cv2_stub = types.ModuleType("cv2")


class DummyVideoCapture:
    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return True

    def read(self):
        return True, b"frame"

    def release(self):
        pass


cv2_stub.VideoCapture = DummyVideoCapture
cv2_stub.imencode = lambda *a, **k: (True, b"data")
cv2_stub.rectangle = lambda *a, **k: None
cv2_stub.putText = lambda *a, **k: None

sys.modules["cv2"] = cv2_stub

import app


class DummyCamera:
    def __init__(self):
        self.frames_read = 0

    def read(self):
        self.frames_read += 1
        # จำลองการบล็อก IO หรือการคำนวณที่ใช้เวลานาน
        time.sleep(0.005)
        return True, b"frame"


async def ticker(duration: float = 0.5, interval: float = 0.01) -> int:
    count = 0
    end = time.monotonic() + duration
    while time.monotonic() < end:
        count += 1
        await asyncio.sleep(interval)
    return count


def test_event_loop_responsive():
    async def main():
        app.camera = DummyCamera()
        app.inference_rois = []
        app.active_source = ""
        app.cv2.imencode = lambda ext, frame: (True, b"data")

        loop_task = asyncio.create_task(app.run_inference_loop())
        tick_task = asyncio.create_task(ticker())

        ticks = await tick_task
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

        frames = app.camera.frames_read
        app.camera = None
        return ticks, frames

    ticks, frames = asyncio.run(main())
    # ยืนยันว่า event loop ยังคง responsive
    assert ticks > 20
    # ยืนยันว่ามีการอ่านเฟรมจำนวนมาก
    assert frames >= 9
