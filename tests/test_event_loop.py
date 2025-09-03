import asyncio
import contextlib
import time

from .stubs import stub_cv2, stub_quart


quart_stub = stub_quart()
cv2_stub = stub_cv2()


class DummyCamera:
    def __init__(self, *a, **k):
        self.frames_read = 0

    def isOpened(self):
        return True

    def read(self):
        self.frames_read += 1
        # จำลองการบล็อก IO หรือการคำนวณที่ใช้เวลานาน
        time.sleep(0.005)
        return True, b"frame"

    def release(self):
        pass


cv2_stub.VideoCapture = DummyCamera
cv2_stub.imencode = lambda *a, **k: (True, b"data")
cv2_stub.rectangle = lambda *a, **k: None
cv2_stub.putText = lambda *a, **k: None
cv2_stub.resize = lambda img, dsize, fx=0, fy=0, **k: img
cv2_stub.destroyAllWindows = lambda: None

import app


async def ticker(duration: float = 0.5, interval: float = 0.01) -> int:
    count = 0
    end = time.monotonic() + duration
    while time.monotonic() < end:
        count += 1
        await asyncio.sleep(interval)
    return count


def test_event_loop_responsive():
    async def main():
        worker = app.CameraWorker(0, asyncio.get_running_loop())
        assert worker.start()
        app.camera_workers["0"] = worker
        app.inference_rois["0"] = []
        app.active_sources["0"] = ""
        app.cv2.imencode = lambda ext, frame: (True, b"data")

        loop_task = asyncio.create_task(app.run_inference_loop("0"))
        tick_task = asyncio.create_task(ticker())

        ticks = await tick_task
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

        frames = worker._cap.frames_read
        await worker.stop()
        app.camera_workers["0"] = None
        return ticks, frames

    ticks, frames = asyncio.run(main())
    # ยืนยันว่า event loop ยังคง responsive
    assert ticks > 20
    # ยืนยันว่ามีการอ่านเฟรมจำนวนมาก
    assert frames >= 9
