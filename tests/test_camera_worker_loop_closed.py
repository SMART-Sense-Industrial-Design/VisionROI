import asyncio
import time


def test_camera_worker_thread_stops_when_loop_closed():
    import app

    class DummyCap:
        def __init__(self, src):
            self.frames_read = 0

        def isOpened(self):
            return True

        def read(self):
            self.frames_read += 1
            time.sleep(0.005)
            return True, b"frame"

        def release(self):
            pass

    orig_vc = app.cv2.VideoCapture
    app.cv2.VideoCapture = DummyCap
    try:
        loop = asyncio.new_event_loop()
        worker = app.CameraWorker(0, loop)
        assert worker.start()
        time.sleep(0.05)
        loop.close()
        worker._thread.join(timeout=0.5)
        assert not worker._thread.is_alive()
    finally:
        app.cv2.VideoCapture = orig_vc
