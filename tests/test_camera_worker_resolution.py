import asyncio


def test_camera_worker_sets_resolution():
    import app

    class DummyCap:
        def __init__(self, src):
            self.src = src
            self.settings = {}

        def isOpened(self):
            return True

        def read(self):
            return True, b"frame"

        def set(self, prop, value):
            self.settings[prop] = value

        def release(self):
            pass

    orig_vc = app.cv2.VideoCapture
    orig_w = getattr(app.cv2, "CAP_PROP_FRAME_WIDTH", 3)
    orig_h = getattr(app.cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    app.cv2.CAP_PROP_FRAME_WIDTH = 3
    app.cv2.CAP_PROP_FRAME_HEIGHT = 4
    app.cv2.VideoCapture = DummyCap
    try:
        loop = asyncio.new_event_loop()
        worker = app.CameraWorker(0, loop, width=640, height=480)
        assert worker._cap.settings[app.cv2.CAP_PROP_FRAME_WIDTH] == 640
        assert worker._cap.settings[app.cv2.CAP_PROP_FRAME_HEIGHT] == 480
    finally:
        app.cv2.VideoCapture = orig_vc
        app.cv2.CAP_PROP_FRAME_WIDTH = orig_w
        app.cv2.CAP_PROP_FRAME_HEIGHT = orig_h


def test_camera_worker_low_latency_env(monkeypatch):
    import camera_worker

    monkeypatch.delenv("CAMERA_FFMPEG_LOW_LATENCY", raising=False)
    assert camera_worker.CameraWorker._resolve_low_latency(None) is True

    monkeypatch.setenv("CAMERA_FFMPEG_LOW_LATENCY", "0")
    assert camera_worker.CameraWorker._resolve_low_latency(None) is False

    monkeypatch.setenv("CAMERA_FFMPEG_LOW_LATENCY", "yes")
    assert camera_worker.CameraWorker._resolve_low_latency(None) is True

    assert camera_worker.CameraWorker._resolve_low_latency(False) is False
    assert camera_worker.CameraWorker._resolve_low_latency(True) is True
