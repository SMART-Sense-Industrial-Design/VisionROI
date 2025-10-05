import asyncio
import types
from queue import Queue


def test_camera_worker_read_returns_none_on_timeout():
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker._q = Queue(maxsize=1)
    worker._q.put_nowait(b"new")

    async def run():
        first = await worker.read()
        second = await worker.read(timeout=0.05)
        return first, second

    first, second = asyncio.run(run())
    assert first == b"new"
    assert second is None

    # cleanup so other tests can stub cv2 before importing camera_worker again
    del sys.modules["camera_worker"]


def test_camera_worker_restart_on_opencv_failures(monkeypatch):
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    class DummyCapture:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

        def set(self, *_args, **_kwargs):
            return True

    dummy_cv2 = types.SimpleNamespace(
        VideoCapture=lambda _src: DummyCapture(),
        CAP_PROP_BUFFERSIZE=1,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        IMREAD_COLOR=1,
        imread=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(camera_worker, "cv2", dummy_cv2)
    monkeypatch.setattr(camera_worker.time, "sleep", lambda _seconds: None)

    worker = camera_worker.CameraWorker(0, backend="opencv")
    worker._opencv_restart_fail_threshold = 3
    worker._opencv_restart_time_threshold = 100.0

    restart_counts: list[int] = []

    def fake_restart():
        restart_counts.append(worker._fail_count)
        worker._stop_evt.set()

    worker._restart_backend = fake_restart
    worker._run()

    assert restart_counts
    assert restart_counts[0] >= 3

    # cleanup patched module for other tests
    del sys.modules["camera_worker"]


def test_camera_worker_restart_ffmpeg_after_poll_failures(monkeypatch):
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    class DummyEvent:
        def __init__(self):
            self._flag = False

        def is_set(self):
            return self._flag

        def set(self):
            self._flag = True

    class DummyProc:
        def __init__(self):
            self.stdout = None
            self.stderr = None

        def poll(self):
            return 1

    monkeypatch.setattr(camera_worker.time, "sleep", lambda _seconds: None)

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker.backend = "ffmpeg"
    worker._proc = DummyProc()
    worker._stop_evt = DummyEvent()
    worker._fail_count = 0
    worker._ffmpeg_fail_count = 0
    worker._ffmpeg_restart_fail_threshold = 3
    worker._ffmpeg_restart_time_threshold = 100.0
    worker._ffmpeg_failure_start = None
    worker._last_returncode_logged = None
    worker._logger = types.SimpleNamespace(
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    worker._log_prefix = "[ffmpeg:test]"
    worker._restart_backoff = 0.0
    worker._err_window = []
    worker._opencv_failure_start = None
    worker._last_stderr = []

    restart_counts: list[int] = []

    def fake_restart():
        restart_counts.append(worker._ffmpeg_fail_count)
        worker._stop_evt.set()

    worker._restart_backend = fake_restart

    worker._run()

    assert restart_counts
    assert restart_counts[0] >= worker._ffmpeg_restart_fail_threshold

    # cleanup patched module for other tests
    del sys.modules["camera_worker"]


def test_ffmpeg_command_for_avfoundation():
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker._loglevel = "error"
    worker.low_latency = True
    worker._ff_rtsp_opts = set()
    worker.width = None
    worker.height = None
    worker._logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    worker._log_prefix = "[ffmpeg:test]"
    worker._rtsp_transport_cycle = ["tcp"]
    worker._rtsp_transport_idx = 0
    worker._avfoundation_pixel_format = "nv12"
    worker._avfoundation_pix_fmt_attempts = set()
    worker.src = "avfoundation:0:none"

    cmd = worker._build_ffmpeg_cmd("avfoundation:0:none")

    assert "-pixel_format" in cmd
    pix_idx = cmd.index("-pixel_format")
    assert cmd[pix_idx + 1] == "nv12"
    assert "-reorder_queue_size" not in cmd

    # cleanup patched module for other tests
    del sys.modules["camera_worker"]


def test_ffmpeg_avfoundation_pixel_format_fallback(monkeypatch):
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker.src = "avfoundation:0:none"
    worker._loglevel = "error"
    worker.low_latency = False
    worker._ff_rtsp_opts = set()
    worker.width = None
    worker.height = None
    worker._logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    worker._log_prefix = "[ffmpeg:test]"
    worker._rtsp_transport_cycle = ["tcp"]
    worker._rtsp_transport_idx = 0
    worker._last_stderr = [
        "[avfoundation @ 0x0] Selected pixel format (yuv420p) is not supported by the input device.",
        "[avfoundation @ 0x0] Supported pixel formats:",
        "[avfoundation @ 0x0]   uyvy422",
        "[avfoundation @ 0x0]   yuyv422",
        "[avfoundation @ 0x0]   nv12",
        "[avfoundation @ 0x0]   0rgb",
        "[avfoundation @ 0x0]   bgr0",
    ]
    worker._avfoundation_pixel_format = "yuv420p"
    worker._avfoundation_pix_fmt_attempts = set()

    cmd = worker._build_ffmpeg_cmd("avfoundation:0:none")

    assert "-pixel_format" in cmd
    pix_idx = cmd.index("-pixel_format")
    assert cmd[pix_idx + 1] == "nv12"
    assert worker._avfoundation_pixel_format == "nv12"
    assert worker._last_stderr == []

    # cleanup patched module for other tests
    del sys.modules["camera_worker"]
