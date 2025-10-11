import asyncio
import os
import threading
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


def test_camera_worker_handles_stdout_fileno_error(monkeypatch):
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    # จำลองสถานการณ์ที่ ffmpeg ส่ง stdout ที่ fileno() ใช้งานไม่ได้ชั่วขณะ
    # เพื่อยืนยันว่า worker จะเรียก restart_backend และไม่ค้างรออ่านเฟรม
    class DummyEvent:
        def __init__(self):
            self._flag = False

        def is_set(self):
            return self._flag

        def set(self):
            self._flag = True

    class DummyStdout:
        def fileno(self):
            raise ValueError("bad fd")

    class DummyProc:
        def __init__(self):
            self.stdout = DummyStdout()
            self.stderr = None

        def poll(self):
            return None

    monkeypatch.setattr(camera_worker.time, "sleep", lambda _seconds: None)

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker.backend = "ffmpeg"
    worker._proc = DummyProc()
    worker._stop_evt = DummyEvent()
    worker._fail_count = 0
    worker._ffmpeg_fail_count = 0
    worker._ffmpeg_restart_fail_threshold = 0
    worker._ffmpeg_restart_time_threshold = 0.0
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
    worker._stdout_fd = None
    worker._read_timeout = 0.1
    worker.width = 1
    worker.height = 1
    worker._ffmpeg_pix_fmt = "bgr24"
    worker._q = Queue(maxsize=1)
    worker._clear_frame_queue = lambda: None

    restart_calls: list[int] = []

    def fake_restart():
        restart_calls.append(worker._fail_count)
        worker._stop_evt.set()

    worker._restart_backend = fake_restart

    # numpy stub so reshape is available if ever reached
    camera_worker.np = types.SimpleNamespace(
        frombuffer=lambda *args, **kwargs: None,
        uint8=int,
    )

    worker._run()

    assert restart_calls
    assert restart_calls[0] == worker._fail_count

    del sys.modules["camera_worker"]


def test_camera_worker_recovers_after_ffmpeg_stdout_eof(monkeypatch):
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    monkeypatch.setattr(camera_worker.time, "sleep", lambda _seconds: None)

    class DummyEvent:
        def __init__(self):
            self._flag = False

        def is_set(self):
            return self._flag

        def set(self):
            self._flag = True

    class DummyStdout:
        def __init__(self, fd):
            self._fd = fd

        def fileno(self):
            return self._fd

    class DummyProc:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = None

        def poll(self):
            return None

    offline_r, offline_w = os.pipe()
    online_r, online_w = os.pipe()

    try:
        offline_proc = DummyProc(DummyStdout(offline_r))
        online_proc = DummyProc(DummyStdout(online_r))

        restart_calls: list[int] = []

        worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
        worker.backend = "ffmpeg"
        worker._proc = offline_proc
        worker._stdout_fd = offline_r
        worker._stop_evt = DummyEvent()
        worker._logger = types.SimpleNamespace(
            error=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        worker._log_prefix = "[ffmpeg:test]"
        worker._ffmpeg_pix_fmt = "bgr24"
        worker.width = 1
        worker.height = 1
        worker.read_interval = 0.0
        worker._read_timeout = 0.1
        worker._q = Queue(maxsize=1)
        worker._clear_frame_queue = lambda: None
        worker._ffmpeg_restart_fail_threshold = 0
        worker._ffmpeg_restart_time_threshold = 0.0
        worker._ffmpeg_fail_count = 0
        worker._ffmpeg_failure_start = None
        worker._fail_count = 0
        worker._err_window = []
        worker._restart_backoff = 0.0
        worker._no_video_data_since = None
        worker._last_no_video_log = 0.0
        worker._last_stderr = []
        worker._opencv_failure_start = None
        worker._last_returncode_logged = None

        def fake_restart_backend():
            restart_calls.append(worker._fail_count)
            worker._proc = online_proc
            worker._stdout_fd = online_r

        worker._restart_backend = fake_restart_backend

        def fake_select(rlist, _wlist, _xlist, _timeout):
            return rlist, [], []

        monkeypatch.setattr(camera_worker.select, "select", fake_select)

        frame_bytes = bytearray(b"\x01\x02\x03")

        def fake_os_read(fd, n):
            if fd == offline_r:
                return b""
            if fd == online_r:
                if not frame_bytes:
                    return b""
                chunk = bytes(frame_bytes[:n])
                del frame_bytes[:n]
                return chunk
            raise AssertionError("unexpected fd")

        monkeypatch.setattr(camera_worker.os, "read", fake_os_read)

        monkeypatch.setattr(
            camera_worker,
            "np",
            types.SimpleNamespace(
                frombuffer=lambda view, _dtype: types.SimpleNamespace(
                    reshape=lambda _shape: bytes(view)
                ),
                uint8=int,
            ),
        )

        worker_thread = threading.Thread(target=worker._run, daemon=True)
        worker_thread.start()

        frame = worker._q.get(timeout=1.0)
        worker._stop_evt.set()
        worker_thread.join(timeout=1.0)

        assert frame == b"\x01\x02\x03"
        assert restart_calls
    finally:
        os.close(offline_r)
        os.close(offline_w)
        os.close(online_r)
        os.close(online_w)

    del sys.modules["camera_worker"]
