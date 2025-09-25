import logging
import threading
from collections import deque
from queue import Queue


def test_ffmpeg_worker_restarts_on_oserror(monkeypatch):
    import camera_worker

    class DummyStdout:
        def fileno(self):
            return 42

        def close(self):
            pass

    class DummyProc:
        def __init__(self):
            self.stdout = DummyStdout()
            self.stderr = None

        def poll(self):
            return None

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker.backend = "ffmpeg"
    worker._stop_evt = threading.Event()
    worker._logger = logging.getLogger("test_camera_worker")
    worker._log_prefix = "[ffmpeg:test]"
    worker.width = 1
    worker.height = 1
    worker._ffmpeg_pix_fmt = "bgr24"
    worker._read_timeout = 0.1
    worker._proc = DummyProc()
    worker._stdout_fd = worker._proc.stdout.fileno()
    worker._q = Queue(maxsize=1)
    worker._last_frame = None
    worker._fail_count = 0
    worker._err_window = deque()
    worker._err_window_secs = 5.0
    worker._last_err_prune = 0.0
    worker._restart_backoff = 0.0
    worker._last_returncode_logged = None
    worker._next_resolution_probe = 0.0
    worker._last_stderr = deque(maxlen=200)
    worker.read_interval = 0.0

    restart_event = threading.Event()

    def fake_restart(self):
        restart_event.set()
        self._proc = DummyProc()
        self._stdout_fd = self._proc.stdout.fileno()

    worker._restart_backend = fake_restart.__get__(worker, camera_worker.CameraWorker)

    def fake_flush(self, fd, remaining):
        return True

    worker._flush_partial_ffmpeg_frame = fake_flush.__get__(
        worker, camera_worker.CameraWorker
    )

    def fake_reshape(self, buffer):
        return b"frame"

    worker._reshape_ffmpeg_frame = fake_reshape.__get__(
        worker, camera_worker.CameraWorker
    )

    monkeypatch.setattr(
        camera_worker.select,
        "select",
        lambda fds, *_: ([fds[0]], [], []),
    )
    monkeypatch.setattr(camera_worker.time, "sleep", lambda *_: None)

    read_calls = {"count": 0}

    def fake_read(fd, remaining):
        read_calls["count"] += 1
        if read_calls["count"] == 1:
            raise OSError("boom")
        worker._stop_evt.set()
        return b"\x00" * remaining

    monkeypatch.setattr(camera_worker.os, "read", fake_read)

    thread = threading.Thread(target=worker._run, daemon=True)
    thread.start()

    assert restart_event.wait(1.0)

    thread.join(1.0)

    assert read_calls["count"] >= 2
    assert not thread.is_alive()
