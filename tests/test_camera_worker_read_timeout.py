import asyncio
import types
from queue import Queue


def test_camera_worker_read_returns_none_on_timeout():
    import sys
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
