import asyncio
from queue import Queue

def test_camera_worker_read_returns_none_on_timeout():
    import sys
    import camera_worker

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker._q = Queue(maxsize=1)
    worker._last_frame = b"old"
    worker._q.put_nowait(b"new")

    async def run():
        first = await worker.read()
        worker._last_frame = first
        second = await worker.read(timeout=0.05)
        return first, second

    first, second = asyncio.run(run())
    assert first == b"new"
    assert second is None

    # cleanup so other tests can stub cv2 before importing camera_worker again
    del sys.modules["camera_worker"]
