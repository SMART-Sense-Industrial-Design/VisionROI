import asyncio

from .stubs import stub_quart, stub_cv2

stub_quart()
stub_cv2()

import app


class _BaseWorker:
    def __init__(self):
        self.read_calls: list[float | None] = []
        self.latest_frame = None

    async def read(self, timeout=0.1):
        self.read_calls.append(timeout)
        return None

    def get_latest_frame(self):
        return self.latest_frame


class WorkerNoRecent(_BaseWorker):
    def has_recent_frame(self, freshness=0.5):
        return False


class WorkerWithRecent(_BaseWorker):
    def __init__(self):
        super().__init__()
        self.latest_frame = object()

    def has_recent_frame(self, freshness=0.5):
        return True


class WorkerStale(_BaseWorker):
    def __init__(self):
        super().__init__()
        self.latest_frame = object()

    def has_recent_frame(self, freshness=0.5):
        return False


def _run_ws_snapshot(cam_id: str):
    async def _run():
        return await app.ws_snapshot(cam_id)

    return asyncio.run(_run())


def test_ws_snapshot_reports_not_ready_when_worker_has_no_frames():
    worker = WorkerNoRecent()
    app.camera_workers["cam_timeout"] = worker
    try:
        resp = _run_ws_snapshot("cam_timeout")
    finally:
        app.camera_workers.pop("cam_timeout", None)
    assert resp == ("Camera not ready", 503)
    assert worker.read_calls
    assert worker.read_calls[0] == 1.5


def test_ws_snapshot_reports_timeout_when_worker_had_recent_frame():
    worker = WorkerWithRecent()
    app.camera_workers["cam_recent"] = worker
    imencode_calls = []
    original_imencode = getattr(app.cv2, "imencode", None)

    def fake_imencode(ext, frame, params=None):
        imencode_calls.append(frame)
        return True, memoryview(b"data")

    app.cv2.imencode = fake_imencode
    try:
        resp = _run_ws_snapshot("cam_recent")
    finally:
        app.camera_workers.pop("cam_recent", None)
        if original_imencode is not None:
            app.cv2.imencode = original_imencode
        elif hasattr(app.cv2, "imencode"):
            delattr(app.cv2, "imencode")
    assert isinstance(resp, app.Response)
    assert getattr(resp, "status_code", None) == 200
    assert not worker.read_calls
    assert imencode_calls
    assert imencode_calls[0] is worker.latest_frame


def test_ws_snapshot_reads_when_cached_frame_stale():
    worker = WorkerStale()
    app.camera_workers["cam_stale"] = worker
    try:
        resp = _run_ws_snapshot("cam_stale")
    finally:
        app.camera_workers.pop("cam_stale", None)
    assert resp == ("Camera not ready", 503)
    assert worker.read_calls
    assert worker.read_calls[0] == 1.5
