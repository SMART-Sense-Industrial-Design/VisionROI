import asyncio

from .stubs import stub_cv2, stub_quart


quart_stub = stub_quart()
cv2_stub = stub_cv2()

import app


def test_read_and_queue_frame_skip_on_none():
    q = app.get_frame_queue("0")
    while not q.empty():
        q.get_nowait()
    class DummyWorker:
        async def read(self):
            return b"frame"

    app.camera_workers["0"] = DummyWorker()

    def fake_imencode(ext, frame):
        raise AssertionError("imencode should not be called")

    app.cv2.imencode = fake_imencode

    async def processor(frame, frame_time):
        return None

    asyncio.run(app.read_and_queue_frame("0", q, processor))

    assert q.empty()
    app.camera_workers["0"] = None


def test_read_and_queue_frame_skip_when_worker_returns_none():
    q = app.get_frame_queue("1")
    while not q.empty():
        q.get_nowait()

    class DummyWorker:
        async def read(self):
            return None

    app.camera_workers["1"] = DummyWorker()

    def fake_imencode(ext, frame, params=None):
        raise AssertionError("imencode should not be called")

    app.cv2.imencode = fake_imencode

    asyncio.run(app.read_and_queue_frame("1", q))

    assert q.empty()
    app.camera_workers["1"] = None
