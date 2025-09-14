import asyncio

from .stubs import stub_cv2, stub_quart

quart_stub = stub_quart()
cv2_stub = stub_cv2()

import app


class SlowWS:
    def __init__(self):
        self.closed = False

    async def send(self, data):
        # Wait longer than the timeout in ws handlers
        await asyncio.sleep(2)

    async def close(self, code=1000):
        self.closed = True


class DummyQueue:
    def __init__(self):
        self.calls = 0

    async def get(self):
        self.calls += 1
        if self.calls == 1:
            return b"frame"
        return None

    def get_nowait(self):
        raise asyncio.QueueEmpty


def run_ws(monkeypatch, func_name, getter_name):
    q = DummyQueue()
    ws_dummy = SlowWS()
    monkeypatch.setattr(app, getter_name, lambda cam_id: q)
    monkeypatch.setattr(app, "websocket", ws_dummy)
    asyncio.run(getattr(app, func_name)("0"))
    # Should attempt to fetch a second frame after timeout
    assert q.calls == 2
    assert ws_dummy.closed


def test_ws_timeout_does_not_close(monkeypatch):
    run_ws(monkeypatch, "ws", "get_frame_queue")


def test_ws_roi_timeout_does_not_close(monkeypatch):
    run_ws(monkeypatch, "ws_roi", "get_roi_frame_queue")

