import asyncio

from .stubs import stub_cv2, stub_quart

quart_stub = stub_quart()
cv2_stub = stub_cv2()

import app


class DummyWS:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def close(self, code=1000):
        self.closed = True


class DummyQueue:
    def __init__(self):
        self.step = 0
        self.extra_given = False

    async def get(self):
        self.step += 1
        if self.step == 1:
            return b"old"
        return None

    def get_nowait(self):
        if self.step == 1 and not self.extra_given:
            self.extra_given = True
            return b"latest"
        raise asyncio.QueueEmpty


def run_ws(monkeypatch, func_name, getter_name):
    q = DummyQueue()
    ws_dummy = DummyWS()
    monkeypatch.setattr(app, getter_name, lambda cam_id: q)
    monkeypatch.setattr(app, "websocket", ws_dummy)
    asyncio.run(getattr(app, func_name)("0"))
    assert ws_dummy.sent == [b"latest"]
    assert ws_dummy.closed


def test_ws_uses_latest_frame(monkeypatch):
    run_ws(monkeypatch, "ws", "get_frame_queue")


def test_ws_roi_uses_latest_frame(monkeypatch):
    run_ws(monkeypatch, "ws_roi", "get_roi_frame_queue")
