import asyncio
import sys
import types

# stub quart module
quart_stub = types.ModuleType("quart")

class DummyQuart:
    def __init__(self, name):
        self.config = {}

    def route(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator

    def websocket(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator

quart_stub.Quart = DummyQuart
quart_stub.render_template = lambda *a, **k: None
quart_stub.websocket = lambda *a, **k: None
quart_stub.request = None
quart_stub.jsonify = lambda *a, **k: None
quart_stub.send_file = lambda *a, **k: None
quart_stub.redirect = lambda *a, **k: None
sys.modules["quart"] = quart_stub

# stub cv2 module
cv2_stub = types.ModuleType("cv2")
cv2_stub.rectangle = lambda *a, **k: None
cv2_stub.putText = lambda *a, **k: None
cv2_stub.resize = lambda img, dsize, fx=0, fy=0, **k: img
cv2_stub.destroyAllWindows = lambda: None
sys.modules["cv2"] = cv2_stub

import app


def test_read_and_queue_frame_skip_on_none():
    q = app.get_frame_queue(0)
    while not q.empty():
        q.get_nowait()
    class DummyWorker:
        async def read(self):
            return b"frame"

    app.camera_workers[0] = DummyWorker()

    def fake_imencode(ext, frame):
        raise AssertionError("imencode should not be called")

    app.cv2.imencode = fake_imencode

    async def processor(frame):
        return None

    asyncio.run(app.read_and_queue_frame(0, q, processor))

    assert q.empty()
    app.camera_workers[0] = None
