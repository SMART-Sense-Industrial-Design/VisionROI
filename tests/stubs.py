import sys
import types


def stub_quart():
    quart_stub = types.ModuleType("quart")

    class DummyQuart:
        def __init__(self, name):
            self.config = {}

        def route(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator

        # เพิ่มเมธอด get/post ให้คืน decorator แบบเดียวกับ route
        get = route
        post = route

        class DummyResponse:
            def __init__(self, status_code=200):
                self.status_code = status_code

        class DummyClient:
            async def get(self, *args, **kwargs):
                return DummyQuart.DummyResponse()

        def test_client(self):
            return DummyQuart.DummyClient()

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
    quart_stub.Response = DummyQuart.DummyResponse
    sys.modules["quart"] = quart_stub
    return quart_stub


def stub_cv2():
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.rectangle = lambda *a, **k: None
    cv2_stub.putText = lambda *a, **k: None
    cv2_stub.resize = lambda img, dsize, fx=0, fy=0, **k: img
    cv2_stub.destroyAllWindows = lambda: None
    cv2_stub.IMWRITE_JPEG_QUALITY = 1
    sys.modules["cv2"] = cv2_stub
    return cv2_stub
