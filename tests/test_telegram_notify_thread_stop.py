import sys
import types
from pathlib import Path

# เพิ่ม path ของ src เพื่อให้สามารถ import packages ได้
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# stub cv2 module so import works without opencv
cv2_stub = types.ModuleType("cv2")
cv2_stub.IMWRITE_JPEG_QUALITY = 1
sys.modules["cv2"] = cv2_stub

# stub requests module and its submodules
requests_stub = types.ModuleType("requests")

class DummySession:
    def __init__(self, *a, **k):
        pass

    def post(self, *a, **k):
        return types.SimpleNamespace(status_code=200, json=lambda: {"ok": True}, text="")

    def mount(self, *a, **k):
        pass

    def close(self):
        pass

requests_stub.Session = DummySession

# exceptions submodule
exceptions_stub = types.ModuleType("requests.exceptions")
class SSLError(Exception):
    pass
exceptions_stub.SSLError = SSLError
requests_stub.exceptions = exceptions_stub
sys.modules["requests"] = requests_stub
sys.modules["requests.exceptions"] = exceptions_stub

# adapters submodule
adapters_stub = types.ModuleType("requests.adapters")
class HTTPAdapter:
    def __init__(self, *a, **k):
        pass
class Retry:
    def __init__(self, *a, **k):
        pass
adapters_stub.HTTPAdapter = HTTPAdapter
adapters_stub.Retry = Retry
sys.modules["requests.adapters"] = adapters_stub

from packages.notification.telegram_notify.telegram_notify import TelegramNotify


def test_close_stops_worker_thread():
    tn = TelegramNotify(token="t", chat_id="c")
    assert tn.worker_thread.is_alive()
    tn.close()
    assert not tn.worker_thread.is_alive()
