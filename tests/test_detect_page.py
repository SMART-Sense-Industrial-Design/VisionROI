import asyncio
import sys
from .stubs import stub_cv2, stub_quart

quart_stub = stub_quart()
cv2_stub = stub_cv2()


def test_detect_page(monkeypatch):
    import app
    monkeypatch.setattr(app, 'jsonify', lambda d: d)
    class DummyReq:
        def __init__(self, payload):
            self.payload = payload
        async def get_json(self):
            return self.payload
    app.page_refs[0] = [{'page': '1', 'image': 'abc'}]
    monkeypatch.setattr(app, 'request', DummyReq({'image': 'abc'}))
    result = asyncio.run(app.detect_page(0))
    assert result == {'page': '1'}
    monkeypatch.delitem(sys.modules, 'app', raising=False)
