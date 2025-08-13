import asyncio
import math
import types

from .stubs import stub_cv2, stub_quart

quart_stub = stub_quart()
cv2_stub = stub_cv2()

import app


def test_inference_switches_module_by_group(monkeypatch):
    calls = []

    def make_mod(name):
        mod = types.SimpleNamespace()
        def process(frame, roi_id=None, save=False, source=""):
            calls.append((name, roi_id))
        mod.process = process
        return mod

    modules = {
        "A": make_mod("A"),
        "B": make_mod("B"),
    }
    monkeypatch.setattr(app, "load_custom_module", lambda name: modules.get(name))

    # stub cv2 functions used in run_inference_loop
    monkeypatch.setattr(app.cv2, "imencode", lambda ext, img: (True, b"data"), raising=False)
    monkeypatch.setattr(app.cv2, "getPerspectiveTransform", lambda src, dst: None, raising=False)
    monkeypatch.setattr(app.cv2, "warpPerspective", lambda frame, matrix, shape: frame, raising=False)
    monkeypatch.setattr(app.cv2, "polylines", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app.cv2, "putText", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app.cv2, "resize", lambda img, dsize, fx, fy: img, raising=False)
    monkeypatch.setattr(app.cv2, "FONT_HERSHEY_SIMPLEX", 0, raising=False)
    monkeypatch.setattr(app.cv2, "LINE_AA", 0, raising=False)

    # minimal numpy stub
    class NPArray(list):
        def astype(self, dtype):
            return self

        def __sub__(self, other):
            return NPArray([a - b for a, b in zip(self, other)])

    np_stub = types.SimpleNamespace(
        float32=float,
        uint8=int,
        array=lambda data, dtype=None: NPArray([NPArray(p) for p in data]),
        linalg=types.SimpleNamespace(norm=lambda v: math.sqrt(sum(x * x for x in v))),
    )
    monkeypatch.setattr(app, "np", np_stub)

    dummy_frame = object()

    rois_group1 = [{
        "id": "1",
        "module": "A",
        "points": [{"x":0,"y":0},{"x":1,"y":0},{"x":1,"y":1},{"x":0,"y":1}],
    }]
    rois_group2 = [{
        "id": "1",
        "module": "B",
        "points": [{"x":0,"y":0},{"x":1,"y":0},{"x":1,"y":1},{"x":0,"y":1}],
    }]

    async def fake_read_and_queue_frame(cam_id, queue, frame_processor=None):
        if fake_read_and_queue_frame.count == 0:
            await frame_processor(dummy_frame)
            app.inference_rois[cam_id] = rois_group2
            fake_read_and_queue_frame.count += 1
        else:
            await frame_processor(dummy_frame)
            raise asyncio.CancelledError()
    fake_read_and_queue_frame.count = 0

    monkeypatch.setattr(app, "read_and_queue_frame", fake_read_and_queue_frame)

    app.inference_rois[0] = rois_group1
    app.active_modules[0] = ""
    app.save_roi_flags[0] = False

    asyncio.run(app.run_inference_loop(0))

    assert calls == [("A", "1"), ("B", "1")]
