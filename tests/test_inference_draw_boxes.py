import asyncio
import types

import app


def test_inference_draws_page_and_roi(monkeypatch):
    colors = []

    # stub cv2 functions
    monkeypatch.setattr(app.cv2, "imencode", lambda ext, img: (True, b"data"), raising=False)
    monkeypatch.setattr(app.cv2, "getPerspectiveTransform", lambda src, dst: None, raising=False)
    monkeypatch.setattr(app.cv2, "warpPerspective", lambda frame, matrix, shape: frame, raising=False)

    def poly_stub(*args, **kwargs):
        # color is 4th positional argument
        colors.append(args[3])
    monkeypatch.setattr(app.cv2, "polylines", poly_stub, raising=False)
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
        linalg=types.SimpleNamespace(norm=lambda v: 1),
    )
    monkeypatch.setattr(app, "np", np_stub)

    # dummy module
    mod = types.SimpleNamespace(process=lambda *a, **k: None)
    monkeypatch.setattr(app, "load_custom_module", lambda name: mod)

    page = [{
        "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}],
    }]
    roi = [{
        "id": "1",
        "module": "m",
        "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}],
    }]

    app.page_rois[0] = page
    app.inference_rois[0] = roi
    app.save_roi_flags[0] = False

    async def fake_read_and_queue_frame(cam_id, queue, frame_processor=None):
        await frame_processor(object())
        raise asyncio.CancelledError()

    monkeypatch.setattr(app, "read_and_queue_frame", fake_read_and_queue_frame)

    try:
        asyncio.run(app.run_inference_loop(0))
    except asyncio.CancelledError:
        pass

    assert colors == [(0, 255, 0), (255, 0, 0)]
