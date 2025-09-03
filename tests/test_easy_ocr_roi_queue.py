import sys
import types
import time
import importlib

from tests.stubs import stub_cv2

stub_cv2()


class DummyReader:
    def __init__(self, langs, gpu=False):
        pass

    def readtext(self, frame, detail=0):
        return ["text"]


_easyocr_stub = types.ModuleType("easyocr")
_easyocr_stub.Reader = DummyReader
sys.modules["easyocr"] = _easyocr_stub

import inference_modules.easy_ocr.custom as custom

importlib.reload(custom)


def test_multiple_roi_processed_in_parallel(monkeypatch):
    custom.cleanup()

    start_times = {}

    def fake_run(frame, roi_id, save, source):
        start_times[roi_id] = time.time()
        time.sleep(0.2)

    monkeypatch.setattr(custom, "_run_ocr_async", fake_run)

    custom.process([], roi_id="r1", interval=0)
    custom.process([], roi_id="r2", interval=0)

    time.sleep(0.5)

    assert "r1" in start_times and "r2" in start_times
    assert abs(start_times["r1"] - start_times["r2"]) < 0.2

    custom.cleanup()


def test_queue_limit_per_roi(monkeypatch):
    custom.cleanup()

    calls = []

    def fake_run(frame, roi_id, save, source):
        calls.append(roi_id)
        time.sleep(0.2)

    monkeypatch.setattr(custom, "_run_ocr_async", fake_run)

    for _ in range(10):
        custom.process([], roi_id="r1", interval=0)

    time.sleep(1.0)

    assert len(calls) <= 4

    custom.cleanup()

