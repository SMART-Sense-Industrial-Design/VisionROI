import logging

from inference_modules.rapid_ocr.custom import RapidOCR
from inference_modules import base_ocr


def test_cleanup_resets_state_and_calls_gc(monkeypatch):
    custom._reader = object()
    custom.last_ocr_times["roi"] = 1
    custom.last_ocr_results["roi"] = "text"


    called = False

    def fake_collect():
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(base_ocr.gc, "collect", fake_collect)

    ocr.cleanup()

    assert custom._reader is None
    assert custom.last_ocr_times == {}
    assert custom.last_ocr_results == {}

    assert called
