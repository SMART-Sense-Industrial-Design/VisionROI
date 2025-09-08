import logging

from inference_modules.rapid_ocr.custom import RapidOCR
from inference_modules import base_ocr


def test_cleanup_resets_state_and_calls_gc(monkeypatch):
    ocr = RapidOCR()
    ocr._reader = object()
    ocr.last_ocr_times["roi"] = 1
    ocr.last_ocr_results["roi"] = "text"
    ocr.get_logger("cleanup_test")
    handler = ocr._handler

    called = False

    def fake_collect():
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(base_ocr.gc, "collect", fake_collect)

    ocr.cleanup()

    assert ocr._reader is None
    assert ocr.last_ocr_times == {}
    assert ocr.last_ocr_results == {}
    assert ocr._handler is None
    assert handler not in ocr.logger.handlers
    assert called
