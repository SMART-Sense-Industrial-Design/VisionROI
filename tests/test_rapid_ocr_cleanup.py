import logging

from inference_modules.rapid_ocr import custom


def test_cleanup_resets_state_and_calls_gc(monkeypatch):
    custom._reader = object()
    custom.last_ocr_times["roi"] = 1
    custom.last_ocr_results["roi"] = "text"
    custom._configure_logger("cleanup_test")
    handler = custom._handler

    called = False

    def fake_collect():
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(custom.gc, "collect", fake_collect)

    custom.cleanup()

    assert custom._reader is None
    assert custom.last_ocr_times == {}
    assert custom.last_ocr_results == {}
    assert custom._handler is None
    assert handler not in custom.logger.handlers
    assert called
