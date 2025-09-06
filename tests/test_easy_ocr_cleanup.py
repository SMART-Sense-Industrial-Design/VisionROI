import sys
import types
import logging
from tests.stubs import stub_cv2

stub_cv2()

class DummyReader:
    def __init__(self, langs, gpu=False):
        self.calls = []

    def readtext(self, frame, detail=0):
        self.calls.append(frame)
        return ["text"]

# stub easyocr before importing module
_easyocr_stub = types.ModuleType("easyocr")
_easyocr_stub.Reader = DummyReader
sys.modules["easyocr"] = _easyocr_stub

import inference_modules.easy_ocr.custom as custom


def test_cleanup_resets_state_and_allows_reuse(monkeypatch):

    handler = logging.StreamHandler()
    custom.logger.addHandler(handler)
    custom._handler = handler
    custom._current_source = "dummy"

    custom.process([], roi_id="r1", source="src")
    old_reader = custom._reader
    assert custom.last_ocr_results["r1"] == "text"

    custom.cleanup()
    assert custom._reader is None
    assert custom.last_ocr_times == {}
    assert custom.last_ocr_results == {}
    assert handler not in custom.logger.handlers

    custom.process([], roi_id="r2", source="src")
    new_reader = custom._reader
    assert new_reader is not None and new_reader is not old_reader
    assert custom.last_ocr_results["r2"] == "text"
