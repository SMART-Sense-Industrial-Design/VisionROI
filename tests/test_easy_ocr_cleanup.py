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

from inference_modules.easy_ocr.custom import EasyOCR


def test_cleanup_resets_state_and_allows_reuse(monkeypatch):

    ocr = EasyOCR()
    handler = logging.StreamHandler()
    ocr.logger.addHandler(handler)
    ocr._handler = handler
    ocr._current_source = "dummy"

    ocr.process([], roi_id="r1", source="src")
    old_reader = ocr._reader
    assert ocr.last_ocr_results["r1"] == "text"

    ocr.cleanup()
    assert ocr._reader is None
    assert ocr.last_ocr_times == {}
    assert ocr.last_ocr_results == {}
    assert handler not in ocr.logger.handlers

    ocr.process([], roi_id="r2", source="src")
    new_reader = ocr._reader
    assert new_reader is not None and new_reader is not old_reader
    assert ocr.last_ocr_results["r2"] == "text"
