import importlib
import sys
import types

import numpy as np

from tests.stubs import stub_cv2


def _install_stubs():
    stub_cv2()

    pytesseract_stub = types.ModuleType("pytesseract")

    calls = {}

    def _image_to_string(image, lang=None, config=None):
        calls["args"] = (image, lang, config)
        return " result "

    pytesseract_stub.image_to_string = _image_to_string
    sys.modules["pytesseract"] = pytesseract_stub
    return calls


def test_tesseract_ocr_process_returns_text(monkeypatch, tmp_path):
    calls = _install_stubs()

    module_name = "inference_modules.tesseract_ocr.custom"
    sys.modules.pop(module_name, None)
    parent_name = "inference_modules.tesseract_ocr"
    sys.modules.pop(parent_name, None)

    custom = importlib.import_module(module_name)
    ocr = custom.TesseractOCR(lang="eng", config="--psm 6")
    monkeypatch.setattr(ocr, "_data_sources_root", tmp_path)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    text = ocr.process(frame, roi_id="roi", save=False, source="source")

    assert text == "result"
    assert ocr.last_ocr_results["roi"] == "result"
    assert calls["args"][1] == "eng"
    assert calls["args"][2] == "--psm 6"
