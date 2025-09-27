import importlib
import sys
import types

import numpy as np

from tests.stubs import stub_cv2


def _install_stubs():
    stub_cv2()

    pytesseract_stub = types.ModuleType("pytesseract")

    calls = {"count": 0}

    def _image_to_string(image, lang=None, config=None):
        calls["count"] += 1
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
    from src.utils import logger as logger_utils

    monkeypatch.setattr(logger_utils, "_DATA_SOURCES_ROOT", tmp_path)
    logger_utils._loggers.clear()

    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    text = ocr.process(frame, roi_id="roi", save=False, source="source")

    assert text == "result"
    assert ocr.last_ocr_results["roi"] == "result"
    assert calls["args"][1] == "eng"
    assert calls["args"][2] == "--psm 6"


def test_tesseract_ocr_returns_cached_result_when_interval_not_elapsed(
    monkeypatch, tmp_path
):
    calls = _install_stubs()

    module_name = "inference_modules.tesseract_ocr.custom"
    sys.modules.pop(module_name, None)
    parent_name = "inference_modules.tesseract_ocr"
    sys.modules.pop(parent_name, None)

    custom = importlib.import_module(module_name)

    # ใช้เวลาเดียวกันในทุกการเรียกเพื่อให้ interval ไม่ผ่าน
    fake_time = types.SimpleNamespace(monotonic=lambda: 100.0)
    import inference_modules.base_ocr as base_ocr_module

    monkeypatch.setattr(base_ocr_module, "time", fake_time)

    ocr = custom.TesseractOCR()
    monkeypatch.setattr(ocr, "_data_sources_root", tmp_path)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    first = ocr.process(frame, roi_id="roi", save=False, source="source")
    second = ocr.process(frame, roi_id="roi", save=False, source="source")

    assert first == "result"
    assert second == "result"
    assert calls["count"] == 1
