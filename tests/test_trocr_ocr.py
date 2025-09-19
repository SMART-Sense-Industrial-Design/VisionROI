import numpy as np

from inference_modules.trocr.custom import TrOCROCR
import inference_modules.trocr.custom as custom


def test_trocr_process_accepts_numpy_frame(monkeypatch):
    ocr = TrOCROCR()
    calls = {}

    def fake_generate(image):
        calls["mode"] = image.mode
        calls["size"] = image.size
        return "hello"

    monkeypatch.setattr(ocr, "_generate_text", fake_generate)

    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    text = ocr.process(frame, roi_id="1", save=False, source="", interval=0)

    assert text == "hello"
    assert ocr.last_ocr_results["1"] == "hello"
    assert calls == {"mode": "RGB", "size": (5, 4)}


def test_trocr_process_handles_generate_error(monkeypatch):
    ocr = TrOCROCR()

    def fail_generate(image):
        raise RuntimeError("boom")

    monkeypatch.setattr(ocr, "_generate_text", fail_generate)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    text = ocr.process(frame, roi_id="e", save=False, source="", interval=0)

    assert text == ""
    assert ocr.last_ocr_results["e"] == ""


def test_trocr_cleanup_resets_singleton(monkeypatch):
    monkeypatch.setattr(custom.TrOCROCR, "_generate_text", lambda self, image: "text")

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    first_result = custom.process(frame, roi_id="a", source="src", interval=0)
    first_instance = custom._get_ocr_instance()

    assert first_result == "text"
    assert first_instance.last_ocr_results["a"] == "text"

    monkeypatch.setattr(custom.TrOCROCR, "_generate_text", lambda self, image: "new")
    custom.cleanup()

    second_result = custom.process(frame, roi_id="b", source="src", interval=0)
    second_instance = custom._get_ocr_instance()

    assert second_result == "new"
    assert "a" not in second_instance.last_ocr_results
    assert second_instance is not first_instance
