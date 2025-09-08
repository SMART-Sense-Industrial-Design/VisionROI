import numpy as np
from inference_modules.rapid_ocr.custom import RapidOCR


def test_rapid_ocr_accepts_dict_results(monkeypatch):
    ocr = RapidOCR()
    ocr.last_ocr_results.clear()
    class DummyReader:
        def __call__(self, frame):
            return ([{"text": "abc"}, {"text": "123"}], None)
    monkeypatch.setattr(ocr, '_get_reader', lambda: DummyReader())
    frame = np.zeros((10,10,3), dtype=np.uint8)
    ocr.process(frame, roi_id='1', save=False, source='', interval=0)
    assert ocr.last_ocr_results['1'] == 'abc 123'


def test_rapid_ocr_accepts_object_with_text(monkeypatch):
    ocr = RapidOCR()
    ocr.last_ocr_results.clear()

    class DummyOutput:
        def __init__(self):
            self.text = "789"

    class DummyReader:
        def __call__(self, frame):
            return DummyOutput()

    monkeypatch.setattr(ocr, '_get_reader', lambda: DummyReader())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    ocr.process(frame, roi_id='2', save=False, source='', interval=0)
    assert ocr.last_ocr_results['2'] == '789'


def test_rapid_ocr_accepts_object_with_txts(monkeypatch):
    ocr = RapidOCR()
    ocr.last_ocr_results.clear()

    class DummyOutput:
        def __init__(self):
            self.txts = ("329",)

    class DummyReader:
        def __call__(self, frame):
            return DummyOutput()

    monkeypatch.setattr(ocr, '_get_reader', lambda: DummyReader())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    ocr.process(frame, roi_id='3', save=False, source='', interval=0)
    assert ocr.last_ocr_results['3'] == '329'
