import numpy as np
from inference_modules.rapid_ocr import custom


def test_rapid_ocr_accepts_dict_results(monkeypatch):
    custom.last_ocr_results.clear()
    class DummyReader:
        def __call__(self, frame):
            return ([{"text": "abc"}, {"text": "123"}], None)
    monkeypatch.setattr(custom, '_get_reader', lambda: DummyReader())
    frame = np.zeros((10,10,3), dtype=np.uint8)
    custom._run_ocr_async(frame, roi_id='1', save=False, source='')
    assert custom.last_ocr_results['1'] == 'abc 123'
