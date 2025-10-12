import numpy as np
import inference_modules.rapid_ocr.custom as custom
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


def test_rapid_ocr_removes_scores(monkeypatch):
    ocr = RapidOCR()
    ocr.last_ocr_results.clear()

    class DummyReader:
        def __call__(self, frame):
            return [[("TEXT123", 0.987), ("XYZ", 0.12)]]

    monkeypatch.setattr(ocr, '_get_reader', lambda: DummyReader())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    ocr.process(frame, roi_id='4', save=False, source='', interval=0)
    assert ocr.last_ocr_results['4'] == 'TEXT123 XYZ'


def test_prepare_frame_copies_numpy_views():
    base = np.zeros((12, 12, 3), dtype=np.uint8)
    view = base[:, :6]

    assert view.base is base

    prepared = custom._prepare_frame_for_reader(view)

    # ต้องได้สำเนาใหม่ที่ไม่แชร์บัฟเฟอร์เดียวกับเฟรมต้นฉบับ
    assert prepared.base is None
    assert prepared.flags.c_contiguous

    base[:, :] = 255
    assert not np.array_equal(prepared, base[:, :6])


def test_prepare_frame_copies_arrays_without_own_data():
    buf = bytearray(12 * 12 * 3)
    view = np.frombuffer(buf, dtype=np.uint8).reshape(12, 12, 3)

    assert view.base is not None
    assert not view.flags.owndata

    prepared = custom._prepare_frame_for_reader(view)

    assert prepared.base is None
    assert prepared.flags.owndata
    assert prepared.flags.c_contiguous

    buf[:] = b"\xff" * len(buf)
    assert not np.array_equal(prepared, view)
