import sys
import types

import numpy as np

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from inference_modules.rapid_ocr import custom


def test_prepare_frame_detaches_numpy_view():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    roi_view = frame[:, :2]

    prepared = custom._prepare_frame_for_reader(roi_view)

    frame[:, :2] = 255

    assert np.all(prepared == 0)
    assert prepared.flags.c_contiguous
    assert prepared.base is None
