import numpy as np

from camera_worker import CameraWorker


def _make_worker_with_array():
    worker = object.__new__(CameraWorker)
    worker._latest_frame = np.arange(9, dtype=np.uint8).reshape(3, 3)
    return worker


def test_get_latest_frame_returns_independent_copy():
    worker = _make_worker_with_array()
    cached = worker._latest_frame
    snapshot_frame = worker.get_latest_frame()
    assert snapshot_frame is not None
    assert snapshot_frame is not cached
    snapshot_frame[0, 0] = 255
    assert cached[0, 0] != 255


def test_get_latest_frame_falls_back_when_copy_missing():
    worker = object.__new__(CameraWorker)

    class NoCopy:
        def __init__(self):
            self.value = "frame"

    placeholder = NoCopy()
    worker._latest_frame = placeholder

    snapshot_frame = worker.get_latest_frame()
    assert snapshot_frame is placeholder
