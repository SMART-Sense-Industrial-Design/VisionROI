import logging

from tests import stubs

stubs.stub_cv2()

import camera_worker


def _make_worker(avf_pixel_format=None):
    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker._loglevel = "error"
    worker._rtsp_transport_cycle = ["tcp"]
    worker._rtsp_transport_idx = 0
    worker._ff_rtsp_opts = set()
    worker.low_latency = False
    worker.width = None
    worker.height = None
    worker._logger = logging.getLogger("test_camera_worker")
    worker._log_prefix = "[test]"
    worker._ffmpeg_pix_fmt = None
    worker.avf_pixel_format = avf_pixel_format
    worker.backend = "ffmpeg"
    return worker


def test_avfoundation_default_pixel_format():
    worker = _make_worker()
    cmd = worker._build_ffmpeg_cmd("avfoundation:0:none")

    assert "-pixel_format" in cmd
    pix_idx = cmd.index("-pixel_format")
    assert cmd[pix_idx + 1] == "nv12"
    assert pix_idx < cmd.index("-i")
    assert "-video_size" in cmd
    size_idx = cmd.index("-video_size")
    assert cmd[size_idx + 1] == "1280x720"


def test_avfoundation_override_pixel_format():
    worker = _make_worker("uyvy422")
    cmd = worker._build_ffmpeg_cmd("avfoundation:0:none")

    assert "-pixel_format" in cmd
    pix_idx = cmd.index("-pixel_format")
    assert cmd[pix_idx + 1] == "uyvy422"
