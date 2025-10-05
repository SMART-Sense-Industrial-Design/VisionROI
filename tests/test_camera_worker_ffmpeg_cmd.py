import logging

from tests.stubs import stub_cv2

stub_cv2()

from camera_worker import CameraWorker


def _make_worker(*, low_latency: bool = True, global_opts: set[str] | None = None) -> CameraWorker:
    worker = CameraWorker.__new__(CameraWorker)
    worker.low_latency = low_latency
    worker._loglevel = "error"
    worker._rtsp_transport_cycle = ["tcp"]
    worker._rtsp_transport_idx = 0
    worker._ff_rtsp_opts = {"rtsp_transport", "rtsp_flags", "rw_timeout"}
    worker._ff_global_opts = set(global_opts or set())
    worker.width = None
    worker.height = None
    worker._logger = logging.getLogger("camera-worker-test")
    worker._log_prefix = "[test]"
    worker.robust = True
    worker._ffmpeg_pix_fmt = None
    worker.backend = "ffmpeg"
    return worker


def test_low_latency_rtsp_includes_reorder_queue_size():
    worker = _make_worker()
    cmd = worker._build_ffmpeg_cmd("rtsp://example.com/stream")
    assert "-reorder_queue_size" in cmd
    idx = cmd.index("-reorder_queue_size")
    assert cmd[idx + 1] == "0"


def test_low_latency_avfoundation_skips_reorder_queue_size():
    worker = _make_worker(global_opts={"reorder_queue_size"})
    cmd = worker._build_ffmpeg_cmd("avfoundation:0:none")
    assert "-reorder_queue_size" not in cmd
    assert "-video_size" in cmd
    size_idx = cmd.index("-video_size")
    assert cmd[size_idx + 1] == "1280x720"
    vf_idx = cmd.index("-vf")
    assert "scale=1280:720" in cmd[vf_idx + 1]
