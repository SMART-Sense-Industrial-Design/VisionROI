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
    worker._ffmpeg_output = "rawvideo"
    worker.backend = "ffmpeg"
    return worker


def test_low_latency_rtsp_uses_passthrough_sync():
    worker = _make_worker()
    cmd = worker._build_ffmpeg_cmd("rtsp://example.com/stream")
    assert "-vsync" in cmd
    vsync_idx = cmd.index("-vsync")
    assert cmd[vsync_idx + 1] == "passthrough"
    fps_idx = cmd.index("-fps_mode")
    assert cmd[fps_idx + 1] == "passthrough"
    probe_idx = cmd.index("-probesize")
    assert cmd[probe_idx + 1] == "512k"
    analyze_idx = cmd.index("-analyzeduration")
    assert cmd[analyze_idx + 1] == "200k"
    fflags_idx = cmd.index("-fflags")
    assert cmd[fflags_idx + 1] == "+discardcorrupt"
    max_delay_idx = cmd.index("-max_delay")
    assert cmd[max_delay_idx + 1] == "250000"


def test_low_latency_avfoundation_keeps_latency_flags_without_reorder_queue():
    worker = _make_worker(global_opts={"reorder_queue_size"})
    cmd = worker._build_ffmpeg_cmd("avfoundation:0:none")
    assert "-reorder_queue_size" not in cmd
    assert "-video_size" in cmd
    size_idx = cmd.index("-video_size")
    assert cmd[size_idx + 1] == "1280x720"
    vf_idx = cmd.index("-vf")
    assert "scale=1280:720" in cmd[vf_idx + 1]
    assert "-vsync" in cmd


def test_mjpeg_output_switches_to_image2pipe():
    worker = _make_worker()
    worker._ffmpeg_output = "mjpeg"
    cmd = worker._build_ffmpeg_cmd("rtsp://example.com/stream")
    assert "image2pipe" in cmd
    assert "mjpeg" in cmd
    assert "-pix_fmt" not in cmd  # ไม่ควรบังคับ bgr24 เมื่อออก MJPEG
