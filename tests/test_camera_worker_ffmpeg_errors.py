from collections import deque
import types


def test_camera_worker_broken_pipe_triggers_restart(monkeypatch):
    import sys

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    import camera_worker

    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    worker.backend = "ffmpeg"
    worker.robust = True
    worker._err_window = deque(maxlen=10)
    worker._err_window_secs = 5.0
    worker._err_threshold = 3
    worker._last_err_prune = 0.0
    worker._last_frame_ts = 0.0
    worker._read_timeout = 0.5
    worker._fail_count = 0
    worker._logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    worker._log_prefix = "[ffmpeg:test]"
    worker.last_ffmpeg_stderr = lambda: "broken pipe"

    switch_calls: list[dict] = []

    def fake_switch_rtsp_transport(**kwargs):
        switch_calls.append(kwargs)
        return True

    worker._switch_rtsp_transport = fake_switch_rtsp_transport

    restarted: list[bool] = []
    worker._restart_backend = lambda: restarted.append(True)

    times = iter([1.0, 1.1, 1.7])
    monkeypatch.setattr(camera_worker.time, "monotonic", lambda: next(times))

    worker._track_ffmpeg_errors("Error writing trailer of pipe:1: Broken pipe")
    worker._track_ffmpeg_errors("av_interleaved_write_frame(): Broken pipe")
    worker._track_ffmpeg_errors("Error while decoding MB 0 4, bytestream -5")

    assert restarted
    assert switch_calls

    del sys.modules["camera_worker"]
