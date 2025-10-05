import pytest

pytest.importorskip("cv2")

import camera_worker


class DummyResult:
    def __init__(self, stdout: str = "640x480\n") -> None:
        self.stdout = stdout
        self.stderr = ""


def test_probe_resolution_uses_avfoundation_format(monkeypatch):
    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    calls = {}

    def fake_run(cmd, capture_output, text, check, timeout):
        calls["cmd"] = cmd
        return DummyResult()

    monkeypatch.setattr(camera_worker.subprocess, "run", fake_run)

    width, height = worker._probe_resolution("avfoundation:0:none")

    assert (width, height) == (640, 480)
    assert "-f" in calls["cmd"]
    fmt_index = calls["cmd"].index("-f")
    assert calls["cmd"][fmt_index + 1] == "avfoundation"
    assert calls["cmd"][-2:] == ["-i", "0:none"]


def test_probe_resolution_generic_source(monkeypatch):
    worker = camera_worker.CameraWorker.__new__(camera_worker.CameraWorker)
    calls = {}

    def fake_run(cmd, capture_output, text, check, timeout):
        calls["cmd"] = cmd
        return DummyResult("1920x1080\n")

    monkeypatch.setattr(camera_worker.subprocess, "run", fake_run)

    width, height = worker._probe_resolution("/path/to/video.mp4")

    assert (width, height) == (1920, 1080)
    assert "-f" not in calls["cmd"]
    assert calls["cmd"][-2:] == ["-i", "/path/to/video.mp4"]
