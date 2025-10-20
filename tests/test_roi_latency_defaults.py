import os
from pathlib import Path

import app


def test_default_low_latency_rtsp_ffmpeg():
    assert app._default_roi_low_latency("ffmpeg", "rtsp://example.com/stream") is False
    assert app._default_roi_low_latency("FFMPEG", "rtsps://secure.example/stream") is False


def test_default_low_latency_local_ffmpeg():
    assert app._default_roi_low_latency("ffmpeg", "file.mp4") is True
    assert app._default_roi_low_latency("ffmpeg", Path("file.mp4")) is True


def test_default_low_latency_other_backends():
    assert app._default_roi_low_latency("opencv", "rtsp://example.com/stream") is True
    assert app._default_roi_low_latency(None, os.fspath(Path("video.mp4"))) is True
