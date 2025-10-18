import asyncio
import importlib
import json

import pytest


@pytest.fixture()
def app_module(monkeypatch, tmp_path):
    module = importlib.import_module("app")
    state_path = tmp_path / "service_state.json"
    monkeypatch.setattr(module, "STATE_FILE", str(state_path))
    # isolate global dictionaries that the test touches
    monkeypatch.setattr(module, "camera_sources", {})
    monkeypatch.setattr(module, "camera_resolutions", {})
    monkeypatch.setattr(module, "camera_backends", {})
    monkeypatch.setattr(module, "active_sources", {})
    monkeypatch.setattr(module, "inference_tasks", {})
    monkeypatch.setattr(module, "inference_groups", {})
    monkeypatch.setattr(module, "inference_intervals", {})
    monkeypatch.setattr(module, "inference_result_timeouts", {})
    monkeypatch.setattr(module, "inference_draw_page_boxes", {})
    monkeypatch.setattr(module, "roi_low_latency_flags", {})
    monkeypatch.setattr(module, "roi_tasks", {})
    return module


def test_roi_low_latency_persisted(app_module):
    cam_id = "cam-test"
    module = app_module
    module.camera_sources[cam_id] = "rtsp://example.com/stream"
    module.camera_resolutions[cam_id] = (None, None)
    module.camera_backends[cam_id] = "ffmpeg"
    module.active_sources[cam_id] = "test source"
    module.roi_low_latency_flags[cam_id] = True

    module.save_service_state()

    with open(module.STATE_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["cameras"][cam_id]["roi_low_latency"] is True

    # wipe state before restore
    module.camera_sources.clear()
    module.camera_resolutions.clear()
    module.camera_backends.clear()
    module.active_sources.clear()
    module.roi_low_latency_flags.clear()

    asyncio.run(module.restore_service_state())

    assert module.roi_low_latency_flags.get(cam_id) is True


def test_roi_low_latency_absent_clears_flag(app_module):
    cam_id = "cam-test"
    module = app_module

    # simulate persisted file without the field
    data = {
        "cameras": {
            cam_id: {
                "source": "rtsp://example.com/stream",
                "resolution": [None, None],
                "backend": "ffmpeg",
                "active_source": "test source",
                "inference_running": False,
            }
        }
    }
    with open(module.STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    module.roi_low_latency_flags[cam_id] = True

    asyncio.run(module.restore_service_state())

    assert cam_id not in module.roi_low_latency_flags
