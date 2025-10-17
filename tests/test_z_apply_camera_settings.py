import asyncio
import importlib
import sys
from pathlib import Path

import pytest


def _cleanup_app_state(app_module, *cam_ids: str) -> None:
    for cam_id in cam_ids:
        app_module.camera_sources.pop(cam_id, None)
        app_module.camera_resolutions.pop(cam_id, None)
        app_module.camera_backends.pop(cam_id, None)
        app_module.active_sources.pop(cam_id, None)
        app_module.inference_tasks.pop(cam_id, None)
        app_module.roi_tasks.pop(cam_id, None)
        app_module.camera_workers.pop(cam_id, None)
    Path(app_module.STATE_FILE).unlink(missing_ok=True)


def _restore_modules(original_cv2, original_quart) -> None:
    sys.modules.pop("app", None)
    sys.modules.pop("camera_worker", None)
    if original_cv2 is not None:
        sys.modules["cv2"] = original_cv2
    else:
        sys.modules.pop("cv2", None)
    if original_quart is not None:
        sys.modules["quart"] = original_quart
    else:
        sys.modules.pop("quart", None)


def _run_async_test(test_coro):
    original_cv2 = sys.modules.get("cv2")
    original_quart = sys.modules.get("quart")
    app_module = importlib.import_module("app")

    try:
        asyncio.run(test_coro(app_module))
    finally:
        _restore_modules(original_cv2, original_quart)


def test_apply_camera_settings_preserves_existing_active_source():
    async def run(app_module):
        cam_id = "preserve_cam"
        app_module.active_sources[cam_id] = "saved-source"
        try:
            result = await app_module.apply_camera_settings(
                cam_id,
                {
                    "source": "0",
                    "stream_type": "opencv",
                },
            )
            preserved = app_module.active_sources.get(cam_id)
        finally:
            _cleanup_app_state(app_module, cam_id)
        assert result is True
        assert preserved == "saved-source"

    _run_async_test(run)


@pytest.mark.parametrize(
    ("source_url", "cam_id"),
    (
        ("http://192.168.10.20:8889/cam1", "network_cam_http"),
        ("rtsp://192.168.10.20:8554/cam1", "network_cam_rtsp"),
    ),
)
def test_apply_camera_settings_forces_ffmpeg_for_network_sources(source_url, cam_id):
    async def run(app_module):
        try:
            result = await app_module.apply_camera_settings(
                cam_id,
                {
                    "source": source_url,
                    "stream_type": "opencv",
                },
            )
            backend = app_module.camera_backends.get(cam_id)
        finally:
            _cleanup_app_state(app_module, cam_id)
        assert result is True
        assert backend == "ffmpeg"

    _run_async_test(run)
