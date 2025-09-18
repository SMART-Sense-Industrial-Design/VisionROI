import asyncio
import importlib
import sys
from pathlib import Path


def test_apply_camera_settings_preserves_existing_active_source():
    original_cv2 = sys.modules.get("cv2")
    original_quart = sys.modules.get("quart")

    app = importlib.import_module("app")

    async def run():
        cam_id = "preserve_cam"
        app.active_sources[cam_id] = "saved-source"
        try:
            result = await app.apply_camera_settings(
                cam_id,
                {
                    "source": "0",
                    "stream_type": "opencv",
                },
            )
            preserved = app.active_sources.get(cam_id)
        finally:
            app.camera_sources.pop(cam_id, None)
            app.camera_resolutions.pop(cam_id, None)
            app.camera_backends.pop(cam_id, None)
            app.active_sources.pop(cam_id, None)
            Path(app.STATE_FILE).unlink(missing_ok=True)
        assert result is True
        assert preserved == "saved-source"

    try:
        asyncio.run(run())
    finally:
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
