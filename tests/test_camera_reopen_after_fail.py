import asyncio


def test_camera_reopen_after_fail():
    import app

    class DummyCap:
        def __init__(self, src):
            self.src = src
            self.released = False

        def isOpened(self):
            return self.src != 5

        def read(self):
            return True, b"frame"

        def set(self, prop, value):
            pass

        def release(self):
            self.released = True

    orig_vc = app.cv2.VideoCapture
    orig_destroy = app.cv2.destroyAllWindows
    app.cv2.VideoCapture = DummyCap
    app.cv2.destroyAllWindows = lambda: None
    try:
        async def run():
            app.camera_sources.clear()
            app.camera_workers.clear()
            app.roi_tasks.clear()
            app.inference_tasks.clear()
            app.camera_sources["cam1"] = 5
            task, resp, status = await app.start_camera_task("cam1", app.roi_tasks, app.run_roi_loop)
            assert status == 400
            assert app.camera_workers.get("cam1") is None
            app.camera_sources["cam1"] = 0
            task, resp, status = await app.start_camera_task("cam1", app.roi_tasks, app.run_roi_loop)
            assert status == 200
            await asyncio.sleep(0)
            app.roi_tasks["cam1"], _, _ = await app.stop_camera_task("cam1", app.roi_tasks)
        asyncio.run(run())
    finally:
        app.cv2.VideoCapture = orig_vc
        app.cv2.destroyAllWindows = orig_destroy
