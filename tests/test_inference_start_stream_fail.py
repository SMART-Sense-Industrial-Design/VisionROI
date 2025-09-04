import re


def test_inference_group_start_fail_message():
    with open("templates/partials/inference_content.html", encoding="utf-8") as f:
        content = f.read()
    assert "Failed to start inference stream" in content
    assert re.search(r"catch\s*\([^)]*\)\s*\{[^}]*running\s*=\s*false;[^}]*startButton\.disabled\s*=\s*false;", content, re.DOTALL)


def test_inference_page_start_fail_message():
    with open("templates/partials/inference_page_content.html", encoding="utf-8") as f:
        content = f.read()
    assert "Failed to start inference stream" in content
    assert re.search(r"catch\s*\([^)]*\)\s*\{[^}]*running\s*=\s*false;[^}]*startButton\.disabled\s*=\s*false;", content, re.DOTALL)


def test_perform_start_inference_returns_false_on_open_fail():
    import asyncio
    import sys
    from tests.stubs import stub_cv2

    orig_cv2 = sys.modules.get("cv2")
    cv2_stub = stub_cv2()

    class DummyCap:
        def __init__(self, src):
            self.src = src

        def isOpened(self):
            return False

        def read(self):
            return False, None

        def set(self, prop, value):
            pass

        def release(self):
            pass

    cv2_stub.VideoCapture = DummyCap
    import app
    try:
        async def run():
            app.camera_sources.clear()
            app.camera_workers.clear()
            app.inference_tasks.clear()
            app.camera_sources["cam1"] = 0
            ok = await app.perform_start_inference("cam1")
            assert not ok
            assert app.inference_tasks.get("cam1") is None
        asyncio.run(run())
    finally:
        if orig_cv2 is not None:
            sys.modules["cv2"] = orig_cv2
        else:
            sys.modules.pop("cv2", None)

