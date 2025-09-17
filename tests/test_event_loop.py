import asyncio
import contextlib
import json
import shutil
import time
import uuid
from types import SimpleNamespace
from queue import Queue
from pathlib import Path

import numpy as np

from .stubs import stub_cv2, stub_quart


quart_stub = stub_quart()
cv2_stub = stub_cv2()


class DummyCamera:
    def __init__(self, *a, **k):
        self.frames_read = 0

    def isOpened(self):
        return True

    def read(self):
        self.frames_read += 1
        # จำลองการบล็อก IO หรือการคำนวณที่ใช้เวลานาน
        time.sleep(0.005)
        return True, b"frame"

    def release(self):
        pass


cv2_stub.VideoCapture = DummyCamera
cv2_stub.imencode = lambda *a, **k: (True, b"data")
cv2_stub.rectangle = lambda *a, **k: None
cv2_stub.putText = lambda *a, **k: None
cv2_stub.resize = lambda img, dsize, fx=0, fy=0, **k: img
cv2_stub.destroyAllWindows = lambda: None

import app


async def ticker(duration: float = 0.5, interval: float = 0.01) -> int:
    count = 0
    end = time.monotonic() + duration
    while time.monotonic() < end:
        count += 1
        await asyncio.sleep(interval)
    return count


def test_event_loop_responsive():
    async def main():
        worker = app.CameraWorker(0, asyncio.get_running_loop())
        assert worker.start()
        app.camera_workers["0"] = worker
        app.inference_rois["0"] = []
        app.active_sources["0"] = ""
        app.cv2.imencode = lambda ext, frame: (True, b"data")

        loop_task = asyncio.create_task(app.run_inference_loop("0"))
        tick_task = asyncio.create_task(ticker())

        ticks = await tick_task
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

        frames = worker._cap.frames_read
        await worker.stop()
        app.camera_workers["0"] = None
        return ticks, frames

    ticks, frames = asyncio.run(main())
    # ยืนยันว่า event loop ยังคง responsive
    assert ticks > 20
    # ยืนยันว่ามีการอ่านเฟรมจำนวนมาก
    assert frames >= 9


def test_event_loop_responsive_heavy_imencode():
    async def main():
        worker = app.CameraWorker(0, asyncio.get_running_loop())
        assert worker.start()
        app.camera_workers["0"] = worker
        app.inference_rois["0"] = []
        app.active_sources["0"] = ""

        def heavy_imencode(ext, frame):
            time.sleep(0.05)
            return True, b"data"

        app.cv2.imencode = heavy_imencode

        loop_task = asyncio.create_task(app.run_inference_loop("0"))
        tick_task = asyncio.create_task(ticker())

        ticks = await tick_task
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

        frames = worker._cap.frames_read
        await worker.stop()
        app.camera_workers["0"] = None
        return ticks, frames

    ticks, frames = asyncio.run(main())
    assert ticks > 20
    assert frames >= 9


def test_inference_timeout_allows_slow_results():
    async def main():
        worker = app.CameraWorker(0, asyncio.get_running_loop())
        assert worker.start()
        app.camera_workers["0"] = worker
        app.active_sources["0"] = "slow-source"
        app.camera_sources["0"] = 0
        app.inference_rois["0"] = [
            {
                "id": "roi_test",
                "type": "roi",
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 10, "y": 0},
                    {"x": 10, "y": 10},
                    {"x": 0, "y": 10},
                ],
                "module": "slow_module",
            }
        ]
        app.inference_intervals["0"] = 0.01
        app.inference_result_timeouts["0"] = 0.05
        app.inference_groups["0"] = "all"

        logs: list[str] = []

        class DummyLogger:
            def info(self, msg, *args, **kwargs):
                if args:
                    logs.append(args[0])
                else:
                    logs.append(msg)

        original_logger = app.get_logger
        app.get_logger = lambda *a, **k: DummyLogger()

        def slow_process(*_args):
            time.sleep(0.12)
            return {"text": "slow"}

        original_loader = app.load_custom_module
        app.load_custom_module = lambda name: SimpleNamespace(process=slow_process)

        original_imencode = app.cv2.imencode
        original_get_perspective = getattr(app.cv2, "getPerspectiveTransform", None)
        original_warp = getattr(app.cv2, "warpPerspective", None)
        original_polylines = getattr(app.cv2, "polylines", None)
        original_put_text = getattr(app.cv2, "putText", None)
        original_font = getattr(app.cv2, "FONT_HERSHEY_SIMPLEX", None)
        original_line_aa = getattr(app.cv2, "LINE_AA", None)

        app.cv2.getPerspectiveTransform = lambda src, dst: np.eye(3, dtype=np.float32)

        def warp_perspective(_frame, _matrix, size):
            w, h = size
            w = max(int(w), 1)
            h = max(int(h), 1)
            return np.zeros((h, w, 3), dtype=np.uint8)

        app.cv2.warpPerspective = warp_perspective
        app.cv2.polylines = lambda *a, **k: None
        app.cv2.putText = lambda *a, **k: None
        app.cv2.imencode = lambda *_a, **_k: (True, np.ones((10,), dtype=np.uint8))
        app.cv2.FONT_HERSHEY_SIMPLEX = 0
        app.cv2.LINE_AA = 0

        loop_task = asyncio.create_task(app.run_inference_loop("0"))
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not logs:
                await asyncio.sleep(0.05)
            return logs
        finally:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task
            await worker.stop()
            app.camera_workers.pop("0", None)
            app.camera_sources.pop("0", None)
            app.inference_rois.pop("0", None)
            app.inference_intervals.pop("0", None)
            app.inference_result_timeouts.pop("0", None)
            app.inference_groups.pop("0", None)
            app.active_sources.pop("0", None)
            app.get_logger = original_logger
            app.load_custom_module = original_loader
            app.cv2.imencode = original_imencode
            if original_get_perspective is not None:
                app.cv2.getPerspectiveTransform = original_get_perspective
            else:
                delattr(app.cv2, "getPerspectiveTransform")
            if original_warp is not None:
                app.cv2.warpPerspective = original_warp
            else:
                delattr(app.cv2, "warpPerspective")
            if original_polylines is not None:
                app.cv2.polylines = original_polylines
            else:
                delattr(app.cv2, "polylines")
            if original_put_text is not None:
                app.cv2.putText = original_put_text
            else:
                delattr(app.cv2, "putText")
            if original_font is not None:
                app.cv2.FONT_HERSHEY_SIMPLEX = original_font
            else:
                delattr(app.cv2, "FONT_HERSHEY_SIMPLEX")
            if original_line_aa is not None:
                app.cv2.LINE_AA = original_line_aa
            else:
                delattr(app.cv2, "LINE_AA")

    logs = asyncio.run(main())
    assert logs, "ควรมี log AGGREGATED_ROI แม้ inference ช้า"
    assert any('"slow"' in entry for entry in logs)


def test_aggregated_roi_logs_all_results_when_queue_full():
    async def main():
        original_queue_size = app._INFERENCE_QUEUE.maxsize
        original_max_workers = app.MAX_WORKERS
        app._stop_inference_workers()
        app._INFERENCE_QUEUE = Queue(maxsize=1)
        app.MAX_WORKERS = 1
        app._EXECUTOR = None
        app._start_inference_workers()

        worker = app.CameraWorker(0, asyncio.get_running_loop())
        assert worker.start()
        app.camera_workers["0"] = worker
        app.camera_sources["0"] = 0
        app.active_sources["0"] = "queue-full"
        app.inference_intervals["0"] = 0.01
        app.inference_groups["0"] = "all"

        rois = []
        for idx in range(9):
            base = idx * 10
            rois.append(
                {
                    "id": f"roi-{idx}",
                    "type": "roi",
                    "points": [
                        {"x": base, "y": base},
                        {"x": base + 5, "y": base},
                        {"x": base + 5, "y": base + 5},
                        {"x": base, "y": base + 5},
                    ],
                    "module": "dummy_module",
                }
            )
        app.inference_rois["0"] = rois

        logs: list[str] = []

        class DummyLogger:
            def info(self, msg, *args, **kwargs):
                if args:
                    logs.append(args[0])
                else:
                    logs.append(msg)

        def process_roi(_image, roi_id, _save_flag, *_args):
            time.sleep(0.02)
            return {"text": f"processed-{roi_id}"}

        dummy_module = SimpleNamespace(process=process_roi)

        original_logger = app.get_logger
        original_loader = app.load_custom_module
        original_get_perspective = getattr(app.cv2, "getPerspectiveTransform", None)
        original_warp = getattr(app.cv2, "warpPerspective", None)
        original_polylines = getattr(app.cv2, "polylines", None)
        original_put_text = getattr(app.cv2, "putText", None)
        original_imencode = getattr(app.cv2, "imencode", None)
        original_font = getattr(app.cv2, "FONT_HERSHEY_SIMPLEX", None)
        original_line_aa = getattr(app.cv2, "LINE_AA", None)

        app.get_logger = lambda *a, **k: DummyLogger()
        app.load_custom_module = lambda name: dummy_module
        app.cv2.getPerspectiveTransform = lambda src, dst: np.eye(3, dtype=np.float32)

        def warp_perspective(_frame, _matrix, size):
            width = max(int(size[0]), 1)
            height = max(int(size[1]), 1)
            return np.zeros((height, width, 3), dtype=np.uint8)

        app.cv2.warpPerspective = warp_perspective
        app.cv2.polylines = lambda *a, **k: None
        app.cv2.putText = lambda *a, **k: None
        app.cv2.imencode = lambda *_a, **_k: (True, np.ones((10,), dtype=np.uint8))
        app.cv2.FONT_HERSHEY_SIMPLEX = 0
        app.cv2.LINE_AA = 0

        loop_task = asyncio.create_task(app.run_inference_loop("0"))
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not logs:
                await asyncio.sleep(0.05)
            return logs
        finally:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task
            await worker.stop()
            app.camera_workers.pop("0", None)
            app.camera_sources.pop("0", None)
            app.active_sources.pop("0", None)
            app.inference_rois.pop("0", None)
            app.inference_intervals.pop("0", None)
            app.inference_groups.pop("0", None)
            app.get_logger = original_logger
            app.load_custom_module = original_loader
            if original_get_perspective is not None:
                app.cv2.getPerspectiveTransform = original_get_perspective
            else:
                delattr(app.cv2, "getPerspectiveTransform")
            if original_warp is not None:
                app.cv2.warpPerspective = original_warp
            else:
                delattr(app.cv2, "warpPerspective")
            if original_polylines is not None:
                app.cv2.polylines = original_polylines
            else:
                delattr(app.cv2, "polylines")
            if original_put_text is not None:
                app.cv2.putText = original_put_text
            else:
                delattr(app.cv2, "putText")
            if original_imencode is not None:
                app.cv2.imencode = original_imencode
            else:
                delattr(app.cv2, "imencode")
            if original_font is not None:
                app.cv2.FONT_HERSHEY_SIMPLEX = original_font
            else:
                delattr(app.cv2, "FONT_HERSHEY_SIMPLEX")
            if original_line_aa is not None:
                app.cv2.LINE_AA = original_line_aa
            else:
                delattr(app.cv2, "LINE_AA")
            app._stop_inference_workers()
            app._INFERENCE_QUEUE = Queue(maxsize=original_queue_size)
            app.MAX_WORKERS = original_max_workers
            app._EXECUTOR = None
            app._start_inference_workers()

    log_entries = asyncio.run(main())
    assert log_entries, "ควรได้ log AGGREGATED_ROI ทุกครั้ง"
    for entry in log_entries:
        payload = json.loads(entry)
        results = payload.get("results", [])
        assert len(results) == 9, "ควรมีผลลัพธ์ ROI ครบ 9 รายการ"
        assert all(item.get("module") == "dummy_module" for item in results)


def test_aggregated_roi_logs_separate_files_per_source():
    from src.utils import logger as logger_utils

    source_one = f"test-source-{uuid.uuid4().hex}"
    source_two = f"test-source-{uuid.uuid4().hex}"
    cam_one = "cam-1"
    cam_two = "cam-2"

    logger_one = None
    logger_two = None
    log_file_one = Path("data_sources") / source_one / "custom.log"
    log_file_two = Path("data_sources") / source_two / "custom.log"

    try:
        logger_one = logger_utils.get_logger("aggregated_roi", source_one)
        logger_two = logger_utils.get_logger("aggregated_roi", source_two)

        entry_one = {
            "frame_time": 1.0,
            "result_time": 2.0,
            "cam_id": cam_one,
            "source": source_one,
            "results": [{"id": "roi-1", "name": "", "text": "foo"}],
        }
        entry_two = {
            "frame_time": 3.0,
            "result_time": 4.0,
            "cam_id": cam_two,
            "source": source_two,
            "results": [{"id": "roi-2", "name": "", "text": "bar"}],
        }

        logger_one.info("AGGREGATED_ROI %s", json.dumps(entry_one, ensure_ascii=False))
        logger_two.info("AGGREGATED_ROI %s", json.dumps(entry_two, ensure_ascii=False))

        for active_logger in (logger_one, logger_two):
            for handler in active_logger.handlers:
                handler.flush()

        content_one = log_file_one.read_text(encoding="utf-8")
        content_two = log_file_two.read_text(encoding="utf-8")

        assert f'"cam_id": "{cam_one}"' in content_one
        assert source_one in content_one
        assert f'"cam_id": "{cam_two}"' not in content_one
        assert source_two not in content_one

        assert f'"cam_id": "{cam_two}"' in content_two
        assert source_two in content_two
        assert f'"cam_id": "{cam_one}"' not in content_two
        assert source_one not in content_two
    finally:
        for active_logger, src_name, log_file in [
            (logger_one, source_one, log_file_one),
            (logger_two, source_two, log_file_two),
        ]:
            if active_logger is not None:
                for handler in list(active_logger.handlers):
                    handler.close()
                    active_logger.removeHandler(handler)
            logger_utils._loggers.pop(("aggregated_roi", src_name or ""), None)
            log_dir = log_file.parent
            if log_dir.exists():
                shutil.rmtree(log_dir, ignore_errors=True)
