import time

import pytest


def test_dashboard_latest_runs_count_all_results_without_duration(monkeypatch):
    import app

    notifications = [
        {
            "timestamp_epoch": time.time(),
            "cam_id": "cam-1",
            "source": "source-a",
            "results": [
                {"id": "roi-1", "name": "A", "module": "ocr"},
                {"id": "roi-2", "name": "B", "module": "ocr"},
            ],
        }
    ]

    monkeypatch.setattr(app, "load_service_state", lambda: {"cam-1": {"source": "source-a"}})
    monkeypatch.setattr(app, "get_recent_notifications", lambda limit=None: list(notifications))
    monkeypatch.setattr(app, "np", object())

    monkeypatch.setattr(app, "camera_sources", {"cam-1": "source-a"})
    monkeypatch.setattr(app, "active_sources", {"cam-1": "Cam One"})
    monkeypatch.setattr(app, "inference_groups", {"cam-1": "all"})
    monkeypatch.setattr(app, "inference_intervals", {"cam-1": 1.0})
    monkeypatch.setattr(app, "inference_rois", {
        "cam-1": [
            {
                "type": "roi",
                "module": "ocr",
                "group": "all",
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 1, "y": 1},
                    {"x": 0, "y": 1},
                ],
            },
            {
                "type": "roi",
                "module": "ocr",
                "group": "all",
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 2, "y": 0},
                    {"x": 2, "y": 2},
                    {"x": 0, "y": 2},
                ],
            },
        ]
    })

    class _DummyTask:
        def done(self):
            return False

    monkeypatch.setattr(app, "inference_tasks", {"cam-1": _DummyTask()})
    monkeypatch.setattr(app, "roi_tasks", {})

    payload = app.build_dashboard_payload()
    runs = payload["roi_metrics"]["source_runs"]

    assert runs, "ควรมีข้อมูลเฟรมล่าสุด"
    assert runs[0]["roi_count"] == 2, "ควรรวม ROI ทั้งหมดแม้ไม่มี duration"
    assert runs[0]["modules"] == ["ocr"], "ควรแสดงชื่อโมดูลจากผลลัพธ์"
    assert payload["summary"]["total_roi"] == 2, "ยอดรวม ROI ต้องสอดคล้องกับที่กำลังประมวลผล"


def test_dashboard_latest_runs_uses_reported_roi_count(monkeypatch):
    import app

    notifications = [
        {
            "timestamp_epoch": time.time(),
            "cam_id": "cam-1",
            "source": "source-a",
            "count": 12,
            "results": [
                {"id": f"roi-{idx}", "name": f"ROI {idx}", "module": "ocr"}
                for idx in range(1, 6)
            ],
        }
    ]

    monkeypatch.setattr(app, "load_service_state", lambda: {"cam-1": {"source": "source-a"}})
    monkeypatch.setattr(app, "get_recent_notifications", lambda limit=None: list(notifications))
    monkeypatch.setattr(app, "np", object())

    monkeypatch.setattr(app, "camera_sources", {"cam-1": "source-a"})
    monkeypatch.setattr(app, "active_sources", {"cam-1": "Cam One"})
    monkeypatch.setattr(app, "inference_groups", {"cam-1": "all"})
    monkeypatch.setattr(app, "inference_intervals", {"cam-1": 1.0})
    monkeypatch.setattr(app, "inference_rois", {
        "cam-1": [
            {
                "type": "roi",
                "module": "ocr",
                "group": "all",
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 1, "y": 1},
                    {"x": 0, "y": 1},
                ],
            }
        ]
    })

    class _DummyTask:
        def done(self):
            return False

    monkeypatch.setattr(app, "inference_tasks", {"cam-1": _DummyTask()})
    monkeypatch.setattr(app, "roi_tasks", {})

    payload = app.build_dashboard_payload()
    runs = payload["roi_metrics"]["source_runs"]

    assert runs, "ควรมีข้อมูลเฟรมล่าสุด"
    assert runs[0]["roi_count"] == 12, "ควรใช้ค่าจำนวน ROI ที่รายงานจากการแจ้งเตือนล่าสุด"


def test_dashboard_uses_actual_fps_when_available(monkeypatch):
    import app

    base = time.time()
    notifications = [
        {
            "timestamp_epoch": base - 2,
            "cam_id": "cam-1",
            "source": "source-a",
            "results": [],
        },
        {
            "timestamp_epoch": base,
            "cam_id": "cam-1",
            "source": "source-a",
            "results": [],
        },
    ]

    monkeypatch.setattr(app, "load_service_state", lambda: {"cam-1": {"source": "source-a"}})
    monkeypatch.setattr(app, "get_recent_notifications", lambda limit=None: list(notifications))
    monkeypatch.setattr(app, "np", object())

    monkeypatch.setattr(app, "camera_sources", {"cam-1": "source-a"})
    monkeypatch.setattr(app, "active_sources", {"cam-1": "Cam One"})
    monkeypatch.setattr(app, "inference_groups", {"cam-1": "all"})
    monkeypatch.setattr(app, "inference_intervals", {"cam-1": 0.5})
    monkeypatch.setattr(app, "inference_rois", {"cam-1": []})

    class _DummyTask:
        def done(self):
            return False

    monkeypatch.setattr(app, "inference_tasks", {"cam-1": _DummyTask()})
    monkeypatch.setattr(app, "roi_tasks", {})

    payload = app.build_dashboard_payload()

    assert payload["summary"]["average_fps"] == pytest.approx(0.5, rel=1e-3)
    assert payload["summary"]["target_average_fps"] == pytest.approx(2.0, rel=1e-3)

    assert payload["cameras"], "ควรมีข้อมูลกล้อง"
    camera_entry = payload["cameras"][0]
    assert camera_entry["fps"] == pytest.approx(0.5, rel=1e-3)
    assert camera_entry["target_fps"] == pytest.approx(2.0, rel=1e-3)

