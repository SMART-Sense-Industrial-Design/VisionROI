import faulthandler
faulthandler.enable()

from quart import Quart, render_template, websocket, request, jsonify, redirect, Response
import asyncio
from src.utils.cv2_compat import cv2
from camera_worker import CameraWorker
try:
    import numpy as np
except Exception:
    np = None
import json
import base64
import shutil
import importlib.util
import os
import re
import sys
import argparse
import logging
from collections import defaultdict, deque
from dataclasses import dataclass

from types import ModuleType
from pathlib import Path
import contextlib
import inspect
import gc
import time
from datetime import datetime
from typing import Callable, Awaitable, Any, TypeVar
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from src.utils.logger import get_logger
from src.utils.memory import malloc_trim
try:
    from websockets.exceptions import ConnectionClosed
except Exception:
    ConnectionClosed = Exception

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

import signal  # signals

# === Runtime state ===
camera_workers: dict[str, CameraWorker | None] = {}
camera_sources: dict[str, int | str] = {}
camera_resolutions: dict[str, tuple[int | None, int | None]] = {}
camera_backends: dict[str, str] = {}
camera_locks: dict[str, asyncio.Lock] = {}
frame_queues: dict[str, asyncio.Queue["FramePacket | None"]] = {}
inference_tasks: dict[str, asyncio.Task | None] = {}
roi_frame_queues: dict[str, asyncio.Queue["FramePacket | None"]] = {}
roi_result_queues: dict[str, asyncio.Queue[str | None]] = {}
roi_tasks: dict[str, asyncio.Task | None] = {}
inference_rois: dict[str, list[dict]] = {}
active_sources: dict[str, str] = {}
save_roi_flags: dict[str, bool] = {}
inference_groups: dict[str, str | None] = {}
inference_intervals: dict[str, float] = {}
last_inference_times: dict[str, float] = {}
last_inference_outputs: dict[str, str] = {}
inference_result_timeouts: dict[str, float] = {}
inference_draw_page_boxes: dict[str, bool] = {}

STREAM_OUTAGE_GRACE = float(os.getenv("STREAM_OUTAGE_GRACE", "3.5") or 3.5)
STREAM_OUTAGE_RETRY = float(os.getenv("STREAM_OUTAGE_RETRY", "1.5") or 1.5)
_stream_disconnect_sent: dict[tuple[str, int], float] = {}

recent_notifications: list[dict[str, Any]] = []
MAX_RECENT_NOTIFICATIONS = 200

MQTT_CONFIG_FILE = "mqtt_configs.json"
mqtt_configs: dict[str, dict[str, Any]] = {}
mqtt_config_lock: asyncio.Lock | None = None
_missing_mqtt_configs_notified: set[str] = set()
_mqtt_dependency_missing_logged = False

STATE_FILE = "service_state.json"
PAGE_SCORE_THRESHOLD = 0.4
DEFAULT_PENDING_RESULT_TIMEOUT = float(os.getenv("INFERENCE_RESULT_TIMEOUT", "10.0") or 10.0)

app = Quart(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_ROI_DIR = os.path.realpath("data_sources")

# =========================
# Global inference queue & thread pool
# =========================
MAX_WORKERS = os.cpu_count() or 1
_INFERENCE_QUEUE: Queue[
    tuple[Callable, tuple, asyncio.Future, asyncio.AbstractEventLoop] | None
] = Queue(maxsize=MAX_WORKERS * 10)
_EXECUTOR: ThreadPoolExecutor | None = None


def _inference_worker():
    while True:
        item = _INFERENCE_QUEUE.get()
        if item is None:
            _INFERENCE_QUEUE.task_done()
            break
        func, args, fut, loop = item
        try:
            res = func(*args)
            if inspect.isawaitable(res):
                res = asyncio.run(res)
            loop.call_soon_threadsafe(fut.set_result, res)
        except Exception as e:
            loop.call_soon_threadsafe(fut.set_exception, e)
        finally:
            _INFERENCE_QUEUE.task_done()


def _start_inference_workers() -> None:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        for _ in range(MAX_WORKERS):
            _EXECUTOR.submit(_inference_worker)


def _stop_inference_workers() -> None:
    global _EXECUTOR
    if _EXECUTOR is None:
        return
    # เคลียร์งานที่ค้างและยกเลิก future
    while True:
        try:
            item = _INFERENCE_QUEUE.get_nowait()
        except Empty:
            break
        if item is None:
            _INFERENCE_QUEUE.task_done()
            continue
        _, _, fut, loop = item
        if fut and not fut.done():
            loop.call_soon_threadsafe(fut.cancel)
        _INFERENCE_QUEUE.task_done()
    # ส่งสัญญาณหยุดให้ worker
    for _ in range(MAX_WORKERS):
        try:
            _INFERENCE_QUEUE.put_nowait(None)
        except Exception:
            pass
    _EXECUTOR.shutdown(wait=False, cancel_futures=True)
    _EXECUTOR = None


_start_inference_workers()


def _free_cam_state(cam_id: str):
    """Drop all references for this camera to allow GC to reclaim big buffers."""
    # Drop conf/state that may hold refs
    camera_sources.pop(cam_id, None)
    camera_resolutions.pop(cam_id, None)
    camera_backends.pop(cam_id, None)
    active_sources.pop(cam_id, None)
    save_roi_flags.pop(cam_id, None)
    inference_groups.pop(cam_id, None)
    inference_rois.pop(cam_id, None)
    inference_intervals.pop(cam_id, None)
    last_inference_times.pop(cam_id, None)
    inference_result_timeouts.pop(cam_id, None)

    # Queues: drain & drop
    _drain_queue(frame_queues.pop(cam_id, None))
    _drain_queue(roi_frame_queues.pop(cam_id, None))
    _drain_queue(roi_result_queues.pop(cam_id, None))

    # Reset outage tracking for all queues of this camera
    for key in [k for k in _stream_disconnect_sent if k[0] == cam_id]:
        _stream_disconnect_sent.pop(key, None)

    # Locks
    camera_locks.pop(cam_id, None)

    # GC & trim
    gc.collect()
    malloc_trim()


# =========================
# Utils / Security helpers
# =========================
def _safe_in_base(base: str, target: str) -> bool:
    base = os.path.realpath(base)
    target = os.path.realpath(target)
    try:
        return os.path.commonpath([base]) == os.path.commonpath([base, target])
    except Exception:
        return False


def get_cam_lock(cam_id: str) -> asyncio.Lock:
    return camera_locks.setdefault(cam_id, asyncio.Lock())


def save_service_state() -> None:
    cam_ids = (
        set(camera_sources)
        | set(active_sources)
        | set(inference_tasks)
        | set(camera_backends)
    )
    data = {
        str(cam_id): {
            "source": camera_sources.get(cam_id),
            "resolution": list(camera_resolutions.get(cam_id, (None, None))),
            "backend": camera_backends.get(cam_id, "opencv"),
            "active_source": active_sources.get(cam_id, ""),
            "inference_running": bool(
                inference_tasks.get(cam_id) and not inference_tasks[cam_id].done()
            ),
            "inference_group": inference_groups.get(cam_id),
            "interval": inference_intervals.get(cam_id, 1.0),
            "result_timeout": inference_result_timeouts.get(cam_id,
                                                            DEFAULT_PENDING_RESULT_TIMEOUT),
            "draw_page_boxes": inference_draw_page_boxes.get(cam_id),
        }
        for cam_id in cam_ids
    }
    with contextlib.suppress(Exception):
        with open(STATE_FILE, "w") as f:
            json.dump({"cameras": data}, f)


def load_service_state() -> dict[str, dict[str, object]]:
    path = Path(STATE_FILE)
    if not path.exists():
        return {}
    with contextlib.suppress(Exception):
        with path.open("r") as f:
            data = json.load(f)
            cams = data.get("cameras")
            if isinstance(cams, dict):
                return cams
    return {}


def ensure_roi_ids(rois):
    if isinstance(rois, list):
        for idx, r in enumerate(rois):
            if isinstance(r, dict) and "id" not in r:
                r["id"] = str(idx + 1)
    return rois


def push_recent_notification(entry: dict[str, Any]) -> None:
    recent_notifications.append(entry)
    if len(recent_notifications) > MAX_RECENT_NOTIFICATIONS:
        del recent_notifications[: len(recent_notifications) - MAX_RECENT_NOTIFICATIONS]


def get_recent_notifications(limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit > 0:
        return list(recent_notifications[-limit:])
    return list(recent_notifications)


def get_mqtt_config_lock() -> asyncio.Lock:
    global mqtt_config_lock
    lock = mqtt_config_lock
    if lock is None:
        lock = asyncio.Lock()
        mqtt_config_lock = lock
    return lock


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _get_mqtt_logger(
    source_name: str | None = None, roi_name: str | None = None
) -> logging.Logger:
    source_value = str(source_name).strip() if source_name is not None else ""
    roi_value = str(roi_name).strip() if roi_name is not None else ""
    if source_value and roi_value:
        return get_logger("mqtt", source_value, roi_value, subdir="mqtt")
    if source_value:
        return get_logger("mqtt", source_value, subdir="mqtt")
    return get_logger("mqtt", subdir="mqtt")


def load_mqtt_configs_from_disk() -> dict[str, dict[str, Any]]:
    if not os.path.exists(MQTT_CONFIG_FILE):
        return {}
    try:
        with open(MQTT_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    configs: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]]
    if isinstance(data, list):
        items = [cfg for cfg in data if isinstance(cfg, dict)]
    elif isinstance(data, dict):
        items = []
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            entry = dict(cfg)
            entry.setdefault("name", name)
            items.append(entry)
    else:
        return {}

    for item in items:
        name = str(item.get("name") or "").strip()
        host = str(item.get("host") or "").strip()
        if not name or not host:
            continue
        cfg = dict(item)
        cfg["name"] = name
        cfg["host"] = host
        port_val = cfg.get("port", 1883)
        try:
            cfg["port"] = int(port_val)
        except (TypeError, ValueError):
            cfg["port"] = 1883
        qos_val = cfg.get("qos")
        if qos_val is not None:
            try:
                qos_int = int(qos_val)
            except (TypeError, ValueError):
                qos_int = 0
            cfg["qos"] = max(0, min(2, qos_int))
        configs[name] = cfg
    return configs


def save_mqtt_configs_to_disk(configs: dict[str, dict[str, Any]]) -> None:
    data: list[dict[str, Any]] = []
    for name, cfg in sorted(configs.items()):
        entry = dict(cfg)
        entry["name"] = name
        data.append(entry)
    tmp_path = f"{MQTT_CONFIG_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, MQTT_CONFIG_FILE)


def _sanitize_topic_segment(segment: Any) -> str:
    seg = str(segment or "").strip()
    seg = seg.replace(" ", "_")
    seg = re.sub(r"[^0-9A-Za-z_\-]+", "_", seg)
    return seg.strip("_")


def build_mqtt_topic(
    cfg: dict[str, Any],
    source: str,
    cam_id: str,
    group: str,
    roi_id: str,
) -> str:
    parts: list[str] = []
    base = cfg.get("base_topic")
    if base:
        for part in str(base).split("/"):
            sanitized = _sanitize_topic_segment(part)
            if sanitized:
                parts.append(sanitized)
    source_part = _sanitize_topic_segment(source)
    if not source_part:
        source_part = _sanitize_topic_segment(cam_id) or "camera"
    parts.append(source_part)
    group_part = _sanitize_topic_segment(group)
    if not group_part:
        group_part = "ungrouped"
    parts.append(group_part)
    roi_part = _sanitize_topic_segment(roi_id)
    if not roi_part:
        roi_part = "roi"
    parts.append(roi_part)
    return "/".join(parts)


def _public_mqtt_config(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    port_val = cfg.get("port", 1883)
    try:
        port_int = int(port_val)
    except (TypeError, ValueError):
        port_int = 1883
    qos_val = cfg.get("qos", 0)
    try:
        qos_int = int(qos_val)
    except (TypeError, ValueError):
        qos_int = 0
    keepalive_val = cfg.get("keepalive", 60)
    try:
        keepalive_int = int(keepalive_val)
    except (TypeError, ValueError):
        keepalive_int = 60
    publish_timeout_val = cfg.get("publish_timeout", 5.0)
    try:
        publish_timeout_float = float(publish_timeout_val)
    except (TypeError, ValueError):
        publish_timeout_float = 5.0
    return {
        "name": name,
        "host": cfg.get("host", ""),
        "port": port_int,
        "username": cfg.get("username", ""),
        "has_password": bool(cfg.get("password")),
        "base_topic": cfg.get("base_topic", ""),
        "client_id": cfg.get("client_id", ""),
        "qos": max(0, min(2, qos_int)),
        "retain": _truthy(cfg.get("retain")),
        "tls": _truthy(cfg.get("tls")),
        "keepalive": keepalive_int,
        "publish_timeout": publish_timeout_float,
    }


async def publish_roi_to_mqtt(
    config_name: str,
    cam_id: str,
    source_name: str,
    group_name: str,
    roi_id: str,
    payload: dict[str, Any],
) -> bool:
    global _mqtt_dependency_missing_logged
    roi_name_for_log = ""
    if isinstance(payload, dict):
        roi_name_for_log = str(payload.get("roi_name") or "").strip()
    logger = _get_mqtt_logger(source_name, roi_name_for_log or None)

    if mqtt is None:
        if not _mqtt_dependency_missing_logged:
            if logger is not None:
                logger.warning(
                    "paho-mqtt not available; MQTT publishing is disabled"
                )
            _mqtt_dependency_missing_logged = True
        return False

    cfg = mqtt_configs.get(config_name)
    if not cfg:
        if config_name and config_name not in _missing_mqtt_configs_notified:
            if logger is not None:
                logger.warning(
                    "MQTT config '%s' not found; skipping publish", config_name
                )
            _missing_mqtt_configs_notified.add(config_name)
        return False

    host = str(cfg.get("host") or "").strip()
    if not host:
        return False

    topic = build_mqtt_topic(cfg, source_name, cam_id, group_name, str(roi_id))
    if not topic:
        topic = "roi"

    message = json.dumps(payload, ensure_ascii=False)

    port_val = cfg.get("port", 1883)
    try:
        port = int(port_val)
    except (TypeError, ValueError):
        port = 1883

    keepalive_val = cfg.get("keepalive", 60)
    try:
        keepalive = int(keepalive_val)
    except (TypeError, ValueError):
        keepalive = 60

    qos_val = cfg.get("qos", 0)
    try:
        qos = int(qos_val)
    except (TypeError, ValueError):
        qos = 0
    qos = max(0, min(2, qos))

    publish_timeout_val = cfg.get("publish_timeout", 5.0)
    try:
        publish_timeout = max(0.1, float(publish_timeout_val))
    except (TypeError, ValueError):
        publish_timeout = 5.0

    username = str(cfg.get("username") or "").strip() or None
    password = cfg.get("password")
    if isinstance(password, str):
        password = password or None
    tls_enabled = _truthy(cfg.get("tls"))
    client_id = str(cfg.get("client_id") or "").strip() or None
    retain = _truthy(cfg.get("retain"))

    def _publish_sync() -> bool:
        client = mqtt.Client(client_id=client_id)  # type: ignore[arg-type]
        if username or password:
            client.username_pw_set(username, password)
        if tls_enabled:
            client.tls_set()
        client.connect(host, port, keepalive)
        client.loop_start()
        try:
            info = client.publish(topic, message, qos=qos, retain=retain)
            info.wait_for_publish(timeout=publish_timeout)
            rc = info.rc
        finally:
            with contextlib.suppress(Exception):
                client.loop_stop()
            with contextlib.suppress(Exception):
                client.disconnect()
        if rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"publish failed with code {rc}")
        return True

    try:
        await asyncio.to_thread(_publish_sync)
        if logger is not None:
            roi_label = roi_name_for_log or str(roi_id)
            logger.info(
                "Published MQTT message via '%s' to topic '%s' for ROI '%s'",
                config_name,
                topic,
                roi_label,
            )
        return True
    except Exception:
        if logger is not None:
            logger.exception(
                "Failed to publish MQTT message for config '%s'", config_name
            )
        return False


mqtt_configs = load_mqtt_configs_from_disk()


async def restore_service_state() -> None:
    cams = load_service_state()
    for cam_id, cfg in cams.items():
        camera_sources[cam_id] = cfg.get("source")
        res = cfg.get("resolution") or [None, None]
        camera_resolutions[cam_id] = (res[0], res[1])
        camera_backends[cam_id] = cfg.get("backend", "opencv")
        active_sources[cam_id] = cfg.get("active_source", "")
        group = cfg.get("inference_group")
        inference_intervals[cam_id] = cfg.get("interval", 1.0)
        timeout_val = cfg.get("result_timeout")
        if timeout_val is not None:
            try:
                inference_result_timeouts[cam_id] = max(
                    0.1, float(timeout_val)
                )
            except (TypeError, ValueError):
                inference_result_timeouts.pop(cam_id, None)
        else:
            inference_result_timeouts.pop(cam_id, None)
        draw_flag = cfg.get("draw_page_boxes")
        if draw_flag is None:
            inference_draw_page_boxes.pop(cam_id, None)
        else:
            inference_draw_page_boxes[cam_id] = bool(draw_flag)
        if cfg.get("inference_running"):
            await perform_start_inference(cam_id, group=group, save_state=False)
        elif group is not None:
            inference_groups[cam_id] = group
    save_service_state()


async def startup_tasks() -> None:
    await restore_service_state()


async def shutdown_tasks() -> None:
    _stop_inference_workers()


if hasattr(app, "before_serving"):
    app.before_serving(startup_tasks)

if hasattr(app, "after_serving"):
    app.after_serving(shutdown_tasks)


def _resolve_interval(cam_id: str, persisted: dict[str, dict[str, object]]) -> float | None:
    interval = inference_intervals.get(cam_id)
    if interval is not None:
        return float(interval)
    persisted_entry = persisted.get(cam_id, {})
    value = persisted_entry.get("interval")
    if isinstance(value, (int, float)):
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None
    return None


def build_dashboard_payload() -> dict[str, Any]:
    persisted = load_service_state()
    known_ids = set(persisted) | set(camera_sources) | set(active_sources)
    known_ids |= set(inference_rois)
    known_ids |= set(inference_tasks)
    known_ids |= set(roi_tasks)
    cameras: list[dict[str, Any]] = []
    group_accumulator: dict[str, dict[str, Any]] = {}
    page_jobs: list[dict[str, Any]] = []
    running_count = 0
    online_count = 0
    intervals: list[float] = []
    fps_values: list[float] = []
    roi_total_count = 0
    module_usage: dict[str, dict[str, Any]] = {}
    source_roi_map: dict[str, dict[str, Any]] = {}
    now_epoch = time.time()
    notifications_snapshot = get_recent_notifications()
    alerts_last_hour = 0
    module_duration_stats: dict[str, dict[str, float | int | None]] = {}
    source_duration_stats: dict[str, dict[str, float | int | None]] = {}
    latest_frame_snapshot: dict[str, Any] | None = None
    latest_source_frames: dict[str, dict[str, Any]] = {}

    def _clean_optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _to_iso(timestamp_value: object) -> str | None:
        if not timestamp_value:
            return None
        try:
            return datetime.fromtimestamp(float(timestamp_value)).isoformat()
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    def _safe_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    for notif in notifications_snapshot:
        try:
            ts = float(notif.get("timestamp_epoch", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        if now_epoch - ts <= 3600:
            alerts_last_hour += 1
        results = notif.get("results") or []
        roi_total_duration = 0.0
        frame_duration_val = _safe_float(notif.get("frame_duration"))
        frame_measurements: list[dict[str, Any]] = []
        for result in results:
            module_name = str(result.get("module") or "ไม่ระบุ")
            try:
                duration_val = float(result.get("duration"))
            except (TypeError, ValueError):
                duration_val = None
            if duration_val is None:
                continue
            roi_total_duration += duration_val
            frame_measurements.append(
                {
                    "name": module_name,
                    "module": module_name,
                    "duration": float(duration_val),
                    "roi_id": _clean_optional_str(result.get("id")),
                    "roi_name": _clean_optional_str(result.get("name")),
                    "cam_id": _clean_optional_str(notif.get("cam_id")),
                    "source": _clean_optional_str(notif.get("source")),
                }
            )
            module_entry = module_duration_stats.setdefault(
                module_name,
                {
                    "total": 0.0,
                    "count": 0,
                    "min": None,
                    "max": None,
                    "latest_duration": None,
                    "latest_timestamp": 0.0,
                    "fastest_duration": None,
                    "fastest_timestamp": 0.0,
                    "fastest_roi_id": None,
                    "fastest_roi_name": None,
                    "fastest_source": None,
                    "fastest_cam_id": None,
                    "slowest_duration": None,
                    "slowest_timestamp": 0.0,
                    "slowest_roi_id": None,
                    "slowest_roi_name": None,
                    "slowest_source": None,
                    "slowest_cam_id": None,
                },
            )
            module_entry["total"] += duration_val
            module_entry["count"] += 1
            module_entry["min"] = (
                duration_val
                if module_entry["min"] is None
                else min(float(module_entry["min"]), duration_val)
            )
            module_entry["max"] = (
                duration_val
                if module_entry["max"] is None
                else max(float(module_entry["max"]), duration_val)
            )
            if ts >= float(module_entry.get("latest_timestamp", 0.0) or 0.0):
                module_entry["latest_duration"] = duration_val
                module_entry["latest_timestamp"] = ts
                module_entry["latest_roi_id"] = _clean_optional_str(result.get("id"))
                module_entry["latest_roi_name"] = _clean_optional_str(result.get("name"))
                module_entry["latest_source"] = _clean_optional_str(notif.get("source"))
                module_entry["latest_cam_id"] = _clean_optional_str(notif.get("cam_id"))

            fastest_duration = module_entry.get("fastest_duration")
            if (
                fastest_duration is None
                or duration_val < float(fastest_duration or 0.0)
            ):
                module_entry["fastest_duration"] = duration_val
                module_entry["fastest_timestamp"] = ts
                module_entry["fastest_roi_id"] = _clean_optional_str(result.get("id"))
                module_entry["fastest_roi_name"] = _clean_optional_str(result.get("name"))
                module_entry["fastest_source"] = _clean_optional_str(notif.get("source"))
                module_entry["fastest_cam_id"] = _clean_optional_str(notif.get("cam_id"))

            slowest_duration = module_entry.get("slowest_duration")
            if (
                slowest_duration is None
                or duration_val > float(slowest_duration or 0.0)
            ):
                module_entry["slowest_duration"] = duration_val
                module_entry["slowest_timestamp"] = ts
                module_entry["slowest_roi_id"] = _clean_optional_str(result.get("id"))
                module_entry["slowest_roi_name"] = _clean_optional_str(result.get("name"))
                module_entry["slowest_source"] = _clean_optional_str(notif.get("source"))
                module_entry["slowest_cam_id"] = _clean_optional_str(notif.get("cam_id"))
        cam_id_clean = _clean_optional_str(notif.get("cam_id"))
        source_clean = _clean_optional_str(notif.get("source"))

        if frame_measurements:
            frame_timestamp_iso = _to_iso(ts) or (
                str(notif.get("timestamp")).strip()
                if isinstance(notif.get("timestamp"), str)
                and str(notif.get("timestamp")).strip()
                else None
            )
            fastest_frame = min(frame_measurements, key=lambda item: item["duration"])
            slowest_frame = max(frame_measurements, key=lambda item: item["duration"])
            fastest_entry = dict(fastest_frame)
            slowest_entry = dict(slowest_frame)
            if frame_timestamp_iso:
                fastest_entry["timestamp"] = frame_timestamp_iso
                slowest_entry["timestamp"] = frame_timestamp_iso
            latest_frame_snapshot = {
                "timestamp": frame_timestamp_iso,
                "cam_id": cam_id_clean,
                "source": source_clean,
                "fastest": fastest_entry,
                "slowest": slowest_entry,
            }
            frame_time_epoch = _safe_float(notif.get("frame_time"))
            result_time_epoch = _safe_float(notif.get("result_time"))
            if frame_duration_val is None and (
                frame_time_epoch is not None and result_time_epoch is not None
            ):
                diff = result_time_epoch - frame_time_epoch
                if diff >= 0:
                    frame_duration_val = diff
            if cam_id_clean:
                frame_modules = sorted(
                    {
                        str(item.get("module") or item.get("name") or "").strip()
                        for item in frame_measurements
                        if (item.get("module") or item.get("name"))
                    }
                )
                latest_snapshot = latest_source_frames.get(cam_id_clean)
                should_update = not latest_snapshot or ts >= float(
                    latest_snapshot.get("timestamp_epoch", 0.0) or 0.0
                )
                if should_update:
                    if frame_duration_val is None and roi_total_duration > 0:
                        duration_for_frame = roi_total_duration
                    else:
                        duration_for_frame = frame_duration_val
                    latest_source_frames[cam_id_clean] = {
                        "timestamp": frame_timestamp_iso,
                        "timestamp_epoch": ts,
                        "cam_id": cam_id_clean,
                        "source": source_clean,
                        "roi_count": len(frame_measurements),
                        "modules": frame_modules,
                        "fastest": fastest_entry,
                        "slowest": slowest_entry,
                        "frame_duration": duration_for_frame,
                        "total_duration": roi_total_duration,
                    }
        if frame_duration_val is None:
            frame_time_epoch = _safe_float(notif.get("frame_time"))
            result_time_epoch = _safe_float(notif.get("result_time"))
            if frame_time_epoch is not None and result_time_epoch is not None:
                diff = result_time_epoch - frame_time_epoch
                if diff >= 0:
                    frame_duration_val = diff
        duration_for_source: float | None
        if frame_duration_val is not None:
            duration_for_source = frame_duration_val
        elif roi_total_duration > 0:
            duration_for_source = roi_total_duration
        else:
            duration_for_source = None
        cam_id = notif.get("cam_id")
        if cam_id and duration_for_source is not None:
            source_entry = source_duration_stats.setdefault(
                str(cam_id),
                {
                    "total": 0.0,
                    "count": 0,
                    "min": None,
                    "max": 0.0,
                    "latest_duration": None,
                    "latest_timestamp": 0.0,
                },
            )
            source_entry["total"] += duration_for_source
            source_entry["count"] += 1
            source_entry["min"] = (
                duration_for_source
                if source_entry["min"] is None
                else min(float(source_entry["min"]), duration_for_source)
            )
            source_entry["max"] = max(
                float(source_entry.get("max", 0.0) or 0.0), duration_for_source
            )
            if ts >= float(source_entry.get("latest_timestamp", 0.0) or 0.0):
                source_entry["latest_duration"] = duration_for_source
                source_entry["latest_timestamp"] = ts
    for cam_id in sorted(known_ids, key=str):
        task = inference_tasks.get(cam_id)
        inference_running = bool(task and not task.done())
        roi_task = roi_tasks.get(cam_id)
        roi_running = bool(roi_task and not roi_task.done())
        persisted_entry = persisted.get(cam_id, {})
        interval = _resolve_interval(cam_id, persisted)
        if interval:
            intervals.append(interval)
            if interval > 0:
                fps_values.append(round(1.0 / interval, 3))
        active_group = inference_groups.get(cam_id) or persisted_entry.get("inference_group")
        rois_for_cam = inference_rois.get(cam_id, []) or []
        forced_group = inference_groups.get(cam_id)
        selected_group = forced_group if forced_group and forced_group != "all" else None
        last_output = last_inference_outputs.get(cam_id, selected_group or "")
        if selected_group is None:
            if forced_group and forced_group != "all":
                selected_group = forced_group
            else:
                selected_group = last_output

        def _roi_matches_group(roi_group: Any) -> bool:
            if forced_group == "all":
                return True
            if forced_group and forced_group != "all":
                return roi_group == forced_group
            if selected_group:
                return roi_group == selected_group
            return False

        active_rois: list[dict[str, Any]] = []
        if inference_running:
            for roi_entry in rois_for_cam:
                if not isinstance(roi_entry, dict):
                    continue
                if roi_entry.get("type") != "roi":
                    continue
                if not _roi_matches_group(roi_entry.get("group")):
                    continue
                active_rois.append(roi_entry)

        roi_count = len(active_rois)
        roi_total_count += roi_count
        unique_modules: set[str] = set()
        for roi_entry in active_rois:
            module_name = str(roi_entry.get("module") or "ไม่ระบุ")
            unique_modules.add(module_name)
            module_entry = module_usage.setdefault(
                module_name,
                {"count": 0, "sources": set(), "types": set()},
            )
            module_entry["count"] = int(module_entry.get("count", 0)) + 1
            module_entry.setdefault("sources", set()).add(cam_id)
            roi_type = roi_entry.get("type")
            if roi_type:
                module_entry.setdefault("types", set()).add(str(roi_type))
        source_roi_map[cam_id] = {
            "cam_id": cam_id,
            "display_name": (
                active_sources.get(cam_id, persisted_entry.get("active_source", ""))
                or str(cam_id)
            ),
            "group": active_group,
            "roi_count": roi_count,
            "modules": sorted(unique_modules),
            "interval": interval,
            "source": camera_sources.get(cam_id, persisted_entry.get("source", "")),
            "inference_running": inference_running,
        }
        last_result = last_inference_outputs.get(cam_id, "")
        last_activity = last_inference_times.get(cam_id)
        if isinstance(last_activity, (int, float)) and last_activity > 0:
            last_activity_iso = datetime.fromtimestamp(last_activity).isoformat()
        else:
            last_activity_iso = None
        status: str
        if inference_running:
            status = "กำลังประมวลผล"
        elif roi_running:
            status = "สตรีม ROI"
        else:
            status = "หยุดทำงาน"
        if inference_running or roi_running:
            online_count += 1
        if inference_running:
            running_count += 1
        recent_alerts_for_cam = [
            notif
            for notif in notifications_snapshot
            if notif.get("cam_id") == cam_id
        ]
        alerts_count = len(recent_alerts_for_cam)
        resolution_raw = persisted_entry.get("resolution")
        if isinstance(resolution_raw, (list, tuple)) and len(resolution_raw) == 2:
            try:
                width = int(resolution_raw[0]) if resolution_raw[0] else None
            except (TypeError, ValueError):
                width = None
            try:
                height = int(resolution_raw[1]) if resolution_raw[1] else None
            except (TypeError, ValueError):
                height = None
        else:
            width = height = None

        camera_entry = {
            "cam_id": cam_id,
            "source": camera_sources.get(cam_id, persisted_entry.get("source", "")),
            "name": active_sources.get(cam_id, persisted_entry.get("active_source", "")),
            "backend": camera_backends.get(cam_id, persisted_entry.get("backend", "opencv")),
            "group": active_group,
            "interval": interval,
            "fps": (round(1.0 / interval, 3) if interval and interval > 0 else None),
            "status": status,
            "last_output": last_result,
            "last_activity": last_activity_iso,
            "alerts_count": alerts_count,
            "roi_count": roi_count,
            "inference_running": inference_running,
            "roi_running": roi_running,
            "resolution": {"width": width, "height": height},
            "snapshot_url": f"/ws_snapshot/{cam_id}?ts={int(now_epoch)}"
            if inference_running
            else None,
        }
        cameras.append(camera_entry)

        group_key = active_group or "ไม่ระบุกลุ่ม"
        group_entry = group_accumulator.setdefault(
            group_key,
            {
                "name": group_key,
                "cameras": 0,
                "running": 0,
                "online": 0,
                "roi": 0,
                "fps_total": 0.0,
                "fps_count": 0,
                "alerts": 0,
            },
        )
        group_entry["cameras"] += 1
        if inference_running:
            group_entry["running"] += 1
        if inference_running or roi_running:
            group_entry["online"] += 1
        roi_total = group_entry.get("roi", 0) or 0
        group_entry["roi"] = roi_total + (camera_entry["roi_count"] or 0)
        if camera_entry["fps"] is not None:
            group_entry["fps_total"] += camera_entry["fps"]
            group_entry["fps_count"] += 1
        group_entry["alerts"] += alerts_count

        cam_id_lower = str(cam_id).lower()
        is_page_job = cam_id_lower.startswith("page_")
        if is_page_job:
            page_jobs.append(
                {
                    "cam_id": cam_id,
                    "title": camera_entry["name"] or cam_id.replace("page_", "") or cam_id,
                    "group": active_group,
                    "interval": interval,
                    "fps": camera_entry["fps"],
                    "status": status,
                    "last_output": last_result,
                    "last_activity": last_activity_iso,
                    "alerts_count": alerts_count,
                    "inference_running": inference_running,
                }
            )

    average_interval = sum(intervals) / len(intervals) if intervals else 0.0
    average_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    recent_alerts = notifications_snapshot[-20:]
    group_entries: list[dict[str, Any]] = []
    running_groups = 0
    for group_entry in group_accumulator.values():
        fps_count = group_entry.pop("fps_count", 0)
        fps_total = group_entry.pop("fps_total", 0.0)
        average_group_fps = fps_total / fps_count if fps_count else 0.0
        if group_entry.get("running"):
            running_groups += 1
        group_entries.append(
            {
                **group_entry,
                "average_fps": average_group_fps,
            }
        )

    page_jobs_running = sum(1 for job in page_jobs if job.get("inference_running"))
    summary = {
        "total_cameras": len(known_ids),
        "online_cameras": online_count,
        "inference_running": running_count,
        "alerts_last_hour": alerts_last_hour,
        "average_interval": average_interval,
        "average_fps": average_fps,
        "total_groups": len(group_entries),
        "running_groups": running_groups,
        "page_jobs": len(page_jobs),
        "page_jobs_running": page_jobs_running,
    }
    summary["total_roi"] = roi_total_count

    module_details: list[dict[str, Any]] = []
    fastest_measurement: dict[str, Any] | None = None
    slowest_measurement: dict[str, Any] | None = None

    for module_name, info in module_usage.items():
        durations = module_duration_stats.get(module_name, {})
        count = int(info.get("count", 0))
        duration_count = int(durations.get("count", 0) or 0)
        average_duration = (
            (durations.get("total", 0.0) or 0.0) / duration_count
            if duration_count
            else None
        )
        latest_timestamp_iso = _to_iso(durations.get("latest_timestamp"))
        fastest_duration = durations.get("fastest_duration")
        fastest_timestamp_iso = _to_iso(durations.get("fastest_timestamp"))
        slowest_duration = durations.get("slowest_duration")
        slowest_timestamp_iso = _to_iso(durations.get("slowest_timestamp"))

        detail_entry = {
            "name": module_name,
            "roi_count": count,
            "source_count": len(info.get("sources", set())),
            "sources": sorted(str(src) for src in info.get("sources", set())),
            "types": sorted(str(t) for t in info.get("types", set()) if t),
            "average_duration": average_duration,
            "min_duration": durations.get("min"),
            "max_duration": durations.get("max"),
            "sample_count": duration_count,
            "latest_duration": durations.get("latest_duration"),
            "latest_timestamp": latest_timestamp_iso,
            "latest_roi_id": durations.get("latest_roi_id"),
            "latest_roi_name": durations.get("latest_roi_name"),
            "latest_cam_id": durations.get("latest_cam_id"),
            "latest_source": durations.get("latest_source"),
            "fastest_duration": fastest_duration,
            "fastest_timestamp": fastest_timestamp_iso,
            "fastest_roi_id": durations.get("fastest_roi_id"),
            "fastest_roi_name": durations.get("fastest_roi_name"),
            "fastest_cam_id": durations.get("fastest_cam_id"),
            "fastest_source": durations.get("fastest_source"),
            "slowest_duration": slowest_duration,
            "slowest_timestamp": slowest_timestamp_iso,
            "slowest_roi_id": durations.get("slowest_roi_id"),
            "slowest_roi_name": durations.get("slowest_roi_name"),
            "slowest_cam_id": durations.get("slowest_cam_id"),
            "slowest_source": durations.get("slowest_source"),
        }

        module_details.append(detail_entry)

        if fastest_duration is not None:
            candidate_fastest = {
                "name": module_name,
                "module": module_name,
                "duration": float(fastest_duration),
                "timestamp": fastest_timestamp_iso,
                "roi_id": durations.get("fastest_roi_id"),
                "roi_name": durations.get("fastest_roi_name"),
                "cam_id": durations.get("fastest_cam_id"),
                "source": durations.get("fastest_source"),
            }
            if (
                fastest_measurement is None
                or candidate_fastest["duration"]
                < fastest_measurement.get("duration", float("inf"))
            ):
                fastest_measurement = candidate_fastest

        if slowest_duration is not None:
            candidate_slowest = {
                "name": module_name,
                "module": module_name,
                "duration": float(slowest_duration),
                "timestamp": slowest_timestamp_iso,
                "roi_id": durations.get("slowest_roi_id"),
                "roi_name": durations.get("slowest_roi_name"),
                "cam_id": durations.get("slowest_cam_id"),
                "source": durations.get("slowest_source"),
            }
            if (
                slowest_measurement is None
                or candidate_slowest["duration"]
                > slowest_measurement.get("duration", float("-inf"))
            ):
                slowest_measurement = candidate_slowest

    module_details.sort(key=lambda item: (-item.get("roi_count", 0), item.get("name", "")))

    source_details: list[dict[str, Any]] = []
    source_run_entries: list[dict[str, Any]] = []

    for cam_id, info in source_roi_map.items():
        perf = source_duration_stats.get(str(cam_id), {})
        sample_count = int(perf.get("count", 0) or 0)
        average_duration = (
            (perf.get("total", 0.0) or 0.0) / sample_count if sample_count else None
        )
        interval_val = info.get("interval")
        latest_duration_raw = perf.get("latest_duration")
        try:
            latest_duration_val = (
                float(latest_duration_raw)
                if latest_duration_raw is not None
                else None
            )
        except (TypeError, ValueError):
            latest_duration_val = None

        meets_interval: bool | None
        interval_gap: float | None
        latest_interval_ratio: float | None
        if (
            isinstance(interval_val, (int, float))
            and interval_val is not None
            and latest_duration_val is not None
        ):
            meets_interval = latest_duration_val <= float(interval_val)
            interval_gap = float(interval_val) - latest_duration_val
            latest_interval_ratio = (
                latest_duration_val / float(interval_val)
                if float(interval_val) > 0
                else None
            )
        else:
            meets_interval = None
            interval_gap = None
            latest_interval_ratio = None
        source_details.append(
            {
                **info,
                "average_duration": average_duration,
                "max_duration": perf.get("max"),
                "min_duration": perf.get("min"),
                "latest_duration": latest_duration_val,
                "latest_completed_at": _to_iso(perf.get("latest_timestamp")),
                "samples": sample_count,
                "meets_interval": meets_interval,
                "interval_gap": interval_gap,
                "latest_interval_ratio": latest_interval_ratio,
            }
        )

        frame_snapshot = latest_source_frames.get(str(cam_id)) or latest_source_frames.get(
            _clean_optional_str(cam_id)
        )
        if frame_snapshot:
            source_run_entries.append(
                {
                    "cam_id": str(cam_id),
                    "display_name": info.get("display_name") or str(cam_id),
                    "group": info.get("group"),
                    "source": frame_snapshot.get("source") or info.get("source"),
                    "timestamp": frame_snapshot.get("timestamp"),
                    "timestamp_epoch": frame_snapshot.get("timestamp_epoch"),
                    "roi_count": frame_snapshot.get("roi_count"),
                    "modules": frame_snapshot.get("modules", []),
                    "frame_duration": frame_snapshot.get("frame_duration"),
                    "total_duration": frame_snapshot.get("total_duration"),
                    "fastest": frame_snapshot.get("fastest"),
                    "slowest": frame_snapshot.get("slowest"),
                    "interval": info.get("interval"),
                }
            )

    source_run_entries.sort(
        key=lambda item: (
            -float(item.get("timestamp_epoch", 0.0) or 0.0)
            if item.get("timestamp_epoch") is not None
            else float("inf")
        )
    )

    for entry in source_run_entries:
        entry.pop("timestamp_epoch", None)

    source_details.sort(
        key=lambda item: (
            -int(item.get("roi_count", 0) or 0),
            str(item.get("cam_id", "")),
        )
    )

    roi_metrics = {
        "total_roi": roi_total_count,
        "unique_modules": len(module_usage),
        "sources_with_roi": sum(1 for item in source_details if item.get("roi_count", 0)),
        "module_details": module_details,
        "module_performance": {
            "fastest": fastest_measurement,
            "slowest": slowest_measurement,
            "latest_frame": latest_frame_snapshot,
        },
        "source_details": source_details,
        "source_runs": source_run_entries,
    }
    return {
        "summary": summary,
        "cameras": cameras,
        "alerts": recent_alerts,
        "groups": sorted(group_entries, key=lambda g: (-g.get("running", 0), -g.get("cameras", 0))),
        "page_jobs": sorted(
            page_jobs,
            key=lambda job: (
                0 if job.get("inference_running") else 1,
                job.get("cam_id") or "",
            ),
        ),
        "roi_metrics": roi_metrics,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


async def build_dashboard_response() -> dict[str, Any]:
    payload = build_dashboard_payload()
    return payload


# ========== stop helpers ==========
async def _safe_stop(cam_id: str,
                     task_dict: dict[str, asyncio.Task | None],
                     queue_dict: dict[str, asyncio.Queue[bytes | None]] | None):
    try:
        await asyncio.wait_for(
            stop_camera_task(cam_id, task_dict, queue_dict),
            timeout=1.2
        )
    except asyncio.TimeoutError:
        worker = camera_workers.get(cam_id)
        if isinstance(worker, CameraWorker):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(worker.ensure_ffmpeg_gone)
    except Exception:
        worker = camera_workers.get(cam_id)
        if isinstance(worker, CameraWorker):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(worker.ensure_ffmpeg_gone)
    finally:
        worker = camera_workers.get(cam_id)
        if isinstance(worker, CameraWorker):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(worker.ensure_ffmpeg_gone)


# ========== Graceful shutdown (fast & robust) ==========
_SHUTTING_DOWN = False  # prevent reentry

async def _shutdown_cleanup_concurrent():
    # run after_serving cleanup but cap time
    if "_shutdown_cleanup" in globals():
        with contextlib.suppress(Exception, asyncio.TimeoutError):
            await asyncio.wait_for(_shutdown_cleanup(), timeout=2.0)

async def _graceful_exit():
    """Cancel tasks, unblock queues, release cameras, then exit fast."""
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return
    _SHUTTING_DOWN = True
    with contextlib.suppress(Exception):
        await _shutdown_cleanup_concurrent()
    # allow 202 response to flush
    await asyncio.sleep(0.05)
    os._exit(0)

def _sync_signal_handler(signum, frame):
    """Ensure SIGTERM/SIGINT triggers the same fast path."""
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(lambda: asyncio.create_task(_graceful_exit()))
    except RuntimeError:
        os._exit(0)

# install sync handlers early (more reliable than loop.add_signal_handler for our case)
signal.signal(signal.SIGTERM, _sync_signal_handler)
signal.signal(signal.SIGINT,  _sync_signal_handler)
# Ignore SIGPIPE to avoid spurious Broken pipe errors when downstream closes
signal.signal(signal.SIGPIPE, signal.SIG_IGN)


# Close all tasks/cameras concurrently at shutdown (Quart lifespan)
if hasattr(app, "after_serving"):
    @app.after_serving
    async def _shutdown_cleanup():
        await asyncio.gather(
            *[_safe_stop(cid, inference_tasks, frame_queues) for cid in list(inference_tasks.keys())],
            *[_safe_stop(cid, roi_tasks, roi_frame_queues) for cid in list(roi_tasks.keys())],
            return_exceptions=True
        )
        await asyncio.gather(
            *[
                asyncio.to_thread(worker.ensure_ffmpeg_gone)
                for worker in list(camera_workers.values())
                if isinstance(worker, CameraWorker)
            ],
            return_exceptions=True,
        )
        _stop_inference_workers()
        for cam_id, q in list(roi_result_queues.items()):
            with contextlib.suppress(Exception):
                await q.put(None)
            roi_result_queues.pop(cam_id, None)
        gc.collect()
        malloc_trim()  # trim heap after shutdown cleanup


# =========================
# Routes (pages)
# =========================
@app.route("/")
async def index():
    return redirect("/home")


@app.route("/home")
async def home():
    return await render_template("home.html")


@app.route("/docs")
async def docs():
    return await render_template("docs.html")


@app.route("/roi")
async def roi_page():
    return await render_template("roi_selection.html")


@app.route("/inference")
async def inference() -> str:
    return await render_template("inference.html")


@app.route("/inference_page")
async def inference_page() -> str:
    return await render_template("inference_page.html")


# Health & quit endpoints (used by systemd)
@app.get("/_healthz")
async def _healthz():
    return "ok", 200


@app.post("/_quit")
async def _quit():
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return Response("already shutting down", status=202, mimetype="text/plain")
    asyncio.create_task(_graceful_exit())
    return Response("shutting down", status=202, mimetype="text/plain")


# =========================
# Inference module loader
# =========================
def load_custom_module(name: str) -> ModuleType | None:
    path = Path("inference_modules") / name / "custom.py"
    if not path.exists():
        return None
    module_name = f"custom_{name}"
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =========================
# Queues getters
# =========================
def get_frame_queue(cam_id: str) -> asyncio.Queue['FramePacket | None']:
    return frame_queues.setdefault(cam_id, asyncio.Queue(maxsize=1))


def get_roi_frame_queue(cam_id: str) -> asyncio.Queue['FramePacket | None']:
    return roi_frame_queues.setdefault(cam_id, asyncio.Queue(maxsize=1))


def get_roi_result_queue(cam_id: str) -> asyncio.Queue[str | None]:
    return roi_result_queues.setdefault(cam_id, asyncio.Queue(maxsize=100))


def _drain_queue(queue: asyncio.Queue[Any] | None) -> None:
    """Remove all pending items from *queue* if it exists."""
    if queue is None:
        return
    with contextlib.suppress(asyncio.QueueEmpty):
        while True:
            queue.get_nowait()


T = TypeVar("T")


def _replace_with_latest(
    queue: asyncio.Queue[T] | None,
    item: T,
) -> None:
    """Drop every pending entry in *queue* before inserting *item*.

    ใช้สำหรับ queue ที่ส่งข้อมูลสตรีมไปยัง WebSocket เพื่อให้ผู้ใช้ได้รับ
    เฉพาะข้อมูลล่าสุด ลดปัญหา backlog เมื่อเครือข่ายช้า
    """
    if queue is None:
        return
    with contextlib.suppress(asyncio.QueueEmpty):
        while True:
            queue.get_nowait()
    queue.put_nowait(item)


# =========================
# Frame readers
# =========================
@dataclass(slots=True)
class FramePacket:
    payload: bytes
    created_at: float
    frame_time: float | None = None


async def read_and_queue_frame(
    cam_id: str,
    queue: asyncio.Queue["FramePacket | None"],
    frame_processor=None,
) -> None:
    worker = camera_workers.get(cam_id)
    if worker is None:
        await asyncio.sleep(0.01)
        return
    queue_key = (cam_id, id(queue))
    now_monotonic = time.monotonic()
    frame = await worker.read()
    if frame is None:
        last_frame_ts = (
            worker.last_frame_timestamp()
            if hasattr(worker, "last_frame_timestamp")
            else 0.0
        )
        if (
            STREAM_OUTAGE_GRACE > 0
            and last_frame_ts > 0.0
            and now_monotonic - last_frame_ts >= STREAM_OUTAGE_GRACE
        ):
            last_signal = _stream_disconnect_sent.get(queue_key, 0.0)
            retry_after = max(STREAM_OUTAGE_RETRY, 0.5) if STREAM_OUTAGE_RETRY >= 0 else 0.0
            if last_signal <= 0.0 or now_monotonic - last_signal >= retry_after:
                try:
                    _replace_with_latest(queue, None)
                except Exception:
                    pass
                else:
                    if retry_after > 0:
                        _stream_disconnect_sent[queue_key] = now_monotonic
        await asyncio.sleep(0.01)
        return
    if np is not None and hasattr(frame, "size") and frame.size == 0:
        await asyncio.sleep(0.01)
        return
    frame_time = time.time()
    if frame_processor:
        processed = await frame_processor(frame, frame_time)
        if isinstance(processed, tuple):
            frame, maybe_ts = processed
            if frame is None:
                return
            if isinstance(maybe_ts, (int, float)):
                frame_time = float(maybe_ts)
        else:
            frame = processed
        if frame is None:
            return
    try:
        encoded, buffer = await asyncio.to_thread(
            cv2.imencode, '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
    except Exception:
        await asyncio.sleep(0.01)
        return
    if not encoded or buffer is None:
        await asyncio.sleep(0.01)
        return
    frame_bytes = buffer.tobytes() if hasattr(buffer, "tobytes") else buffer
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    packet = FramePacket(payload=frame_bytes, created_at=time.monotonic(), frame_time=frame_time)
    _stream_disconnect_sent.pop(queue_key, None)
    await queue.put(packet)
    await asyncio.sleep(0)


# =========================
# Loops
# =========================
async def run_inference_loop(cam_id: str):
    _start_inference_workers()
    module_cache: dict[str, tuple[ModuleType | None, bool, bool, bool]] = {}
    pending_results: defaultdict[
        tuple[str, float],
        defaultdict[str | int, dict[str, Any]],
    ] = defaultdict(lambda: defaultdict(dict))
    pending_expected: dict[tuple[str, float], int] = {}
    pending_deadlines: dict[tuple[str, float], float] = {}
    pending_ready: set[tuple[str, float]] = set()
    pending_context: dict[tuple[str, float], dict[str, Any]] = {}
    pending_extensions: dict[tuple[str, float], int] = {}
    pending_roi_start: dict[tuple[str, float], dict[str | int, float]] = {}

    def get_pending_timeout() -> float:
        timeout = inference_result_timeouts.get(cam_id)
        if timeout is None:
            timeout = DEFAULT_PENDING_RESULT_TIMEOUT
        try:
            timeout_val = float(timeout)
        except (TypeError, ValueError):
            timeout_val = DEFAULT_PENDING_RESULT_TIMEOUT
        if timeout_val <= 0:
            return 0.1
        return timeout_val

    def cleanup_pending_results(force: bool = False) -> None:
        if force:
            keys = list(
                set(pending_results.keys())
                | set(pending_deadlines.keys())
                | set(pending_ready)
                | set(pending_extensions.keys())
                | set(pending_roi_start.keys())
            )
        else:
            now = time.time()
            keys: list[tuple[str, float]] = []
            for key, expiry in list(pending_deadlines.items()):
                if expiry > now:
                    continue
                total = pending_expected.get(key, 0)
                completed = len(pending_results.get(key, {}))
                if total > completed:
                    pending_deadlines[key] = now + get_pending_timeout()
                    pending_extensions[key] = pending_extensions.get(key, 0) + 1
                    continue
                keys.append(key)
        for key in keys:
            pending_results.pop(key, None)
            pending_expected.pop(key, None)
            pending_deadlines.pop(key, None)
            pending_ready.discard(key)
            pending_context.pop(key, None)
            pending_extensions.pop(key, None)
            pending_roi_start.pop(key, None)

    def flush_pending_result(key: tuple[str, float]) -> None:
        results_map = pending_results.pop(key, None)
        pending_expected.pop(key, None)
        pending_deadlines.pop(key, None)
        pending_ready.discard(key)
        pending_extensions.pop(key, None)
        pending_roi_start.pop(key, None)
        if not results_map:
            return
        results_map = dict(results_map)
        if not results_map:
            return
        ordered_keys = sorted(results_map.keys(), key=lambda rid: str(rid))
        results_list = [results_map[rid] for rid in ordered_keys]
        if not results_list:
            return
        now_time = time.time()
        payload_dict = {
            'frame_time': key[1],
            'result_time': now_time,
            'results': results_list,
            'cam_id': key[0],
        }
        frame_duration: float | None
        try:
            frame_duration = float(payload_dict['result_time']) - float(payload_dict['frame_time'])
        except (TypeError, ValueError):
            frame_duration = None
        if frame_duration is not None:
            if frame_duration < 0:
                frame_duration = 0.0
            payload_dict['frame_duration'] = frame_duration
        context = pending_context.pop(key, None) or {}
        source_name = context.get('source') or active_sources.get(key[0], '')
        group_name = context.get('group') or ''
        if group_name:
            payload_dict['group'] = group_name
        if source_name:
            payload_dict['source'] = source_name
        try:
            if source_name:
                agg_logger = get_logger('aggregated_roi', source_name)
            else:
                agg_logger = get_logger('aggregated_roi')
            if agg_logger is not None:
                sanitized_results: list[dict[str, Any]] = []
                for res in results_list:
                    roi_identifier = res.get('id')
                    if roi_identifier is None:
                        continue
                    sanitized_entry = {
                        'id': str(roi_identifier),
                        'name': str(res.get('name') or ''),
                        'text': str(res.get('text') or ''),
                        'module': str(res.get('module') or ''),
                    }
                    duration_val = res.get('duration')
                    if isinstance(duration_val, (int, float)):
                        sanitized_entry['duration'] = float(duration_val)
                    sanitized_results.append(sanitized_entry)
                if sanitized_results:
                    log_entry = {
                        'frame_time': payload_dict['frame_time'],
                        'result_time': payload_dict['result_time'],
                        'cam_id': key[0],
                        'group': group_name,
                        'source': source_name,
                        'results': sanitized_results,
                    }
                    if frame_duration is not None:
                        log_entry['frame_duration'] = frame_duration
                    agg_logger.info(
                        "AGGREGATED_ROI %s",
                        json.dumps(log_entry, ensure_ascii=False),
                    )
                    try:
                        ts = datetime.fromtimestamp(
                            float(payload_dict['result_time'])
                        ).isoformat()
                    except Exception:
                        ts = datetime.utcnow().isoformat()
                    recent_entry = {
                        "cam_id": key[0],
                        "group": group_name or "",
                        "source": source_name or "",
                        "count": len(sanitized_results),
                        "results": sanitized_results[:5],
                        "timestamp": ts,
                        "timestamp_epoch": float(
                            payload_dict.get('result_time', time.time())
                        ),
                    }
                    if frame_duration is not None:
                        recent_entry["frame_duration"] = frame_duration
                    recent_entry["frame_time"] = payload_dict.get('frame_time')
                    recent_entry["result_time"] = payload_dict.get('result_time')
                    push_recent_notification(recent_entry)
        except Exception:
            pass
        try:
            q = get_roi_result_queue(key[0])
            payload = json.dumps(payload_dict)
            _replace_with_latest(q, payload)
        except Exception:
            pass

    async def process_frame(frame, frame_timestamp):
        cleanup_pending_results()
        rois = inference_rois.get(cam_id, [])
        forced_group = inference_groups.get(cam_id)
        selected_group = (
            forced_group if forced_group and forced_group != 'all' else None
        )
        source_name = active_sources.get(cam_id, '')
        draw_page_boxes = inference_draw_page_boxes.get(cam_id, True)
        if not rois:
            return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), frame_timestamp
        now = time.time()
        interval = inference_intervals.get(cam_id, 1.0)
        meets_interval = now - last_inference_times.get(cam_id, 0.0) >= interval

        save_flag = bool(save_roi_flags.get(cam_id))
        frame_time = (
            float(frame_timestamp)
            if isinstance(frame_timestamp, (int, float))
            else time.time()
        )
        best_score = -1.0
        scores: list[dict[str, float | str]] = []
        has_page = False
        output = last_inference_outputs.get(cam_id, selected_group or '')

        for i, r in enumerate(rois):
            if np is None or r.get('type') != 'page':
                continue
            has_page = True
            pts = r.get('points', [])
            if len(pts) != 4:
                continue
            src = np.array([[p['x'], p['y']] for p in pts], dtype=np.float32)
            color = (0, 255, 0)
            width_a = np.linalg.norm(src[0] - src[1])
            width_b = np.linalg.norm(src[2] - src[3])
            max_w = int(max(width_a, width_b))
            height_a = np.linalg.norm(src[0] - src[3])
            height_b = np.linalg.norm(src[1] - src[2])
            max_h = int(max(height_a, height_b))
            dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            roi = await asyncio.to_thread(cv2.warpPerspective, frame, matrix, (max_w, max_h))
            template = r.get('_template')
            if template is not None:
                try:
                    roi_gray = await asyncio.to_thread(cv2.cvtColor, roi, cv2.COLOR_BGR2GRAY)
                except Exception:
                    roi_gray = None
                if roi_gray is not None:
                    if roi_gray.shape != template.shape:
                        roi_gray = await asyncio.to_thread(cv2.resize, roi_gray, (template.shape[1], template.shape[0]))
                    try:
                        res = await asyncio.to_thread(cv2.matchTemplate, roi_gray, template, cv2.TM_CCOEFF_NORMED)
                        score = float(res[0][0])
                        scores.append({'page': r.get('page', ''), 'score': score})
                        if score > best_score:
                            best_score = score
                            output = r.get('page', '')
                    except Exception:
                        pass
            if draw_page_boxes:
                cv2.polylines(frame, [src.astype(int)], True, color, 2)
                label_pt = src[0].astype(int)
                cv2.putText(
                    frame,
                    str(r.get('id', i + 1)),
                    (int(label_pt[0]), max(0, int(label_pt[1]) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        if not has_page:
            output = forced_group or ''
        elif best_score <= PAGE_SCORE_THRESHOLD:
            output = ''

        if forced_group and forced_group != 'all':
            selected_group = forced_group
        else:
            selected_group = output

        scores.sort(key=lambda x: x['score'], reverse=True)

        active_group = selected_group or ''
        if active_group or scores:
            try:
                q = get_roi_result_queue(cam_id)
                payload = json.dumps({'group': active_group, 'scores': scores})
                _replace_with_latest(q, payload)
            except Exception:
                pass

        last_inference_outputs[cam_id] = active_group

        if selected_group is None:
            if forced_group and forced_group != 'all':
                selected_group = forced_group
            else:
                selected_group = output

        def roi_matches_group(roi_group: str | None) -> bool:
            if forced_group == 'all':
                return True
            if forced_group and forced_group != 'all':
                return roi_group == forced_group
            if selected_group:
                return roi_group == selected_group
            return False

        loop = asyncio.get_running_loop() if meets_interval else None
        scheduled_any = False
        for i, r in enumerate(rois):
            if r.get('type') != 'roi':
                continue
            if np is None:
                continue
            pts = r.get('points', [])
            if len(pts) != 4:
                continue

            should_process_roi = roi_matches_group(r.get('group'))
            if not should_process_roi:
                continue

            src = np.array([[p['x'], p['y']] for p in pts], dtype=np.float32)

            if meets_interval and r.get('module'):
                mod_name = r.get('module')
                module_entry = module_cache.get(mod_name)
                if module_entry is None:
                    module = load_custom_module(mod_name)
                    takes_source = False
                    takes_cam_id = False
                    takes_interval = False
                    if module:
                        proc = getattr(module, 'process', None)
                        if callable(proc):
                            params = inspect.signature(proc).parameters
                            takes_source = 'source' in params
                            takes_cam_id = 'cam_id' in params
                            takes_interval = 'interval' in params
                    module_cache[mod_name] = (module, takes_source, takes_cam_id, takes_interval)
                else:
                    module, takes_source, takes_cam_id, takes_interval = module_entry
                if module is None:
                    print(f"module '{mod_name}' not found for ROI {r.get('id', i)}")
                else:
                    process_fn = getattr(module, 'process', None)
                    if not callable(process_fn):
                        print(f"process function not found in module '{mod_name}' for ROI {r.get('id', i)}")
                    else:
                        width_a = np.linalg.norm(src[0] - src[1])
                        width_b = np.linalg.norm(src[2] - src[3])
                        max_w = int(max(width_a, width_b))
                        height_a = np.linalg.norm(src[0] - src[3])
                        height_b = np.linalg.norm(src[1] - src[2])
                        max_h = int(max(height_a, height_b))
                        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
                        matrix = cv2.getPerspectiveTransform(src, dst)
                        roi = await asyncio.to_thread(cv2.warpPerspective, frame, matrix, (max_w, max_h))
                        roi_identifier = r.get('id', i)
                        args = [roi, r.get('id', str(i)), save_flag]
                        if takes_source:
                            args.append(source_name)
                        if takes_cam_id:
                            args.append(cam_id)
                        if takes_interval:
                            args.append(inference_intervals.get(cam_id, 1.0))
                        fut = loop.create_future()
                        pending_key = (cam_id, frame_time)
                        start_time = time.time()
                        try:
                            await asyncio.to_thread(
                                _INFERENCE_QUEUE.put,
                                (process_fn, tuple(args), fut, loop),
                            )
                        except Exception:
                            fut.cancel()
                            queue_logger = get_logger('inference_queue')
                            if queue_logger is not None:
                                queue_logger.exception(
                                    "failed to enqueue inference task for ROI %s", r.get('id', i)
                                )
                            continue

                        pending_expected[pending_key] = (
                            pending_expected.get(pending_key, 0) + 1
                        )
                        pending_deadlines[pending_key] = (
                            time.time() + get_pending_timeout()
                        )
                        start_map = pending_roi_start.setdefault(
                            pending_key, {}
                        )
                        start_map[roi_identifier] = start_time
                        scheduled_any = True

                        def _on_done(
                            f: asyncio.Future,
                            roi_img=roi,
                            roi_id=roi_identifier,
                            frame_time=frame_time,
                            key=pending_key,
                            roi_name=r.get('name') or '',
                            roi_group=r.get('group') or r.get('page') or '',
                            module_name=mod_name or '',
                            mqtt_name=str(r.get('mqtt_config') or '').strip(),
                            source_name=source_name,
                        ) -> None:
                            if key not in pending_expected:
                                return
                            try:
                                result = f.result()
                            except asyncio.CancelledError:
                                return
                            except Exception:
                                result = None

                            if loop is None:
                                return

                            async def process_result() -> None:
                                result_text: str
                                if isinstance(result, str):
                                    result_text = result
                                elif isinstance(result, dict):
                                    if 'text' in result:
                                        result_text = str(result['text'])
                                    elif 'result' in result:
                                        result_text = str(result['result'])
                                    else:
                                        result_text = ''
                                elif result is None:
                                    result_text = ''
                                else:
                                    result_text = str(result)

                                start_map = pending_roi_start.get(key)
                                start_ts = None
                                if start_map is not None:
                                    start_ts = start_map.pop(roi_id, None)
                                    if not start_map:
                                        pending_roi_start.pop(key, None)

                                result_timestamp = time.time()
                                duration = None
                                if isinstance(start_ts, (int, float)):
                                    duration = max(0.0, result_timestamp - start_ts)

                                entry = {
                                    'id': roi_id,
                                    'image': '',
                                    'text': result_text,
                                    'frame_time': frame_time,
                                    'result_time': result_timestamp,
                                    'name': roi_name,
                                    'group': roi_group,
                                    'module': module_name,
                                }
                                entry['cam_id'] = cam_id
                                entry['source'] = source_name
                                if duration is not None:
                                    entry['duration'] = duration
                                try:
                                    if roi_img is not None:
                                        encoded, roi_buf = await asyncio.to_thread(
                                            cv2.imencode,
                                            '.jpg',
                                            roi_img,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                                        )
                                        if encoded and roi_buf is not None:
                                            image_bytes = await asyncio.to_thread(
                                                base64.b64encode,
                                                roi_buf,
                                            )
                                            entry['image'] = image_bytes.decode('ascii')
                                except Exception:
                                    pass

                                if mqtt_name:
                                    entry['mqtt_config'] = mqtt_name
                                    group_value = (
                                        str(roi_group)
                                        if roi_group is not None
                                        else ''
                                    )
                                    mqtt_payload = {
                                        'cam_id': cam_id,
                                        'source': source_name,
                                        'group': group_value,
                                        'roi_id': str(roi_id),
                                        'roi_name': roi_name,
                                        'module': module_name,
                                        'text': result_text,
                                        'frame_time': frame_time,
                                        'result_time': result_timestamp,
                                    }
                                    if duration is not None:
                                        mqtt_payload['duration'] = duration
                                    try:
                                        loop.create_task(
                                            publish_roi_to_mqtt(
                                                mqtt_name,
                                                cam_id,
                                                source_name,
                                                group_value,
                                                str(roi_id),
                                                mqtt_payload,
                                            )
                                        )
                                    except RuntimeError:
                                        logger = _get_mqtt_logger(source_name, roi_name)
                                        if logger is not None:
                                            logger.exception(
                                                "failed to schedule MQTT publish for config '%s'",
                                                mqtt_name,
                                            )

                                pending_results[key][roi_id] = entry
                                pending_deadlines[key] = (
                                    time.time() + get_pending_timeout()
                                )
                                total_rois = pending_expected.get(key, 0)
                                if (
                                    key in pending_ready
                                    and total_rois
                                    and len(pending_results[key]) >= total_rois
                                ):
                                    flush_pending_result(key)

                            try:
                                loop.create_task(process_result())
                            except RuntimeError:
                                logger = get_logger('inference_queue')
                                if logger is not None:
                                    logger.exception(
                                        "failed to schedule inference result processing for ROI %s",
                                        roi_id,
                                    )

                        fut.add_done_callback(_on_done)
            elif meets_interval and not r.get('module'):
                print(f"module missing for ROI {r.get('id', i)}")

        if scheduled_any:
            last_inference_times[cam_id] = now
            pending_key = (cam_id, frame_time)
            pending_ready.add(pending_key)
            if pending_key not in pending_context:
                if forced_group and forced_group != 'all':
                    context_group = forced_group
                elif forced_group == 'all':
                    context_group = 'all'
                else:
                    context_group = selected_group or ''
                pending_context[pending_key] = {
                    'group': context_group,
                    'source': source_name,
                }
            total_rois = pending_expected.get(pending_key, 0)
            current_results = pending_results.get(pending_key)
            if total_rois and current_results and len(current_results) >= total_rois:
                flush_pending_result(pending_key)

        for i, r in enumerate(rois):
            if np is None or r.get('type') != 'roi':
                continue
            should_show = roi_matches_group(r.get('group'))
            if not should_show:
                continue
            pts = r.get('points', [])
            if len(pts) != 4:
                continue
            src = np.array([[p['x'], p['y']] for p in pts], dtype=np.float32)
            src_int = src.astype(int)
            cv2.polylines(frame, [src_int], True, (255, 0, 0), 2)
            label_pt = src_int[0]
            cv2.putText(
                frame,
                str(r.get('id', i + 1)),
                (int(label_pt[0]), max(0, int(label_pt[1]) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), frame_time

    queue = get_frame_queue(cam_id)
    try:
        while True:
            await read_and_queue_frame(cam_id, queue, process_frame)
    except asyncio.CancelledError:
        pass
    finally:
        cleanup_pending_results(force=True)
        pending_results.clear()
        pending_expected.clear()
        pending_deadlines.clear()
        pending_ready.clear()
        pending_context.clear()
        pending_extensions.clear()
        pending_roi_start.clear()
        for mod_name, (module, _, _, _) in module_cache.items():
            if module is not None:
                with contextlib.suppress(Exception):
                    cleanup_fn = getattr(module, "cleanup", None)
                    if callable(cleanup_fn):
                        cleanup_fn()
                sys.modules.pop(f"custom_{mod_name}", None)
        module_cache.clear()
        gc.collect()


async def run_roi_loop(cam_id: str):
    queue = get_roi_frame_queue(cam_id)
    try:
        while True:
            await read_and_queue_frame(cam_id, queue)
    except asyncio.CancelledError:
        pass


# =========================
# Start/Stop helpers
# =========================
async def start_camera_task(
    cam_id: str,
    task_dict: dict[str, asyncio.Task | None],
    loop_func: Callable[[str], Awaitable[Any]],
    *,
    low_latency: bool | None = None,
):
    async with get_cam_lock(cam_id):
        task = task_dict.get(cam_id)
        if task is not None and not task.done():
            return task, {"status": "already_running", "cam_id": cam_id}, 200

        worker = camera_workers.get(cam_id)
        if (
            worker is not None
            and low_latency is not None
            and getattr(worker, "backend", "") == "ffmpeg"
            and bool(getattr(worker, "low_latency", False)) != bool(low_latency)
        ):
            await worker.stop()
            camera_workers.pop(cam_id, None)
            worker = None
        if worker is None:
            src = camera_sources.get(cam_id, 0)
            width, height = camera_resolutions.get(cam_id, (None, None))
            backend = camera_backends.get(cam_id, "opencv")
            worker_low_latency = bool(low_latency) if low_latency is not None else False
            worker = CameraWorker(
                src,
                asyncio.get_running_loop(),
                width,
                height,
                backend=backend,
                low_latency=worker_low_latency,
            )
            if not worker.start():
                await worker.stop()
                camera_workers.pop(cam_id, None)
                return (
                    task,
                    {"status": "error", "message": "open_failed", "cam_id": cam_id},
                    400,
                )
            camera_workers[cam_id] = worker
            camera_resolutions[cam_id] = (worker.width, worker.height)

        task = asyncio.create_task(loop_func(cam_id))
        task_dict[cam_id] = task
        return task, {"status": "started", "cam_id": cam_id}, 200


async def stop_camera_task(
    cam_id: str,
    task_dict: dict[str, asyncio.Task | None],
    queue_dict: dict[str, asyncio.Queue['FramePacket | None']] | None = None,
):
    async with get_cam_lock(cam_id):
        task = task_dict.pop(cam_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=0.8)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            status = "stopped"
        else:
            status = "no_task"

        if queue_dict is not None:
            queue = queue_dict.pop(cam_id, None)
            if queue is not None:
                # Drain stale frames before sending the stop sentinel so
                # websocket consumers still observe the ``None`` item.
                _drain_queue(queue)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(queue.put(None), timeout=0.2)

        inf_task = inference_tasks.get(cam_id)
        roi_task = roi_tasks.get(cam_id)
        worker = camera_workers.get(cam_id)
        if (
            (inf_task is None or inf_task.done())
            and (roi_task is None or roi_task.done())
            and worker
        ):
            await worker.stop()
            camera_workers.pop(cam_id, None)

            # NEW: fully free per-camera state & trim memory
            _free_cam_state(cam_id)

            del worker
            gc.collect()
            malloc_trim()

        if not inference_tasks:
            _stop_inference_workers()

        return task, {"status": status, "cam_id": cam_id}, 200


async def _wait_for_roi_ffmpeg_ready(cam_id: str, *, timeout: float = 2.5) -> None:
    worker = camera_workers.get(cam_id)
    if not isinstance(worker, CameraWorker):
        return
    queue = roi_frame_queues.get(cam_id)
    deadline = time.monotonic() + max(float(timeout), 0.0)
    last_seen = worker.last_frame_timestamp()
    while time.monotonic() < deadline:
        if queue is not None and queue.qsize() > 0:
            return
        if worker.has_recent_frame(freshness=0.6):
            if queue is None:
                return
            await asyncio.sleep(0.05)
            return
        current = worker.last_frame_timestamp()
        if current > 0.0 and current != last_seen:
            last_seen = current
        await asyncio.sleep(0.05)
    _CAMERA_START_LOGGER.warning(
        "cam_id=%s ffmpeg avfoundation warmup timed out after %.1fs",
        cam_id,
        timeout,
    )


# =========================
# WebSockets
# =========================
STREAM_MAX_FRAME_DELAY = float(os.getenv("STREAM_MAX_FRAME_DELAY", "1.5") or 1.5)
STREAM_SEND_TIMEOUT = float(os.getenv("STREAM_SEND_TIMEOUT", "0.6") or 0.6)
_STREAM_LOGGER = get_logger("websocket_stream")
_CAMERA_START_LOGGER = get_logger("camera_startup")


async def _stream_queue_over_websocket(
    queue: asyncio.Queue[FramePacket | bytes | str | None],
    ws: Any = websocket,
) -> None:
    """Relay items from *queue* to the active websocket connection.

    ถ้าเครือข่ายช้า เฟรมภาพจะต่อคิวจนเกิดดีเลย์สะสมเรื่อย ๆ
    จึงดึงเฉพาะเฟรมล่าสุดและทิ้งเฟรมภาพเก่าที่รอคิวอยู่ทันที
    (เฉพาะข้อมูลประเภทไบต์) ขณะที่ข้อมูลชนิดอื่นยังถูกส่งครบถ้วน
    """
    with contextlib.suppress(RuntimeError):
        await ws.accept()
    close_sent = False
    try:
        pending: deque[FramePacket | bytes | str | None] = deque()

        while True:
            if pending:
                item = pending.popleft()
            else:
                item = await queue.get()
            latest_packet: FramePacket | bytes | str | None
            if isinstance(item, FramePacket):
                latest_packet = item
            elif isinstance(item, (bytes, bytearray, memoryview)):
                payload = bytes(item) if not isinstance(item, bytes) else item
                latest_packet = FramePacket(payload=payload, created_at=time.monotonic())
            else:
                latest_packet = item

            if isinstance(latest_packet, FramePacket):
                packet = latest_packet

                while not queue.empty():
                    try:
                        maybe_next = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if maybe_next is None:
                        packet = None
                        break
                    if isinstance(maybe_next, FramePacket):
                        packet = maybe_next
                        continue
                    if isinstance(maybe_next, (bytes, bytearray, memoryview)):
                        payload = bytes(maybe_next) if not isinstance(maybe_next, bytes) else maybe_next
                        packet = FramePacket(payload=payload, created_at=time.monotonic())
                        continue
                    pending.append(maybe_next)
                if packet is None:
                    item = None
                else:
                    if STREAM_MAX_FRAME_DELAY > 0:
                        age = time.monotonic() - packet.created_at
                        if age > STREAM_MAX_FRAME_DELAY:
                            # เฟรมเก่าเกินกำหนด ทิ้งแล้วขอเฟรมใหม่
                            continue
                    item = packet.payload
            else:
                item = latest_packet

            if item is None:
                if not close_sent:
                    await ws.close(code=1000)
                    close_sent = True
                break
            try:
                if STREAM_SEND_TIMEOUT > 0:
                    await asyncio.wait_for(ws.send(item), timeout=STREAM_SEND_TIMEOUT)
                else:
                    await ws.send(item)
            except asyncio.TimeoutError:
                _STREAM_LOGGER.warning(
                    "websocket send timeout (%.2fs); closing stream", STREAM_SEND_TIMEOUT
                )
                if not close_sent:
                    with contextlib.suppress(Exception):
                        await ws.close(code=1011, reason="send_timeout")
                        close_sent = True
                break
    except ConnectionClosed:
        pass
    finally:
        if not close_sent:
            with contextlib.suppress(Exception):
                await ws.close()


@app.websocket('/ws/<string:cam_id>')
async def ws(cam_id: str):
    await _stream_queue_over_websocket(get_frame_queue(cam_id), websocket)


@app.websocket('/ws_roi/<string:cam_id>')
async def ws_roi(cam_id: str):
    await _stream_queue_over_websocket(get_roi_frame_queue(cam_id), websocket)


@app.websocket('/ws_roi_result/<string:cam_id>')
async def ws_roi_result(cam_id: str):
    await _stream_queue_over_websocket(get_roi_result_queue(cam_id), websocket)


# =========================
# Camera config
# =========================
async def apply_camera_settings(cam_id: str, data: dict) -> bool:
    async with get_cam_lock(cam_id):
        name_val = data.get("name")
        if isinstance(name_val, str) and name_val:
            active_sources[cam_id] = name_val
        elif cam_id not in active_sources:
            active_sources[cam_id] = ""
        source_val = data.get("source", "")
        width_val = data.get("width")
        height_val = data.get("height")
        backend = data.get("stream_type", "opencv")
        try:
            w = int(width_val) if width_val not in (None, "") else None
        except ValueError:
            w = None
        try:
            h = int(height_val) if height_val not in (None, "") else None
        except ValueError:
            h = None
        camera_resolutions[cam_id] = (w, h)
        camera_backends[cam_id] = backend

        worker = camera_workers.pop(cam_id, None)
        if worker:
            await worker.stop()

        try:
            camera_sources[cam_id] = int(source_val)
        except ValueError:
            camera_sources[cam_id] = source_val

        if (
            (inference_tasks.get(cam_id) and not inference_tasks[cam_id].done())
            or (roi_tasks.get(cam_id) and not roi_tasks[cam_id].done())
        ):
            width, height = camera_resolutions.get(cam_id, (None, None))
            roi_task = roi_tasks.get(cam_id)
            roi_running = roi_task is not None and not roi_task.done()
            worker_low_latency = bool(roi_running)
            if backend == "ffmpeg":
                src_for_latency = camera_sources.get(cam_id)
                src_str = str(src_for_latency)
                if src_str.startswith("avfoundation:"):
                    worker_low_latency = False
            worker = CameraWorker(
                src=camera_sources[cam_id],
                loop=asyncio.get_running_loop(),
                width=width,
                height=height,
                read_interval=0.0,       # ถ้าอยากเว้นจังหวะอ่านใส่เพิ่มได้
                backend=backend,         # ใช้ "ffmpeg" เพื่อโหมด robust
                robust=True,             # เปิด fallback/วอทช์ด็อก
                low_latency=worker_low_latency,
                loglevel="error",        # ลดสแปม log swscale
            )

            if not worker.start():
                save_service_state()
                return False
            camera_workers[cam_id] = worker

        save_service_state()
        return True


# =========================
# Source / Module listing
# =========================
@app.route("/data_sources")
async def list_sources():
    base_dir = Path(ALLOWED_ROI_DIR)
    try:
        names = [d.name for d in base_dir.iterdir() if d.is_dir()]
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.route("/inference_modules")
async def list_inference_modules():
    base_dir = Path(__file__).resolve().parent / "inference_modules"
    try:
        names = [
            d.name
            for d in base_dir.iterdir()
            if d.is_dir() and not d.name.startswith("__")
        ]
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.route("/groups")
async def list_groups():
    base_dir = Path(ALLOWED_ROI_DIR)
    groups: set[str] = set()
    try:
        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            roi_path = d / "rois.json"
            if not roi_path.exists():
                continue
            try:
                data = json.loads(roi_path.read_text())
                for r in data:
                    p = r.get("group") or r.get("page") or r.get("name")
                    if p:
                        groups.add(str(p))
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return jsonify(sorted(groups))


@app.route("/create_mqtt")
async def create_mqtt():
    return await render_template("create_mqtt.html")


@app.route("/mqtt_configs", methods=["GET", "POST"])
async def handle_mqtt_configs():
    if request.method == "GET":
        async with get_mqtt_config_lock():
            configs = [
                _public_mqtt_config(name, cfg)
                for name, cfg in sorted(mqtt_configs.items())
            ]
        return jsonify(configs)

    data = await request.get_json() or {}
    name = str(data.get("name") or "").strip()
    host = str(data.get("host") or "").strip()
    if not name or not host:
        return jsonify({"status": "error", "message": "missing data"}), 400

    port_val = data.get("port", 1883)
    try:
        port = int(port_val)
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid port"}), 400
    if not (1 <= port <= 65535):
        return jsonify({"status": "error", "message": "invalid port"}), 400

    qos_val = data.get("qos", 0)
    try:
        qos = int(qos_val)
    except (TypeError, ValueError):
        qos = 0
    if qos not in (0, 1, 2):
        return jsonify({"status": "error", "message": "invalid qos"}), 400

    keepalive_val = data.get("keepalive", 60)
    try:
        keepalive = int(keepalive_val)
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid keepalive"}), 400
    if keepalive <= 0:
        return jsonify({"status": "error", "message": "invalid keepalive"}), 400

    publish_timeout_val = data.get("publish_timeout", 5.0)
    try:
        publish_timeout = max(0.1, float(publish_timeout_val))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid publish_timeout"}), 400

    username = str(data.get("username") or "").strip()
    password_val = data.get("password")
    password = str(password_val) if password_val is not None else ""
    base_topic = str(data.get("base_topic") or "").strip()
    client_id = str(data.get("client_id") or "").strip()
    retain = _truthy(data.get("retain"))
    tls_enabled = _truthy(data.get("tls"))

    cfg: dict[str, Any] = {
        "name": name,
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "base_topic": base_topic,
        "client_id": client_id,
        "qos": qos,
        "retain": retain,
        "keepalive": keepalive,
        "publish_timeout": publish_timeout,
    }
    if tls_enabled:
        cfg["tls"] = True
    else:
        cfg["tls"] = False

    async with get_mqtt_config_lock():
        if name in mqtt_configs:
            return jsonify({"status": "error", "message": "name exists"}), 400
        mqtt_configs[name] = cfg
        try:
            await asyncio.to_thread(save_mqtt_configs_to_disk, mqtt_configs)
        except Exception:
            mqtt_configs.pop(name, None)
            return jsonify({"status": "error", "message": "save failed"}), 500

    _missing_mqtt_configs_notified.discard(name)
    return (
        jsonify({
            "status": "created",
            "config": _public_mqtt_config(name, cfg),
        }),
        201,
    )


@app.route("/mqtt_configs/<string:name>", methods=["DELETE"])
async def delete_mqtt_config(name: str):
    name = str(name or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "invalid name"}), 400

    async with get_mqtt_config_lock():
        if name not in mqtt_configs:
            return jsonify({"status": "error", "message": "not found"}), 404
        mqtt_configs.pop(name, None)
        try:
            await asyncio.to_thread(save_mqtt_configs_to_disk, mqtt_configs)
        except Exception:
            return jsonify({"status": "error", "message": "save failed"}), 500

    _missing_mqtt_configs_notified.discard(name)
    return jsonify({"status": "deleted"})


@app.route("/source_list", methods=["GET"])
async def source_list():
    base_dir = Path(ALLOWED_ROI_DIR)
    result = []
    try:
        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            cfg_path = d / "config.json"
            if not cfg_path.exists():
                continue
            try:
                with cfg_path.open("r") as f:
                    cfg = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            result.append(
                {
                    "name": cfg.get("name", d.name),
                    "source": cfg.get("source", ""),
                    "width": cfg.get("width"),
                    "height": cfg.get("height"),
                    "stream_type": cfg.get("stream_type", "opencv"),
                }
            )
    except FileNotFoundError:
        pass
    return jsonify(result)


@app.route("/source_config")
async def source_config():
    name = os.path.basename(request.args.get("name", "").strip())
    path = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name, "config.json"))
    if not _safe_in_base(ALLOWED_ROI_DIR, path) or not os.path.exists(path):
        return jsonify({"status": "error", "message": "not found"}), 404
    with open(path, "r") as f:
        cfg = json.load(f)
    return jsonify(cfg)


# =========================
# Source CRUD (secure)
# =========================
@app.route("/create_source", methods=["GET", "POST"])
async def create_source():
    if request.method == "GET":
        return await render_template("create_source.html")

    form = await request.form
    name = os.path.basename(form.get("name", "").strip())
    source = form.get("source", "").strip()
    width = form.get("width")
    height = form.get("height")
    stream_type = form.get("stream_type", "opencv")
    if not name or not source:
        return jsonify({"status": "error", "message": "missing data"}), 400

    source_dir = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, source_dir):
        return jsonify({"status": "error", "message": "invalid name"}), 400

    try:
        os.makedirs(source_dir, exist_ok=False)
    except FileExistsError:
        return jsonify({"status": "error", "message": "name exists"}), 400

    try:
        config = {
            "name": name,
            "source": source,
            "rois": "rois.json",
            "stream_type": stream_type,
        }
        if width:
            try:
                config["width"] = int(width)
            except ValueError:
                pass
        if height:
            try:
                config["height"] = int(height)
            except ValueError:
                pass
        rois_path = os.path.join(source_dir, "rois.json")
        with open(rois_path, "w") as f:
            f.write("[]")
        with open(os.path.join(source_dir, "config.json"), "w") as f:
            json.dump(config, f)
    except Exception:
        shutil.rmtree(source_dir, ignore_errors=True)
        return jsonify({"status": "error", "message": "save failed"}), 500

    return jsonify({"status": "created"})


@app.route("/delete_source/<name>", methods=["DELETE"])
async def delete_source(name: str):
    name = os.path.basename(name.strip())
    directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, directory) or not os.path.exists(directory):
        return jsonify({"status": "error", "message": "not found"}), 404
    shutil.rmtree(directory)
    json_path = "sources.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = [s for s in data if s != name]
            elif isinstance(data, dict):
                data.pop(name, None)
                if "sources" in data and isinstance(data["sources"], list):
                    data["sources"] = [s for s in data["sources"] if s != name]
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    return jsonify({"status": "deleted"})


@app.route("/update_stream_type/<name>", methods=["PATCH"])
async def update_stream_type(name: str):
    name = os.path.basename(name.strip())
    directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, directory):
        return jsonify({"status": "error", "message": "invalid name"}), 400
    cfg_path = os.path.join(directory, "config.json")
    if not os.path.exists(cfg_path):
        return jsonify({"status": "error", "message": "not found"}), 404
    data = await request.get_json()
    stream_type = data.get("stream_type") if data else None
    if stream_type not in ("opencv", "ffmpeg"):
        return jsonify({"status": "error", "message": "invalid stream_type"}), 400
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        cfg["stream_type"] = stream_type
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
    except Exception:
        return jsonify({"status": "error", "message": "save failed"}), 500
    return jsonify({"status": "updated"})


@app.route("/update_source/<name>", methods=["PATCH"])
async def update_source(name: str):
    name = os.path.basename(name.strip())
    directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, directory):
        return jsonify({"status": "error", "message": "invalid name"}), 400

    cfg_path = os.path.join(directory, "config.json")
    if not os.path.exists(cfg_path):
        return jsonify({"status": "error", "message": "not found"}), 404

    data = await request.get_json()
    new_source = (data or {}).get("source", "").strip()
    if not new_source:
        return jsonify({"status": "error", "message": "missing source"}), 400

    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        cfg["source"] = new_source
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
    except Exception:
        return jsonify({"status": "error", "message": "save failed"}), 500

    return jsonify({"status": "updated"})


@app.route("/rename_source/<name>", methods=["PATCH"])
async def rename_source(name: str):
    old_name = os.path.basename(name.strip())
    data = await request.get_json()
    new_name = os.path.basename((data or {}).get("name", "").strip())

    if not old_name or not new_name:
        return jsonify({"status": "error", "message": "missing name"}), 400
    if old_name == new_name:
        return jsonify({"status": "unchanged"})

    old_directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, old_name))
    new_directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, new_name))

    if not _safe_in_base(ALLOWED_ROI_DIR, old_directory) or not os.path.exists(old_directory):
        return jsonify({"status": "error", "message": "not found"}), 404
    if not _safe_in_base(ALLOWED_ROI_DIR, new_directory):
        return jsonify({"status": "error", "message": "invalid name"}), 400
    if os.path.exists(new_directory):
        return jsonify({"status": "error", "message": "name exists"}), 400

    try:
        os.rename(old_directory, new_directory)
        cfg_path = os.path.join(new_directory, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            cfg["name"] = new_name
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

        json_path = "sources.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    sources_data = json.load(f)
                if isinstance(sources_data, list):
                    sources_data = [new_name if s == old_name else s for s in sources_data]
                elif isinstance(sources_data, dict):
                    if old_name in sources_data and new_name in sources_data:
                        raise ValueError("name exists")
                    if old_name in sources_data:
                        sources_data[new_name] = sources_data.pop(old_name)
                    if (
                        "sources" in sources_data
                        and isinstance(sources_data["sources"], list)
                    ):
                        sources_data["sources"] = [
                            new_name if s == old_name else s
                            for s in sources_data["sources"]
                        ]
                with open(json_path, "w") as f:
                    json.dump(sources_data, f, indent=2)
            except ValueError:
                raise
            except Exception:
                pass
    except Exception:
        if os.path.exists(new_directory) and not os.path.exists(old_directory):
            try:
                os.rename(new_directory, old_directory)
            except Exception:
                pass
        return jsonify({"status": "error", "message": "rename failed"}), 500

    return jsonify({"status": "renamed", "name": new_name})


# =========================
# ROI save/load (secure)
# =========================
@app.route("/save_roi", methods=["POST"])
async def save_roi():
    data = await request.get_json()
    rois = ensure_roi_ids(data.get("rois", []))
    path = request.args.get("path", "")
    base_dir = ALLOWED_ROI_DIR

    if path:
        full_path = os.path.realpath(os.path.join(base_dir, path))
        if not _safe_in_base(base_dir, full_path):
            return jsonify({"status": "error", "message": "path outside allowed directory"}), 400
        dir_path = os.path.dirname(full_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(rois, f, indent=2)
        return jsonify({"status": "saved", "filename": full_path})

    name = os.path.basename(data.get("source", "").strip())
    if not name:
        return jsonify({"status": "error", "message": "missing source"}), 400
    directory = os.path.realpath(os.path.join(base_dir, name))
    if not _safe_in_base(base_dir, directory):
        return jsonify({"status": "error", "message": "invalid source path"}), 400

    config_path = os.path.join(directory, "config.json")
    if not os.path.exists(config_path):
        return jsonify({"status": "error", "message": "config not found"}), 404
    with open(config_path, "r") as f:
        cfg = json.load(f)
    roi_file = cfg.get("rois", "rois.json")
    roi_path = os.path.realpath(os.path.join(directory, roi_file))
    if not _safe_in_base(directory, roi_path):
        return jsonify({"status": "error", "message": "invalid ROI filename"}), 400
    with open(roi_path, "w") as f:
        json.dump(rois, f, indent=2)
    return jsonify({"status": "saved", "filename": roi_path})


@app.route("/load_roi/<name>")
async def load_roi(name: str):
    name = os.path.basename(name.strip())
    directory = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, name))
    if not _safe_in_base(ALLOWED_ROI_DIR, directory):
        return jsonify({"rois": [], "filename": "None"})
    config_path = os.path.join(directory, "config.json")
    if not os.path.exists(config_path):
        return jsonify({"rois": [], "filename": "None"})
    with open(config_path, "r") as f:
        cfg = json.load(f)
    roi_file = cfg.get("rois", "rois.json")
    roi_path = os.path.realpath(os.path.join(directory, roi_file))
    if not _safe_in_base(directory, roi_path) or not os.path.exists(roi_path):
        return jsonify({"rois": [], "filename": roi_file})
    with open(roi_path, "r") as f:
        rois = ensure_roi_ids(json.load(f))
    return jsonify({"rois": rois, "filename": roi_file})


@app.route("/load_roi_file")
async def load_roi_file():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"rois": [], "filename": "None"})
    full_path = os.path.realpath(path)
    if not _safe_in_base(ALLOWED_ROI_DIR, full_path) or not os.path.exists(full_path):
        return jsonify({"rois": [], "filename": "None"})
    with open(full_path, "r") as f:
        rois = ensure_roi_ids(json.load(f))
    return jsonify({"rois": rois, "filename": os.path.basename(full_path)})


@app.route("/read_log")
async def read_log():
    source = request.args.get("source", "")
    try:
        lines = int(request.args.get("lines", 40))
    except ValueError:
        lines = 40
    if not source:
        return jsonify({"lines": []})
    source = os.path.basename(source.strip())
    log_path = os.path.realpath(os.path.join(ALLOWED_ROI_DIR, source, "custom.log"))
    if not _safe_in_base(ALLOWED_ROI_DIR, log_path) or not os.path.exists(log_path):
        return jsonify({"lines": []})
    try:
        with open(log_path, "r") as f:
            content = f.readlines()[-lines:]
    except Exception:
        content = []
    return jsonify({"lines": [line.rstrip() for line in content]})


# =========================
# Inference controls
# =========================
async def perform_start_inference(cam_id: str, rois=None, group: str | None = None, save_state: bool = True):
    """Start an inference task for the given camera.

    Returns a tuple ``(ok, resp, status)`` where ``ok`` is a boolean indicating
    success, ``resp`` is the response dictionary from ``start_camera_task`` and
    ``status`` is the HTTP status code. This mirrors the behaviour of
    ``start_roi_stream`` which bubbles up the camera start error message (e.g.
    ``"open_failed"``) to the caller so the frontend can display a more useful
    alert.
    """

    if roi_tasks.get(cam_id) and not roi_tasks[cam_id].done():
        return False, {"status": "roi_running", "cam_id": cam_id}, 400
    if rois is None:
        source = active_sources.get(cam_id, "")
        source_dir = os.path.join(ALLOWED_ROI_DIR, source)
        rois_path = os.path.join(source_dir, "rois.json")
        try:
            with open(rois_path) as f:
                rois = json.load(f)
        except FileNotFoundError:
            rois = []
    elif isinstance(rois, str):
        try:
            rois = json.loads(rois)
        except json.JSONDecodeError:
            rois = []
    if not isinstance(rois, list):
        rois = []
    rois = ensure_roi_ids(rois)

    if np is not None:
        for r in rois:
            if isinstance(r, dict) and r.get("type") == "page":
                img_b64 = r.get("image")
                if img_b64:
                    try:
                        arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
                        tmpl = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        tmpl = None
                    r["_template"] = tmpl

    inference_rois[cam_id] = rois
    if group is None:
        inference_groups.pop(cam_id, None)
    else:
        inference_groups[cam_id] = group

    q = get_roi_result_queue(cam_id)
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break

    _task, resp, status = await start_camera_task(cam_id, inference_tasks, run_inference_loop)
    if status != 200:
        inference_rois.pop(cam_id, None)
        inference_groups.pop(cam_id, None)
        return False, resp, status
    if save_state:
        save_service_state()
    return True, resp, status


async def perform_stop_inference(cam_id: str, save_state: bool = True):
    _, resp, status = await stop_camera_task(cam_id, inference_tasks, frame_queues)
    queue = roi_result_queues.get(cam_id)
    if queue is not None:
        await queue.put(None)
        roi_result_queues.pop(cam_id, None)
    rois = inference_rois.pop(cam_id, [])
    seen: set[str] = set()
    for r in rois:
        mod_name = r.get("module")
        if not mod_name or mod_name in seen:
            continue
        seen.add(mod_name)
        mod_key = f"custom_{mod_name}"
        module = sys.modules.get(mod_key)
        if module is not None:
            with contextlib.suppress(Exception):
                cleanup_fn = getattr(module, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn()
            sys.modules.pop(mod_key, None)
    inference_groups.pop(cam_id, None)
    save_roi_flags.pop(cam_id, None)
    inference_intervals.pop(cam_id, None)
    last_inference_times.pop(cam_id, None)
    inference_result_timeouts.pop(cam_id, None)
    inference_draw_page_boxes.pop(cam_id, None)
    _free_cam_state(cam_id)
    gc.collect()
    malloc_trim()
    if save_state:
        save_service_state()
    return resp, status


@app.route('/start_inference/<string:cam_id>', methods=["POST"])
async def start_inference(cam_id: str):
    if roi_tasks.get(cam_id) and not roi_tasks[cam_id].done():
        return jsonify({"status": "roi_running", "cam_id": cam_id}), 400
    data = await request.get_json() or {}
    cfg = dict(data)
    rois = cfg.pop("rois", None)
    group = cfg.pop("group", None)
    interval = cfg.pop("interval", None)
    draw_page_boxes = cfg.pop("draw_page_boxes", None)
    timeout_sentinel = object()
    result_timeout = cfg.pop("result_timeout", timeout_sentinel)
    if draw_page_boxes is None:
        inference_draw_page_boxes.pop(cam_id, None)
    else:
        inference_draw_page_boxes[cam_id] = bool(draw_page_boxes)
    if interval is not None:
        try:
            inference_intervals[cam_id] = float(interval)
        except (TypeError, ValueError):
            inference_intervals[cam_id] = 1.0
    else:
        inference_intervals.pop(cam_id, None)
    if result_timeout is not timeout_sentinel:
        if result_timeout is None:
            inference_result_timeouts.pop(cam_id, None)
        else:
            try:
                inference_result_timeouts[cam_id] = max(0.1, float(result_timeout))
            except (TypeError, ValueError):
                inference_result_timeouts.pop(cam_id, None)
    if cfg:
        cfg_ok = await apply_camera_settings(cam_id, cfg)
        if not cfg_ok:
            return jsonify({"status": "error", "cam_id": cam_id}), 400
    ok, resp, status = await perform_start_inference(cam_id, rois, group)
    return jsonify(resp), status


@app.route('/stop_inference/<string:cam_id>', methods=["POST"])
async def stop_inference(cam_id: str):
    resp, status = await perform_stop_inference(cam_id)
    return jsonify(resp), status


@app.route("/dashboard")
async def dashboard():
    return redirect("/home")


@app.route("/api/dashboard", methods=["GET"])
async def api_dashboard():
    payload = await build_dashboard_response()
    return jsonify(payload)


@app.route('/start_roi_stream/<string:cam_id>', methods=["POST"])
async def start_roi_stream(cam_id: str):
    if inference_tasks.get(cam_id) and not inference_tasks[cam_id].done():
        return jsonify({"status": "inference_running", "cam_id": cam_id}), 400
    data = await request.get_json() or {}
    if data:
        cfg_ok = await apply_camera_settings(cam_id, data)
        if not cfg_ok:
            return jsonify({"status": "error", "cam_id": cam_id}), 400
    _drain_queue(get_roi_frame_queue(cam_id))
    desired_low_latency = True
    backend = camera_backends.get(cam_id)
    src_val = camera_sources.get(cam_id)
    src_str = str(src_val)
    if backend == "ffmpeg" and src_str.startswith("avfoundation:"):
        desired_low_latency = False

    _, resp, status = await start_camera_task(
        cam_id,
        roi_tasks,
        run_roi_loop,
        low_latency=desired_low_latency,
    )
    if (
        status == 200
        and isinstance(resp, dict)
        and resp.get("status") == "started"
        and backend == "ffmpeg"
        and src_str.startswith("avfoundation:")
    ):
        await _wait_for_roi_ffmpeg_ready(cam_id)
    return jsonify(resp), status


@app.route('/stop_roi_stream/<string:cam_id>', methods=["POST"])
async def stop_roi_stream(cam_id: str):
    _, resp, status = await stop_camera_task(cam_id, roi_tasks, roi_frame_queues)

    # NEW: free state & trim for ROI-only mode too
    _free_cam_state(cam_id)
    gc.collect()
    malloc_trim()

    return jsonify(resp), status


@app.route('/inference_status/<string:cam_id>', methods=["GET"])
async def inference_status(cam_id: str):
    running = inference_tasks.get(cam_id) is not None and not inference_tasks[cam_id].done()
    source = active_sources.get(cam_id, "")
    group = inference_groups.get(cam_id)
    if not isinstance(group, str):
        group = ""
    return jsonify({
        "running": running,
        "cam_id": cam_id,
        "source": source,
        "group": group,
    })


@app.route('/roi_stream_status/<string:cam_id>', methods=["GET"])
async def roi_stream_status(cam_id: str):
    running = roi_tasks.get(cam_id) is not None and not roi_tasks[cam_id].done()
    source = active_sources.get(cam_id, "")
    return jsonify({"running": running, "cam_id": cam_id, "source": source})


# =========================
# Snapshot (safer Response)
# =========================
@app.route("/ws_snapshot/<string:cam_id>")
async def ws_snapshot(cam_id: str):
    worker = camera_workers.get(cam_id)
    if worker is None:
        return "Camera not initialized", 400
    frame = await worker.read()
    if frame is None:
        return "Camera error", 500
    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return "Encode error", 500
    return Response(buffer.tobytes(), mimetype="image/jpeg")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VisionROI server")
    parser.add_argument("--port", type=int, default=5000, help="Port for the web server")
    parser.add_argument("--use-uvicorn", action="store_true", help="Run with uvicorn instead of built-in server")
    args = parser.parse_args()

    if args.use_uvicorn:
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
            lifespan="on",
            timeout_keep_alive=2,
            timeout_graceful_shutdown=2,
            workers=1,
        )
    else:
        app.run(port=args.port)
