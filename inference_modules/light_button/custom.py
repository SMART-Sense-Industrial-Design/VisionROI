from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger
from src.utils.image import save_image_async

try:  # pragma: no cover - optional dependencies
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependencies
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependencies
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

MODULE_NAME = "light_button"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)
_DATA_SOURCES_ROOT = Path(__file__).resolve().parents[2] / "data_sources"


def _percentile(gray: Any, percentile: float) -> float:
    """Return percentile value without relying on NumPy being available."""

    if np is not None:
        return float(np.percentile(gray, percentile))

    cv2_lib = _ensure_cv2()
    hist = cv2_lib.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = float(gray.shape[0] * gray.shape[1]) if hasattr(gray, "shape") else 0.0
    if total_pixels <= 0:
        return 0.0

    target = total_pixels * (percentile / 100.0)
    cumulative = 0.0
    for value in range(256):
        cumulative += float(hist[value][0])
        if cumulative >= target:
            return float(value)
    return 255.0


def _ensure_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for the light_button module")
    return cv2


def _prepare_frame(frame: Any) -> Any:
    cv2_lib = _ensure_cv2()
    if Image is not None and isinstance(frame, Image.Image):
        if np is None:
            raise RuntimeError(
                "NumPy is required to convert PIL images inside light_button module"
            )
        frame = cv2_lib.cvtColor(np.array(frame), cv2_lib.COLOR_RGB2BGR)
    return frame


def _save_image(frame: Any, roi_id: Any, source: str) -> None:
    _ensure_cv2()
    base_dir = (
        _DATA_SOURCES_ROOT / source if source else Path(__file__).resolve().parent
    )
    roi_folder = f"{roi_id}" if roi_id is not None else "roi"
    save_dir = base_dir / "images" / roi_folder
    os.makedirs(save_dir, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
    path = save_dir / filename
    save_image_async(str(path), frame, logger=logger)


def process(
    frame: Any,
    roi_id: Any = None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """ตรวจสอบสถานะไฟของปุ่มจากค่าเฉลี่ยความสว่างของภาพ."""

    logger = get_logger(MODULE_NAME, source)
    try:
        frame = _prepare_frame(frame)
        cv2_lib = _ensure_cv2()

        gray = cv2_lib.cvtColor(frame, cv2_lib.COLOR_BGR2GRAY)
        mean_brightness = float(cv2_lib.mean(gray)[0])
        std_dev = float(cv2_lib.meanStdDev(gray)[1][0][0])
        _, max_value, _, _ = cv2_lib.minMaxLoc(gray)
        max_value = float(max_value)
        p75 = _percentile(gray, 75.0)
        p90 = _percentile(gray, 90.0)

        highlight_cutoff = max(p90, max_value - 15.0, 180.0)
        highlight_cutoff = min(255.0, max(0.0, highlight_cutoff))
        bright_mask = cv2_lib.inRange(gray, int(round(highlight_cutoff)), 255)
        bright_pixels = float(cv2_lib.countNonZero(bright_mask))
        total_pixels = float(gray.shape[0] * gray.shape[1]) if hasattr(gray, "shape") else 0.0
        bright_ratio = bright_pixels / total_pixels if total_pixels > 0 else 0.0

        highlight_gain = 0.0
        if bright_ratio > 0:
            highlight_gain = min(
                60.0,
                max(0.0, max_value - p75) * bright_ratio * 300.0,
            )

        combined_score = mean_brightness + (0.8 * std_dev) + highlight_gain

        adaptive_threshold = threshold
        if adaptive_threshold is None:
            baseline_threshold = (p75 + p90) / 2.0
            adaptive_threshold = max(70.0, min(200.0, baseline_threshold))
            score_condition = combined_score >= adaptive_threshold
            highlight_condition = (
                bright_ratio >= 0.012
                or (bright_ratio >= 0.002 and max_value >= 220.0)
                or (
                    bright_ratio >= 0.004
                    and (max_value - p75) >= 25.0
                )
            )
            contrast_condition = (
                std_dev >= 30.0
                and (max_value - p75) >= 20.0
                and bright_ratio >= 0.0015
            )
            is_on = 1 if (score_condition or highlight_condition or contrast_condition) else 0
        else:
            adaptive_threshold = float(adaptive_threshold)
            is_on = 1 if mean_brightness >= adaptive_threshold else 0

        state_text = "on" if is_on else "off"

        logger.info(
            "roi_id=%s light_button result: %s (brightness=%.2f, score=%.2f, ratio=%.4f, threshold=%.2f)",
            roi_id,
            state_text,
            mean_brightness,
            combined_score,
            bright_ratio,
            adaptive_threshold,
        )

        if save:
            _save_image(frame, roi_id, source)

        return {"text": state_text, "value": is_on, "brightness": mean_brightness}
    except Exception as exc:  # pragma: no cover - log unexpected errors
        logger.exception("light_button processing error for roi_id=%s: %s", roi_id, exc)
        return {"text": "error", "value": None}


def cleanup() -> None:
    """ไม่มีสถานะถาวรให้เคลียร์ แต่คงฟังก์ชันไว้เพื่อความสอดคล้อง"""
    # ฟังก์ชันนี้มีไว้เพื่อให้สอดคล้องกับอินเทอร์เฟซของโมดูลอื่น ๆ
    # ไม่มีทรัพยากรถาวรที่ต้องจัดการ
    return None
