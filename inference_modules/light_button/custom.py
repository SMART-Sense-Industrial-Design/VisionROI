from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

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
    cv2_lib = _ensure_cv2()
    base_dir = (
        _DATA_SOURCES_ROOT / source if source else Path(__file__).resolve().parent
    )
    roi_folder = f"{roi_id}" if roi_id is not None else "roi"
    save_dir = base_dir / "images" / roi_folder
    os.makedirs(save_dir, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
    path = save_dir / filename
    try:
        cv2_lib.imwrite(str(path), frame)
    except Exception:  # pragma: no cover - logging best effort
        logger.exception("failed to save light_button frame to %s", path)


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

        p10 = _percentile(gray, 10.0)
        p25 = _percentile(gray, 25.0)
        p50 = _percentile(gray, 50.0)
        p75 = _percentile(gray, 75.0)
        p90 = _percentile(gray, 90.0)
        p95 = _percentile(gray, 95.0)
        p99 = _percentile(gray, 99.0)

        total_pixels = float(gray.shape[0] * gray.shape[1]) if hasattr(gray, "shape") else 0.0

        def ratio_above(lower: int) -> float:
            if total_pixels <= 0:
                return 0.0
            mask = cv2_lib.inRange(gray, lower, 255)
            return float(cv2_lib.countNonZero(mask)) / total_pixels

        glow_ratio = ratio_above(170)
        bright_ratio = ratio_above(200)
        hot_ratio = ratio_above(230)
        spark_ratio = ratio_above(245)

        background_level = (p10 + p25) / 2.0
        upper_level = (p90 + p95) / 2.0
        contrast = upper_level - background_level
        peak_contrast = max_value - p75

        dynamic_score = (
            max(0.0, mean_brightness - background_level) * 0.8
            + max(0.0, contrast) * 1.5
            + hot_ratio * 450.0
            + bright_ratio * 220.0
            + glow_ratio * 120.0
            + spark_ratio * 600.0
            + max(0.0, peak_contrast) * 0.6
            + max(0.0, p99 - p75) * 0.9
        )

        adaptive_threshold = threshold
        if adaptive_threshold is None:
            adaptive_threshold = 150.0
            on_conditions = [
                dynamic_score >= adaptive_threshold,
                (
                    peak_contrast >= 35.0
                    and hot_ratio >= 0.0035
                    and (spark_ratio >= 0.0005 or bright_ratio >= 0.01)
                ),
                (
                    contrast >= 45.0
                    and glow_ratio >= 0.06
                    and mean_brightness >= background_level + 28.0
                ),
                (
                    bright_ratio >= 0.04
                    and contrast >= 40.0
                    and max_value >= 215.0
                ),
                (
                    std_dev >= 32.0
                    and hot_ratio >= 0.004
                    and mean_brightness >= 125.0
                ),
            ]
            is_on = 1 if any(on_conditions) else 0
        else:
            adaptive_threshold = float(adaptive_threshold)
            is_on = 1 if mean_brightness >= adaptive_threshold else 0

        state_text = "on" if is_on else "off"

        logger.info(
            (
                "roi_id=%s light_button result: %s "
                "(brightness=%.2f, std=%.2f, score=%.2f, glow=%.4f, bright=%.4f, "
                "hot=%.4f, threshold=%.2f)"
            ),
            roi_id,
            state_text,
            mean_brightness,
            std_dev,
            dynamic_score,
            glow_ratio,
            bright_ratio,
            hot_ratio,
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
