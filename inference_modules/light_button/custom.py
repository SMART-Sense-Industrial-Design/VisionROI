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
        mean_brightness = float(gray.mean())
        adaptive_threshold = threshold
        if adaptive_threshold is None:
            # ใช้ค่ากลางจาก histogram แบบง่ายเพื่อรองรับแสงที่เปลี่ยนไป
            adaptive_threshold = max(30.0, min(220.0, float(gray.max()) * 0.5))

        is_on = 1 if mean_brightness >= adaptive_threshold else 0
        state_text = "on" if is_on else "off"

        logger.info(
            "roi_id=%s light_button result: %s (brightness=%.2f, threshold=%.2f)",
            roi_id,
            state_text,
            mean_brightness,
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
