from __future__ import annotations

import cv2
import threading
import logging
from typing import Optional

_imwrite_lock = threading.Lock()


def save_image_async(path: str, image, logger: Optional[logging.Logger] = None) -> None:
    """บันทึกรูปภาพแบบไม่บล็อกพร้อมล็อกการเรียก cv2.imwrite"""

    log = logger or logging.getLogger(__name__)

    def _save():
        try:
            with _imwrite_lock:
                cv2.imwrite(path, image)
        except Exception as e:  # pragma: no cover
            log.exception(f"Failed to save image {path}: {e}")

    threading.Thread(target=_save, daemon=True).start()
