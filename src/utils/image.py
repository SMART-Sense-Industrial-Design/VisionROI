import cv2
import threading
import logging

_imwrite_lock = threading.Lock()


def save_image_async(path: str, image) -> None:
    """บันทึกรูปภาพแบบไม่บล็อกพร้อมล็อกการเรียก cv2.imwrite"""

    def _save():
        try:
            with _imwrite_lock:
                cv2.imwrite(path, image)
        except Exception as e:  # pragma: no cover
            logging.getLogger(__name__).exception(
                f"Failed to save image {path}: {e}"
            )

    threading.Thread(target=_save, daemon=True).start()
