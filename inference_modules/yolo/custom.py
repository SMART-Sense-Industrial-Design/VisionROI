import time
from src.packages.models.yolov8.yolov8onnx.yolov8object.YOLOv8 import YOLOv8
from typhoon_ocr import ocr_document
from PIL import Image
import cv2
import base64
import logging
import os
from datetime import datetime
import threading
from pathlib import Path
from src.utils.logger import get_logger
try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None


MODULE_NAME = "yolo"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"


# โหลดโมเดล (ถ้ามี)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

# ตัวแปรควบคุมเวลาเรียก OCR แยกตาม roi พร้อมตัวล็อกป้องกันการเข้าถึงพร้อมกัน
last_ocr_times = {}
_last_ocr_lock = threading.Lock()
_imwrite_lock = threading.Lock()



def _save_image_async(path, image):
    """บันทึกรูปภาพแบบแยกเธรด"""
    try:
        with _imwrite_lock:
            cv2.imwrite(path, image)
    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to save image {path}: {e}")


def process(
    frame,
    roi_id=None,
    save=False,
    source="",
    cam_id: int | None = None,
):
    """ประมวลผล ROI และเรียก OCR เมื่อเวลาห่างจากครั้งก่อน >= 2 วินาที
    บันทึกรูปภาพแบบไม่บล็อกเมื่อระบุให้บันทึก"""
    logger = get_logger(MODULE_NAME, source)

    if cam_id is not None:
        try:
            import app  # type: ignore
            app.save_roi_flags[cam_id] = True
        except Exception:  # pragma: no cover
            pass

    if isinstance(frame, Image.Image) and np is not None:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    current_time = time.monotonic()

    with _last_ocr_lock:
        last_time = last_ocr_times.get(roi_id)
        diff_time = 0 if last_time is None else current_time - last_time
        if last_time is None or diff_time >= 2:
            last_ocr_times[roi_id] = current_time
            should_ocr = True
        else:
            should_ocr = False

    if should_ocr:

        if save:
            base_dir = (
                _data_sources_root / source
                if source
                else Path(__file__).resolve().parent
            )
            roi_folder = f"{roi_id}" if roi_id is not None else "roi"
            save_dir = base_dir / "images" / roi_folder
            os.makedirs(save_dir, exist_ok=True)
            filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
            path = save_dir / filename
            threading.Thread(
                target=_save_image_async, args=(str(path), frame.copy()), daemon=True
            ).start()

    # else:
    #     logger.info(f"OCR skipped for ROI {roi_id} (throttled)")


    return frame
