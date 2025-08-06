import time
from src.packages.models.yolov8.yolov8onnx.yolov8object.YOLOv8 import YOLOv8
from typhoon_ocr import ocr_document
from PIL import Image
import cv2
import base64
from logging.handlers import TimedRotatingFileHandler
import logging
import os
from datetime import datetime
import threading
try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_path = os.path.join(os.path.dirname(__file__), "custom.log")
_handler = TimedRotatingFileHandler(_log_path, when="D", interval=1, backupCount=7)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)


# โหลดโมเดล (ถ้ามี)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

# ตัวแปรควบคุมเวลาเรียก OCR แยกตาม roi พร้อมตัวล็อกป้องกันการเข้าถึงพร้อมกัน
last_ocr_times = {}
_last_ocr_lock = threading.Lock()


def _run_ocr(base64_string, roi_id=None):
    """เรียก OCR ในเธรดพื้นหลังและบันทึกผล"""
    try:
        markdown = ocr_document(base64_string)
        if roi_id is not None:
            logger.info(f"roi_id={roi_id} OCR result: {markdown}")
        else:
            logger.info(f"OCR result: {markdown}")
    except Exception as e:
        if roi_id is not None:
            logger.exception(f"roi_id={roi_id} OCR error: {e}")
        else:
            logger.exception(f"OCR error: {e}")

def _save_image_async(path, image):
    """บันทึกรูปภาพแบบแยกเธรด"""
    cv2.imwrite(path, image)


def process(frame, roi_id=None, save=False):
    """ประมวลผล ROI และเรียก OCR เมื่อเวลาห่างจากครั้งก่อน >= 2 วินาที
    บันทึกรูปภาพแบบไม่บล็อกเมื่อระบุให้บันทึก"""

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

    logger.info(f"roi_id={roi_id} diff_time={diff_time}")

    if should_ocr:
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_string = base64.b64encode(buffer).decode('utf-8')
            threading.Thread(
                target=_run_ocr, args=(base64_string, roi_id), daemon=True
            ).start()
        except Exception as e:
            logger.exception(f"roi_id={roi_id} OCR error: {e}")
    else:
        logger.info(f"OCR skipped for ROI {roi_id} (throttled)")

    if save:
        save_dir = os.path.join(os.path.dirname(__file__), "images", str(roi_id))
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        path = os.path.join(save_dir, filename)
        threading.Thread(
            target=_save_image_async, args=(path, frame.copy()), daemon=True
        ).start()

    return frame
