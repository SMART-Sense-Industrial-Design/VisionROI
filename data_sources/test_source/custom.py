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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_path = os.path.join(os.path.dirname(__file__), "custom.log")
_handler = TimedRotatingFileHandler(_log_path, when="D", interval=1, backupCount=7)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)


# โหลดโมเดล (ถ้ามี)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

# ตัวแปรควบคุมเวลาเรียก OCR แยกตาม roi
last_ocr_times = {}

def process(frame, roi_id=None, save=False):
    """ประมวลผล ROI และเรียก OCR เมื่อเวลาห่างจากครั้งก่อน >= 2 วินาที
    บันทึกรูปภาพเมื่อระบุให้บันทึก"""
    if save:
        save_dir = os.path.join(os.path.dirname(__file__), "images", "roi1")
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"

        cv2.imwrite(os.path.join(save_dir, filename), frame)

    current_time = time.monotonic()
    last_time = last_ocr_times.get(roi_id)

    # คำนวณเวลาที่ห่างจากการเรียกครั้งก่อน (เป็นวินาที)
    diff_time = 0 if last_time is None else current_time - last_time
    logger.info(f"roi_id={roi_id} diff_time={diff_time}")

    if last_time is None or diff_time >= 2:
        last_ocr_times[roi_id] = current_time
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_string = base64.b64encode(buffer).decode('utf-8')
            markdown = ocr_document(base64_string)
            logger.info(f"roi_id={roi_id} OCR result: {markdown}")
        except Exception as e:
            logger.exception(f"roi_id={roi_id} OCR error: {e}")
    else:
        logger.info(f"OCR skipped for ROI {roi_id} (throttled)")

    return frame
