import time
from src.packages.models.yolov8.yolov8onnx.yolov8object.YOLOv8 import YOLOv8
from typhoon_ocr import ocr_document
from PIL import Image
import cv2
import base64
from logging.handlers import TimedRotatingFileHandler
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_path = os.path.join(os.path.dirname(__file__), "custom.log")
_handler = TimedRotatingFileHandler(_log_path, when="D", interval=1, backupCount=7)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)


# โหลดโมเดล (ถ้ามี)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

# ตัวแปรควบคุมเวลาเรียก OCR
last_ocr_time = time.time()  # หน่วยเป็นวินาที (timestamp)

def process(frame, roi_id=None):
    global last_ocr_time

    # เช็คว่าเวลาห่างจากการเรียกครั้งก่อนครบ 2 วินาทีหรือยัง
    current_time = time.time()
    logger.info(f"current_time - last_ocr_time = {current_time - last_ocr_time}")
    if current_time - last_ocr_time >= 2:
        _, buffer = cv2.imencode('.jpg', frame)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        markdown = ocr_document(base64_string)
        logger.info(f"roi_id={roi_id} OCR result: {markdown}")
        last_ocr_time = current_time  # อัปเดตเวลาใหม่
    else:
        logger.info(f"OCR skipped for ROI {roi_id} (throttled)")

    return frame
