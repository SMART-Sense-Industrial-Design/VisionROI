# data_sources/<your_source>/custom.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from src.packages.models.yolov8.yolov8onnx.yolov8object.YOLOv8 import YOLOv8

# ตั้งค่า logging โดยเก็บไฟล์ log ไว้ 7 วัน
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_path = os.path.join(os.path.dirname(__file__), "custom.log")
_handler = TimedRotatingFileHandler(_log_path, when="D", interval=1, backupCount=7)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# โหลดโมเดล (ใส่ path ไปยังไฟล์ .onnx ของคุณ)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

def process(frame, roi_id=None):
    """ประมวลผลเฟรมตาม ROI ที่ระบุและบันทึกค่าไว้ใน log"""
    logger.info("process called with roi_id=%s", roi_id)
    # ตรวจจับวัตถุภายใน ROI ที่ระบุ
    # boxes, scores, class_ids = model(frame)
    # สามารถใช้ roi_id เพื่อตัดสินใจเพิ่มเติมได้
    return frame
