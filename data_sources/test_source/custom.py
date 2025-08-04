# data_sources/<your_source>/custom.py
from src.packages.models.yolov8.yolov8onnx.yolov8object.YOLOv8 import YOLOv8

# โหลดโมเดล (ใส่ path ไปยังไฟล์ .onnx ของคุณ)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

def process(frame, roi_id=None):
    # ตรวจจับวัตถุภายใน ROI ที่ระบุ
    # boxes, scores, class_ids = model(frame)
    # สามารถใช้ roi_id เพื่อตัดสินใจเพิ่มเติมได้
    return frame
