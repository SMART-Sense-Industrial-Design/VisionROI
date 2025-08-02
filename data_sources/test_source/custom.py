# data_sources/<your_source>/custom.py
from src.packages.models.yolov8.yolov8onnx.yolov8object.YOLOv8 import YOLOv8

# โหลดโมเดล (ใส่ path ไปยังไฟล์ .onnx ของคุณ)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

def process(frame):
    # ตรวจจับวัตถุ
    # boxes, scores, class_ids = model(frame)
    # สามารถวาดกรอบหรือประมวลผลต่อได้ตามต้องการ
    return frame
