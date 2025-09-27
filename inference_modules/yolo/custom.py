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



def process(
    frame,
    roi_id=None,
    save=False,
    source="",
    cam_id: int | None = None,
):
    """ประมวลผล ROI และเรียก OCR ทุกเฟรม
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
        # save_image_async(str(path), frame.copy())

    # else:
    #     logger.info(f"OCR skipped for ROI {roi_id} (throttled)")


    return frame
