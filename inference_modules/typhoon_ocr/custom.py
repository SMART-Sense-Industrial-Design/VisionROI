from __future__ import annotations

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

os.environ["TYPHOON_OCR_API_KEY"] = "sk-UgKIYNT2ZaU0Ph3bZ5O8rfHc9QBJLNz5yQtshQldHf5Gw8gD"  # หรือใส่ TYPHOON_OCR_API_KEY ก็ได้

MODULE_NAME = "typhoon_ocr"
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.INFO)
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"


# โหลดโมเดล (ถ้ามี)
# model = YOLOv8("data_sources/<your_source>/model.onnx")

# ตัวแปรควบคุมเวลาเรียก OCR แยกตาม roi พร้อมตัวล็อกป้องกันการเข้าถึงพร้อมกัน
last_ocr_times = {}
last_ocr_results = {}
_last_ocr_lock = threading.Lock()


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



class TyphoonOCR(BaseOCR):
    MODULE_NAME = "typhoon_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._cv_lock = threading.Lock()

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        result_text = ""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_string = base64.b64encode(buffer).decode('utf-8')
            markdown = ocr_document(base64_string)
            logger.info(
                f"roi_id={roi_id} {MODULE_NAME} OCR result: {markdown}"

                if roi_id is not None
                else f"{self.MODULE_NAME} OCR result: {result_text}"
            )
        except Exception as e:  # pragma: no cover
            self.logger.exception(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {e}"
            )
            roi_folder = f"{roi_id}" if roi_id is not None else "roi"
            save_dir = base_dir / "images" / roi_folder
            os.makedirs(save_dir, exist_ok=True)
            filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
            path = save_dir / filename
            save_image_async(str(path), frame.copy())


        return result_text

    def _update_save_flag(self, cam_id: int | None) -> None:
        if cam_id is not None:
            try:
                import app  # type: ignore
                app.save_roi_flags[cam_id] = True
            except Exception:  # pragma: no cover
                pass
