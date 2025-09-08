from __future__ import annotations

import base64
import threading
from pathlib import Path
from src.utils.image import save_image_async
try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None

os.environ["TYPHOON_OCR_API_KEY"] = "sk-UgKIYNT2ZaU0Ph3bZ5O8rfHc9QBJLNz5yQtshQldHf5Gw8gD"  # หรือใส่ TYPHOON_OCR_API_KEY ก็ได้

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODULE_NAME = "typhoon_ocr"

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_handler: TimedRotatingFileHandler | None = None
_current_source: str | None = None
_data_sources_root = Path(__file__).resolve().parents[2] / "data_sources"


def _configure_logger(source: str | None) -> None:
    """ตั้งค่า handler ของ logger ให้บันทึกตามโฟลเดอร์ของ source"""
    global _handler, _current_source
    source = source or ""
    if _current_source == source:
        return
    log_dir = _data_sources_root / source if source else Path(__file__).resolve().parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "custom.log"
    if _handler:
        logger.removeHandler(_handler)
        _handler.close()
    _handler = TimedRotatingFileHandler(
        str(log_path), when="D", interval=1, backupCount=7
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _current_source = source


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

    _configure_logger(source)

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
