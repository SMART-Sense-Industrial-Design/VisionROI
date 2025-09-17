from __future__ import annotations

import threading
from typing import Any

from inference_modules.base_ocr import BaseOCR, Image, cv2, np

try:  # pragma: no cover - ข้ามเมื่อนำเข้า pytesseract ไม่สำเร็จ
    import pytesseract
except Exception:  # pragma: no cover - fallback เมื่อไม่มี pytesseract
    pytesseract = None  # type: ignore[assignment]

MODULE_NAME = "tesseract_ocr"
DEFAULT_TESSERACT_LANG = "eng"

_ocr_instance: "TesseractOCR" | None = None
_ocr_instance_lock = threading.Lock()


class TesseractOCR(BaseOCR):
    """โมดูล OCR ที่ใช้ Tesseract ผ่าน pytesseract"""

    MODULE_NAME = MODULE_NAME

    def __init__(self, lang: str | None = None, config: str | None = None) -> None:
        super().__init__()
        # บังคับให้ Tesseract ใช้ภาษาอังกฤษเสมอ แม้จะมีการส่งค่าภาษาอื่นเข้ามา
        self.lang = DEFAULT_TESSERACT_LANG
        self.config = config
        self._ocr_lock = threading.Lock()

    # ------------------------- internal helpers -------------------------
    def _ensure_pil_image(self, frame: Any) -> Image.Image:
        """แปลงข้อมูลภาพให้อยู่ในรูป PIL.Image"""
        if isinstance(frame, Image.Image):
            return frame

        if np is not None and isinstance(frame, np.ndarray):
            if frame.ndim == 2:
                return Image.fromarray(frame)
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(converted)

        raise TypeError(
            f"Unsupported frame type for TesseractOCR: {type(frame)!r}"
        )

    # ------------------------- hooks implementations -------------------------
    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:  # type: ignore[override]
        text = ""
        if pytesseract is not None:
            try:
                pil_image = self._ensure_pil_image(frame)
                with self._ocr_lock:
                    text = pytesseract.image_to_string(
                        pil_image, lang=self.lang, config=self.config
                    ).strip()
            except Exception as exc:  # pragma: no cover - จัดการข้อผิดพลาด OCR
                self.logger.exception(
                    f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {exc}"
                )
        else:  # pragma: no cover - แจ้งเตือนเมื่อไม่มี pytesseract
            self.logger.error("pytesseract library is not installed")

        if text:
            self.logger.info(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {text}"
                if roi_id is not None
                else f"{self.MODULE_NAME} OCR result: {text}"
            )

        if save:
            self._save_image(frame, roi_id, source)

        return text


def _get_ocr_instance() -> "TesseractOCR":
    """สร้างและคืนอินสแตนซ์ OCR แบบ singleton เพื่อใช้ร่วมกันทั้งโมดูล"""

    global _ocr_instance
    with _ocr_instance_lock:
        if _ocr_instance is None:
            _ocr_instance = TesseractOCR()
        return _ocr_instance


def process(
    frame,
    roi_id=None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
    interval: float = 1.0,
):
    """จุดเข้าใช้งานหลักให้สอดคล้องกับโมดูล inference อื่น ๆ"""

    ocr = _get_ocr_instance()
    return ocr.process(
        frame,
        roi_id=roi_id,
        save=save,
        source=source,
        cam_id=cam_id,
        interval=interval,
    )


def cleanup() -> None:
    """คืนทรัพยากรของอินสแตนซ์ OCR และรีเซ็ต singleton"""

    global _ocr_instance
    with _ocr_instance_lock:
        if _ocr_instance is not None:
            _ocr_instance.cleanup()
            _ocr_instance = None
