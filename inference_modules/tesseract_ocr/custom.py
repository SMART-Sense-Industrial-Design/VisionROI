from __future__ import annotations

import threading
from typing import Any

from inference_modules.base_ocr import BaseOCR, Image, cv2, np

try:  # pragma: no cover - ข้ามเมื่อนำเข้า pytesseract ไม่สำเร็จ
    import pytesseract
except Exception:  # pragma: no cover - fallback เมื่อไม่มี pytesseract
    pytesseract = None  # type: ignore[assignment]

MODULE_NAME = "tesseract_ocr"


class TesseractOCR(BaseOCR):
    """โมดูล OCR ที่ใช้ Tesseract ผ่าน pytesseract"""

    MODULE_NAME = MODULE_NAME

    def __init__(self, lang: str = "eng+tha", config: str | None = None) -> None:
        super().__init__()
        self.lang = lang
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
