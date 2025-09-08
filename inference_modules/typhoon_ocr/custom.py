from __future__ import annotations

import base64
import threading

from inference_modules.base_ocr import BaseOCR, np, Image, cv2
from typhoon_ocr import ocr_document


class TyphoonOCR(BaseOCR):
    MODULE_NAME = "typhoon_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._cv_lock = threading.Lock()

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        result_text = ""
        try:
            with self._cv_lock:
                _, buffer = cv2.imencode(".jpg", frame)
            base64_string = base64.b64encode(buffer).decode("utf-8")
            result_text = ocr_document(base64_string)
            self.logger.info(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {result_text}"
                if roi_id is not None
                else f"{self.MODULE_NAME} OCR result: {result_text}"
            )
        except Exception as e:  # pragma: no cover
            self.logger.exception(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {e}"
            )
        if save:
            self._save_image(frame.copy(), roi_id, source)
        return result_text

    def _update_save_flag(self, cam_id: int | None) -> None:
        if cam_id is not None:
            try:
                import app  # type: ignore
                app.save_roi_flags[cam_id] = True
            except Exception:  # pragma: no cover
                pass
