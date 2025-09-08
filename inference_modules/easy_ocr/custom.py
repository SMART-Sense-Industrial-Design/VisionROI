from __future__ import annotations

import threading

from inference_modules.base_ocr import BaseOCR, np, Image, cv2

try:
    import easyocr
except Exception:  # pragma: no cover - fallback when easyocr missing
    easyocr = None


class EasyOCR(BaseOCR):
    MODULE_NAME = "easy_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._reader: easyocr.Reader | None = None  # type: ignore[name-defined]
        self._reader_lock = threading.Lock()

    def _get_reader(self) -> easyocr.Reader:
        if easyocr is None:
            raise RuntimeError("easyocr library is not installed")
        with self._reader_lock:
            if self._reader is None:
                self._reader = easyocr.Reader(["en", "th"], gpu=False)
            return self._reader

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        reader = self._get_reader()
        ocr_result = reader.readtext(frame, detail=0)
        text = " ".join(ocr_result)
        self.logger.info(
            f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {text}"
            if roi_id is not None
            else f"{self.MODULE_NAME} OCR result: {text}"
        )
        if save:
            self._save_image(frame, roi_id, source)
        return text

    def _update_save_flag(self, cam_id: int | None) -> None:
        if cam_id is not None:
            try:
                import app  # type: ignore
                app.save_roi_flags[cam_id] = False
            except Exception:  # pragma: no cover
                pass

    def _cleanup_extra(self) -> None:
        with self._reader_lock:
            self._reader = None
