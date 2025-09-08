from __future__ import annotations

import threading

from inference_modules.base_ocr import BaseOCR, np, Image, cv2

try:
    from rapidocr import RapidOCR
except Exception:  # pragma: no cover - fallback when rapidocr missing
    RapidOCR = None  # type: ignore[assignment]


class RapidOCR(BaseOCR):
    MODULE_NAME = "rapid_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._reader = None
        self._reader_lock = threading.Lock()

    def _get_reader(self):
        if RapidOCR is None:
            raise RuntimeError("rapidocr library is not installed")
        with self._reader_lock:
            if self._reader is None:
                self._reader = RapidOCR()
            return self._reader

    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        text = ""
        try:
            reader = self._get_reader()
            result = reader(frame)
            if (
                isinstance(result, (list, tuple))
                and len(result) == 2
                and isinstance(result[0], (list, tuple))
                and not isinstance(result[1], (list, tuple, dict))
            ):
                ocr_result = result[0]
            else:
                ocr_result = result

            text_items: list[str] = []
            if isinstance(ocr_result, (list, tuple)):
                for res in ocr_result:
                    if isinstance(res, (list, tuple)) and len(res) > 1:
                        text_items.append(str(res[1]))
                    elif isinstance(res, dict) and "text" in res:
                        text_items.append(str(res["text"]))
            elif isinstance(ocr_result, dict) and "text" in ocr_result:
                text_items.append(str(ocr_result["text"]))
            elif hasattr(ocr_result, "text"):
                text_items.append(str(getattr(ocr_result, "text")))
            elif hasattr(ocr_result, "texts"):
                texts_attr = getattr(ocr_result, "texts")
                if isinstance(texts_attr, (list, tuple)):
                    text_items.extend(str(t) for t in texts_attr)
                elif texts_attr is not None:
                    text_items.append(str(texts_attr))
            elif hasattr(ocr_result, "txts"):
                txts_attr = getattr(ocr_result, "txts")
                if isinstance(txts_attr, (list, tuple)):
                    text_items.extend(str(t) for t in txts_attr)
                elif txts_attr is not None:
                    text_items.append(str(txts_attr))
            text = " ".join(text_items)
            self.logger.info(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {text}"
                if roi_id is not None
                else f"{self.MODULE_NAME} OCR result: {text}"
            )
        except Exception as e:  # pragma: no cover - log any OCR error
            self.logger.exception(f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {e}")
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
