from __future__ import annotations

import threading
from typing import Any

from inference_modules.base_ocr import BaseOCR, Image, cv2, np

try:  # pragma: no cover - ข้ามเมื่อไม่สามารถนำเข้า torch ได้
    import torch
except Exception:  # pragma: no cover - กรณีไม่มี torch
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - ข้ามเมื่อไม่สามารถนำเข้า transformers ได้
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception:  # pragma: no cover - กรณีไม่มี transformers
    TrOCRProcessor = None  # type: ignore[assignment]
    VisionEncoderDecoderModel = None  # type: ignore[assignment]

MODULE_NAME = "trocr"


class TrOCROCR(BaseOCR):
    """โมดูล OCR ที่ใช้โมเดล TrOCR จาก HuggingFace"""

    MODULE_NAME = MODULE_NAME

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        super().__init__()
        self.model_name = model_name or "microsoft/trocr-small-printed"
        self.device_override = device
        self._model_lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None

    # ------------------------- helpers -------------------------
    def _prepare_image(self, frame: Any) -> Image.Image:
        """แปลงอินพุตให้เป็นภาพ RGB สำหรับ TrOCR"""
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        if np is not None and isinstance(frame, np.ndarray):
            if frame.ndim == 2:  # grayscale
                return Image.fromarray(frame).convert("RGB")
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(converted)

        raise TypeError(f"Unsupported frame type for TrOCROCR: {type(frame)!r}")

    def _get_device(self) -> str:
        if self.device_override:
            return self.device_override
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - ขึ้นกับฮาร์ดแวร์
            return "cuda"
        return "cpu"

    def _load_components(self) -> tuple[Any, Any]:
        """โหลด processor และ model แบบ lazy พร้อมล็อก"""
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None or torch is None:
            raise RuntimeError("TrOCR dependencies are not installed")

        with self._model_lock:
            if self._processor is None or self._model is None:
                self._processor = TrOCRProcessor.from_pretrained(self.model_name)
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                self._model.to(self._get_device())
            return self._processor, self._model

    def _generate_text(self, image: Image.Image) -> str:
        processor, model = self._load_components()
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self._get_device())
        generated_ids = model.generate(pixel_values)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return texts[0].strip() if texts else ""

    # ------------------------- hooks implementations -------------------------
    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:  # type: ignore[override]
        text = ""
        try:
            image = self._prepare_image(frame)
            text = self._generate_text(image)
        except Exception as exc:  # pragma: no cover - จัดการข้อผิดพลาด OCR
            self.logger.exception(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR error: {exc}"
            )

        if text:
            self.logger.info(
                f"roi_id={roi_id} {self.MODULE_NAME} OCR result: {text}"
                if roi_id is not None
                else f"{self.MODULE_NAME} OCR result: {text}"
            )

        if save:
            self._save_image(frame, roi_id, source)

        return text

    def _cleanup_extra(self) -> None:
        with self._model_lock:
            self._processor = None
            self._model = None


_ocr_instance: TrOCROCR | None = None
_ocr_instance_lock = threading.Lock()


def _get_ocr_instance() -> TrOCROCR:
    global _ocr_instance
    with _ocr_instance_lock:
        if _ocr_instance is None:
            _ocr_instance = TrOCROCR()
        return _ocr_instance


def process(
    frame,
    roi_id=None,
    save: bool = False,
    source: str = "",
    cam_id: int | None = None,
    interval: float = 1.0,
):
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
    global _ocr_instance
    with _ocr_instance_lock:
        if _ocr_instance is not None:
            _ocr_instance.cleanup()
            _ocr_instance = None
