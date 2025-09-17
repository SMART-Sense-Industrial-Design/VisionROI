from __future__ import annotations

import time
from logging import Logger
from PIL import Image
import cv2
import os
from datetime import datetime
import threading
from pathlib import Path
import gc

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None


class BaseOCR:
    """คลาสพื้นฐานสำหรับโมดูล OCR แต่ละชนิด"""

    MODULE_NAME = "base_ocr"

    def __init__(self) -> None:
        from src.utils.logger import get_logger as _get_shared_logger

        self._get_shared_logger = _get_shared_logger
        self.logger = self._get_shared_logger(self.MODULE_NAME)
        # base_ocr อยู่ในโฟลเดอร์ inference_modules ดังนั้น parents[1] คือ root
        self._data_sources_root = Path(__file__).resolve().parents[1] / "data_sources"

        self.last_ocr_times: dict = {}
        self.last_ocr_results: dict = {}
        self._last_ocr_lock = threading.Lock()
        self._imwrite_lock = threading.Lock()

    # ------------------------- logging -------------------------
    def get_logger(self, source: str | None) -> Logger:
        """คืน logger ที่บันทึก log ตาม source"""
        self.logger = self._get_shared_logger(self.MODULE_NAME, source)
        return self.logger

    # ------------------------- image saving -------------------------
    def _save_image_async(self, path: str, image) -> None:
        try:
            with self._imwrite_lock:
                cv2.imwrite(path, image)
        except Exception as e:  # pragma: no cover
            self.logger.exception(f"Failed to save image {path}: {e}")

    def _save_image(self, frame, roi_id, source: str) -> None:
        base_dir = (
            self._data_sources_root / source
            if source
            else Path(__file__).resolve().parent
        )
        roi_folder = f"{roi_id}" if roi_id is not None else "roi"
        save_dir = base_dir / "images" / roi_folder
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        path = save_dir / filename
        threading.Thread(
            target=self._save_image_async, args=(str(path), frame), daemon=True
        ).start()

    # ------------------------- hooks -------------------------
    def _run_ocr(self, frame, roi_id, save: bool, source: str) -> str:
        raise NotImplementedError

    def _update_save_flag(self, cam_id: int | None) -> None:
        """อัปเดตธงบันทึกรูปจาก app หากจำเป็น"""
        # โมดูลที่ต้องการพฤติกรรมพิเศษจะ override เมธอดนี้
        pass

    # ------------------------- main APIs -------------------------
    def process(
        self,
        frame,
        roi_id=None,
        save: bool = False,
        source: str = "",
        cam_id: int | None = None,
        interval: float = 1.0,
    ):
        """ประมวลผล ROI และเรียก OCR เมื่อถึงเวลา"""
        self.get_logger(source)
        self._update_save_flag(cam_id)

        if isinstance(frame, Image.Image) and np is not None:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        current_time = time.monotonic()
        with self._last_ocr_lock:
            last_time = self.last_ocr_times.get(roi_id)
        diff_time = 0 if last_time is None else current_time - last_time
        should_ocr = last_time is None or diff_time >= interval

        if should_ocr:
            with self._last_ocr_lock:
                self.last_ocr_times[roi_id] = current_time
            text = self._run_ocr(frame, roi_id, save, source)
            with self._last_ocr_lock:
                self.last_ocr_results[roi_id] = text
            return text
        return None

    def stop(self, roi_id) -> None:
        with self._last_ocr_lock:
            self.last_ocr_times.pop(roi_id, None)
            self.last_ocr_results.pop(roi_id, None)

    def _cleanup_extra(self) -> None:
        """ให้คลาสลูก override หากต้องเคลียร์ทรัพยากรเพิ่ม"""
        pass

    def cleanup(self) -> None:
        with self._last_ocr_lock:
            self.last_ocr_times.clear()
            self.last_ocr_results.clear()
        self._cleanup_extra()
        gc.collect()
