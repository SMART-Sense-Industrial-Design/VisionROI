from concurrent.futures import ThreadPoolExecutor

import sys
import types

import pytest


np = pytest.importorskip("numpy")


# สร้างโมดูล PIL แบบจำลองเพื่อหลีกเลี่ยงการติดตั้งจริง
pil_image_module = types.ModuleType("PIL.Image")


class DummyImage:  # pragma: no cover - โครงร่างสำหรับการตรวจสอบ isinstance
    pass


pil_image_module.Image = DummyImage
pil_package = types.ModuleType("PIL")
pil_package.Image = pil_image_module
sys.modules["PIL"] = pil_package
sys.modules["PIL.Image"] = pil_image_module


from inference_modules.easy_ocr import custom as mod


def test_process_multiple_roi_thread_safe(tmp_path, monkeypatch):
    """เรียก process พร้อมกันหลาย ROI เพื่อทดสอบว่าการบันทึกรูปปลอดภัย"""

    monkeypatch.setattr(mod, "_data_sources_root", tmp_path)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def _call(i):
        return mod.process(frame.copy(), roi_id=i, save=True, source="test")

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_call, i) for i in range(5)]
        results = [f.result() for f in futures]

    assert len(results) == 5
