"""Utility helpers สำหรับ import ``cv2`` ให้ปลอดภัยในสภาพแวดล้อมที่ไม่มี backend."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

try:  # optional dependency สำหรับสร้าง dummy frame
    import numpy as _np
except Exception:  # pragma: no cover - numpy อาจไม่พร้อมใช้งาน
    _np = None


_DEFAULT_CAP_PROP_VALUES = {
    "CAP_PROP_FRAME_WIDTH": 3,
    "CAP_PROP_FRAME_HEIGHT": 4,
    "CAP_PROP_BUFFERSIZE": 38,
}


class _UnavailableVideoCapture:
    """Dummy VideoCapture ที่คืนค่าพื้นฐานแทนการล้มเหลว."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        # พร้อมใช้งานทันทีเพื่อให้โค้ดหลักไม่เจอ AttributeError
        self._opened = True
        self.frames_read = 0
        self._last_frame = None
        self.settings: dict[Any, Any] = {}

    def isOpened(self) -> bool:  # noqa: D401
        return self._opened

    def read(self) -> tuple[bool, Any]:  # noqa: D401
        # จำลองความหน่วงเล็กน้อยเหมือนกล้องจริงเพื่อไม่ให้ event loop วิ่งเร็วผิดปกติ
        import time

        time.sleep(0.005)
        self.frames_read += 1
        if _np is not None:
            if self._last_frame is None:
                self._last_frame = _np.zeros((1, 1, 3), dtype=_np.uint8)
            return True, self._last_frame
        return True, b"frame"

    def set(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        if len(args) >= 2:
            self.settings[args[0]] = args[1]
        return None

    def release(self) -> None:  # noqa: D401
        self._opened = False
        return None


def ensure_cv2() -> ModuleType:
    """คืนโมดูล ``cv2`` ที่มีแอตทริบิวต์พื้นฐานครบและพร้อม monkeypatch."""

    cv2_mod = sys.modules.get("cv2")
    if cv2_mod is None:
        try:
            import cv2 as cv2_mod  # type: ignore
        except Exception:  # pragma: no cover - กรณีไม่มี OpenCV เลย
            cv2_mod = ModuleType("cv2")
        sys.modules.setdefault("cv2", cv2_mod)

    if not hasattr(cv2_mod, "VideoCapture"):
        cv2_mod.VideoCapture = _UnavailableVideoCapture  # type: ignore[attr-defined]

    for name, value in _DEFAULT_CAP_PROP_VALUES.items():
        if not hasattr(cv2_mod, name):
            setattr(cv2_mod, name, value)

    return cv2_mod


class _Cv2Proxy:
    """พร็อกซีที่สะท้อน ``sys.modules['cv2']`` แบบเรียลไทม์."""

    def __getattr__(self, name: str) -> Any:  # noqa: D401
        module = ensure_cv2()
        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D401
        module = ensure_cv2()
        setattr(module, name, value)

    def __delattr__(self, name: str) -> None:  # noqa: D401
        module = ensure_cv2()
        delattr(module, name)

    def __repr__(self) -> str:  # noqa: D401
        module = ensure_cv2()
        return f"<Cv2Proxy for {module!r}>"


cv2 = _Cv2Proxy()

__all__ = ["cv2", "ensure_cv2"]

