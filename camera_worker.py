# camera_worker.py
import cv2
import time
import threading
import asyncio
from typing import Optional
from queue import Queue, Empty
from contextlib import contextmanager

@contextmanager
def silent():
    try:
        yield
    except Exception:
        pass

def _is_rtsp(src) -> bool:
    return isinstance(src, str) and src.strip().lower().startswith("rtsp://")

class CameraWorker:
    def __init__(self, src,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 read_interval: float = 0.0) -> None:
        self.src = src
        self.loop = loop
        self.width = width
        self.height = height
        self.read_interval = read_interval

        # บน macOS/RTSP ลองใช้ FFMPEG backend ก่อน (ช่วยลด segfault จาก AVFoundation)
        if _is_rtsp(src):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)

        self._cap = cap

        with silent():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:
            with silent():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            with silent():
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q: Queue = Queue(maxsize=1)

    def start(self) -> bool:
        if not self._cap or not self._cap.isOpened():
            return False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True

    def _run(self) -> None:
        # ป้องกันการ read ต่อหลังถูกสั่งหยุด
        while not self._stop_evt.is_set():
            ok, frame = self._cap.read() if (self._cap is not None) else (False, None)
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # สำคัญ: copy() ตัดขาดจากบัฟเฟอร์เดิม ลด use-after-free/reuse
            try:
                frame_copy = frame.copy()
            except Exception:
                time.sleep(0.01)
                continue

            if self._q.full():
                try:
                    _ = self._q.get_nowait()
                except Empty:
                    pass
            try:
                self._q.put_nowait(frame_copy)
            except Exception:
                pass

            if self.read_interval > 0:
                time.sleep(self.read_interval)

    async def read(self):
        try:
            return self._q.get_nowait()
        except Empty:
            pass
        for _ in range(3):
            await asyncio.sleep(0.03)
            try:
                return self._q.get_nowait()
            except Empty:
                continue
        return None

    async def stop(self) -> None:
        self._stop_evt.set()

        # ปล่อยกล้อง (ปลดบล็อค read) แล้วพักเล็กน้อย ลด crash race
        with silent():
            if self._cap is not None and self._cap.isOpened():
                await asyncio.to_thread(self._cap.release)
        # ให้ backend ปิดเธรดภายในก่อนเล็กน้อย
        await asyncio.sleep(0.05)
        self._cap = None  # กันโค้ดส่วนอื่นเผลอใช้ต่อ

        with silent():
            if self._thread is not None and self._thread.is_alive():
                await asyncio.to_thread(self._thread.join, 0.5)

        # ล้างคิว
        while True:
            try:
                _ = self._q.get_nowait()
            except Empty:
                break
