import cv2
import threading
import queue
import time
import sys
import os
from typing import Optional, Tuple

# ลดปัญหา race/memory บนบางแพลตฟอร์ม
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(1)
except Exception:
    pass


class CameraWorker:
    """
    เปิดกล้องแบบปลอดภัยด้วยเธรดเดียว
    - ใช้คิว 1 ช่อง ตัดเฟรมเก่า
    - ปิดแบบเป็นลำดับ: stop flag -> join thread -> release capture -> clear buffers
    - ไม่ release ขณะเธรดอ่านทำงาน
    """

    def __init__(self, source, loop, width: Optional[int] = None, height: Optional[int] = None):
        self._source = source
        self._loop = loop
        self._width = width
        self._height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._opened = False

        # คิว 1 ช่อง ลด memory growth
        self._q: "queue.Queue[Optional[object]]" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

    # -------- public API --------
    def start(self) -> bool:
        with self._lock:
            if self._opened:
                return True

            # เลือก backend เสถียรตามแพลตฟอร์ม
            backend = 0
            if sys.platform == "darwin":
                backend = cv2.CAP_AVFOUNDATION  # เสถียรบน macOS
            # บน Linux (Jetson) ปล่อยให้ OpenCV auto-เลือก หรือคุณจะลอง CAP_V4L2 ก็ได้

            try:
                self._cap = cv2.VideoCapture(self._source, backend)
            except Exception:
                self._cap = cv2.VideoCapture(self._source)  # fallback

            if not (self._cap and self._cap.isOpened()):
                self._safe_release()
                return False

            # ลดบัฟเฟอร์ภายใน ถ้า backend รองรับ
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if self._width:
                try:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._width))
                except Exception:
                    pass
            if self._height:
                try:
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._height))
                except Exception:
                    pass

            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="CameraWorkerThread", daemon=True)
            self._thread.start()
            self._opened = True
            return True

    async def read(self):
        """
        ดึงเฟรมล่าสุดแบบ non-blocking (async เรียกได้)
        คืนค่า ndarray (BGR) หรือ None (ถ้าไม่มีเฟรม)
        """
        try:
            frame = self._q.get_nowait()
        except queue.Empty:
            return None
        if frame is None:
            return None
        return frame

    async def stop(self) -> None:
        """
        หยุดแบบปลอดภัย:
        - set stop flag
        - join เธรดอ่าน
        - release capture
        - ล้างคิว/อ็อบเจ็กต์
        """
        with self._lock:
            if not self._opened:
                self._clear_queue()
                self._safe_release()
                return

            # 1) ส่งสัญญาณหยุด
            self._stop.set()

            # 2) ปลดบล็อกเธรดอ่าน ถ้าคิวเต็ม
            self._clear_queue(put_none=True)

            # 3) รอเธรดปิด
            t = self._thread
            if t is not None and t.is_alive():
                t.join(timeout=1.5)
            self._thread = None

            # 4) ปล่อย capture หลังเธรดหยุดจริง
            self._safe_release()

            # 5) ล้างอีกครั้งและ mark ปิด
            self._clear_queue()
            self._opened = False

    # -------- internal helpers --------
    def _run(self):
        cap = self._cap
        if cap is None:
            return

        # อ่านวนจนสั่งหยุด
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                # กล้องสะดุด: หน่วงสั้น ๆ ไม่ spin
                time.sleep(0.01)
                continue

            # ตัดคอเฟรมเก่าถ้าคิวเต็ม
            if self._q.full():
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._q.put_nowait(frame)
            except queue.Full:
                # ถ้าเต็มจริง ๆ ก็ทิ้งเฟรมนี้
                pass

        # จะหยุดแล้ว: ใส่ None เพื่อปลดบล็อก consumer ทั้งหมด
        self._clear_queue(put_none=True)

    def _clear_queue(self, put_none: bool = False):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if put_none:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                # ถ้าเต็มก็เคลียร์แล้วใส่ None อีกรอบ
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(None)
                except Exception:
                    pass

    def _safe_release(self):
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # เข้าถึง VideoCapture จากภายนอกเพื่อการทดสอบ
    @property
    def cap(self):
        return self._cap
