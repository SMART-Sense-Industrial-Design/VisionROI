# camera_worker.py
import cv2
import time
import threading
import asyncio
import gc
import subprocess
from typing import Optional
from queue import Queue, Empty
from contextlib import contextmanager

try:
    import numpy as np
except Exception:
    np = None

@contextmanager
def silent():
    try:
        yield
    except Exception:
        pass


def _malloc_trim() -> None:
    """คืนพื้นที่ heap ให้ OS ถ้าเป็นไปได้ (Linux/glibc)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

class CameraWorker:
    def __init__(
        self,
        src,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        read_interval: float = 0.0,
        backend: str = "opencv",
    ) -> None:
        self.src = src
        self.loop = loop
        self.width = width
        self.height = height
        self.read_interval = read_interval
        self.backend = backend

        self._cap = None
        self._proc: subprocess.Popen | None = None

        if backend == "ffmpeg":
            if self.width is None or self.height is None:
                self.width, self.height = self._probe_resolution(src)
            cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(src),
            ]
            if self.width and self.height:
                cmd += ["-vf", f"scale={int(self.width)}:{int(self.height)}"]
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
            with silent():
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, bufsize=10**8
                )
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

    def _probe_resolution(self, src) -> tuple[Optional[int], Optional[int]]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(src),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            w, h = res.stdout.strip().split("x")
            return int(w), int(h)
        except Exception:
            return None, None

    def start(self) -> bool:
        if self.backend == "ffmpeg":
            if self._proc is None or self._proc.poll() is not None:
                return False
        else:
            if not self._cap or not self._cap.isOpened():
                return False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True

    def _run(self) -> None:
        # ป้องกันการ read ต่อหลังถูกสั่งหยุด
        if self.backend == "ffmpeg":
            frame_size = (
                int(self.width) * int(self.height) * 3
                if self.width and self.height
                else 0
            )
        while not self._stop_evt.is_set():
            if self.backend == "ffmpeg":
                if self._proc is None or self._proc.poll() is not None:
                    time.sleep(0.02)
                    continue
                if frame_size <= 0 or np is None:
                    time.sleep(0.02)
                    continue
                raw = self._proc.stdout.read(frame_size)
                if not raw or len(raw) != frame_size:
                    time.sleep(0.02)
                    continue
                frame = np.frombuffer(raw, np.uint8).reshape(
                    (int(self.height), int(self.width), 3)
                )
            else:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.02)
                    continue
                ok, frame = self._cap.read()
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
        with silent():
            if self._thread is not None and self._thread.is_alive():
                await asyncio.to_thread(self._thread.join, 0.5)

        # ปล่อยกล้อง/โปรเซส (ปลดบล็อค read) แล้วพักเล็กน้อย ลด crash race
        if self.backend == "ffmpeg":
            with silent():
                if self._proc is not None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=0.5)
                    except Exception:
                        pass
            self._proc = None
        else:
            with silent():
                if self._cap is not None and self._cap.isOpened():
                    await asyncio.to_thread(self._cap.release)
            await asyncio.sleep(0.05)
            self._cap = None  # กันโค้ดส่วนอื่นเผลอใช้ต่อ

        # ล้างคิว
        while True:
            try:
                _ = self._q.get_nowait()
            except Empty:
                break

        # เก็บขยะและคืนหน่วยความจำกลับให้ระบบ
        gc.collect()
        _malloc_trim()

