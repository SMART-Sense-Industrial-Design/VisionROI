# camera_worker.py
import cv2
import time
import threading
import asyncio
import gc
import subprocess
import os
from typing import Optional, Tuple
from queue import Queue, Empty
from contextlib import contextmanager
from collections import deque

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
    """
    backend='opencv' : ใช้ cv2.VideoCapture
    backend='ffmpeg' : ใช้ ffmpeg ถ่ม bgr24/rawvideo ผ่าน stdout
    """

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
        self._stderr_thread: Optional[threading.Thread] = None
        self._last_stderr = deque(maxlen=200)  # เก็บบรรทัดล่าสุดไว้ดู error

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q: Queue = Queue(maxsize=1)

        if backend == "ffmpeg":
            # ถ้ายังไม่รู้ขนาด ลอง probe; ถ้าไม่ได้จะ fallback ไป OpenCV 1 เฟรม
            if self.width is None or self.height is None:
                w, h = self._probe_resolution(str(src))
                if w and h:
                    self.width, self.height = w, h
                else:
                    w2, h2 = self._probe_with_opencv_once(str(src))
                    if w2 and h2:
                        self.width, self.height = w2, h2

            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin"]

            if self._is_rtsp(str(src)):
                cmd += [
                    "-rtsp_transport", "tcp",
                    "-rtsp_flags", "prefer_tcp",
                    "-rw_timeout", "7000000",   # 7s (microseconds)
                    "-max_delay", "500000",     # 0.5s (microseconds)
                ]

            # ช่วยให้วิเคราะห์สตรีมในเครือข่ายที่หน่วง
            cmd += ["-probesize", "32M", "-analyzeduration", "5M"]

            cmd += [
                "-fflags", "+discardcorrupt",
                "-i", str(src),
                "-map", "0:v:0",
                "-an",
            ]

            if self.width and self.height:
                cmd += ["-vf", f"scale={int(self.width)}:{int(self.height)}:flags=lanczos"]

            cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

            with silent():
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**8,
                    preexec_fn=os.setsid,  # Linux: แยก process group
                )
                if self._proc and self._proc.stderr:
                    self._stderr_thread = threading.Thread(
                        target=self._drain_stderr, daemon=True
                    )
                    self._stderr_thread.start()

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

    # ---------- helpers ----------

    @staticmethod
    def _is_rtsp(src: str) -> bool:
        s = str(src).lower()
        return s.startswith("rtsp://") or s.startswith("rtsps://")

    def _probe_resolution(self, src: str) -> Tuple[Optional[int], Optional[int]]:
        """ใช้ ffprobe หา width/height พร้อม RTSP/timeout"""
        cmd = ["ffprobe", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self._is_rtsp(src):
            cmd += [
                "-rtsp_transport", "tcp",
                "-rtsp_flags", "prefer_tcp",
                "-rw_timeout", "7000000",
            ]
        cmd += [
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            src,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=8)
            out = res.stdout.strip()
            if "x" in out:
                w, h = out.split("x")
                return int(w), int(h)
        except Exception:
            pass
        return None, None

    def _probe_with_opencv_once(self, src: str) -> Tuple[Optional[int], Optional[int]]:
        """เปิดด้วย OpenCV ชั่วคราว 1 เฟรมเพื่อถามขนาด (เผื่อ ffprobe ใช้ไม่ได้กับบางรุ่น)"""
        try:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                return None, None
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                return w, h
        except Exception:
            pass
        return None, None

    def _drain_stderr(self) -> None:
        """ดูด stderr กันบล็อก และเก็บบรรทัดท้าย ๆ ไว้ดีบัก"""
        proc = self._proc
        if not proc or not proc.stderr:
            return
        try:
            while not self._stop_evt.is_set():
                chunk = proc.stderr.readline()
                if not chunk:
                    break
                try:
                    line = chunk.decode("utf-8", "ignore").strip()
                except Exception:
                    line = str(chunk)
                if line:
                    self._last_stderr.append(line)
        except Exception:
            pass

    # ---------- lifecycle ----------

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
        while not self._stop_evt.is_set():
            print(self.last_ffmpeg_stderr())
            if self.backend == "ffmpeg":
                if self._proc is None or self._proc.poll() is not None:
                    time.sleep(0.05)
                    continue

                # ต้องมี W,H ครบก่อนอ่าน
                if not (self.width and self.height and np is not None):
                    time.sleep(0.03)
                    continue
                frame_size = int(self.width) * int(self.height) * 3

                buffer = bytearray()
                while len(buffer) < frame_size and not self._stop_evt.is_set():
                    chunk = self._proc.stdout.read(frame_size - len(buffer))
                    if not chunk:
                        break
                    buffer.extend(chunk)

                if len(buffer) != frame_size:
                    # ยังไม่ครบเฟรม ทิ้ง
                    time.sleep(0.01)
                    continue

                try:
                    frame = np.frombuffer(memoryview(buffer), np.uint8).reshape(
                        (int(self.height), int(self.width), 3)
                    )
                except Exception:
                    time.sleep(0.01)
                    continue

            else:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.05)
                    continue
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

            # ตัดขาดบัฟเฟอร์เดิม
            try:
                frame_copy = frame.copy()
            except Exception:
                time.sleep(0.005)
                continue

            # เก็บเฟรมล่าสุดลงคิว
            if self._q.full():
                with silent():
                    _ = self._q.get_nowait()
            with silent():
                self._q.put_nowait(frame_copy)

            if self.read_interval > 0:
                time.sleep(self.read_interval)

    async def read(self):
        """อ่านเฟรมล่าสุดแบบ non-blocking"""
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

        if self.backend == "ffmpeg":
            with silent():
                if self._proc is not None:
                    try:
                        # ปิด stdout/stderr ก่อน เพื่อปลดบล็อก thread/reader
                        if self._proc.stdout:
                            self._proc.stdout.close()
                        if self._proc.stderr:
                            self._proc.stderr.close()
                    except Exception:
                        pass
                    try:
                        self._proc.terminate()
                        self._proc.wait(timeout=0.8)
                    except Exception:
                        try:
                            self._proc.kill()
                        except Exception:
                            pass
            self._proc = None

            with silent():
                if self._stderr_thread and self._stderr_thread.is_alive():
                    self._stderr_thread.join(timeout=0.2)
            self._stderr_thread = None

        else:
            with silent():
                if self._cap is not None and self._cap.isOpened():
                    await asyncio.to_thread(self._cap.release)
            await asyncio.sleep(0.05)
            self._cap = None

        # ล้างคิว
        while True:
            try:
                _ = self._q.get_nowait()
            except Empty:
                break

        gc.collect()
        _malloc_trim()

    # -------- diagnostics (ถ้าต้องการ) --------
    def last_ffmpeg_stderr(self) -> str:
        """คืนบรรทัด stderr ล่าสุดของ ffmpeg เพื่อช่วยดีบัก"""
        return "\n".join(self._last_stderr)
