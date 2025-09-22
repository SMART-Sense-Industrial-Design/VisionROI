# camera_worker.py
import cv2
import time
import threading
import asyncio
import gc
import subprocess
import os
import logging
import select
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
        self._last_stderr = deque(maxlen=200)  # เก็บบรรทัด stderr ล่าสุดไว้ดู error
        self._logger = logging.getLogger("camera_worker")
        self._log_prefix = f"[{backend}:{src}]"

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q: Queue = Queue(maxsize=1)
        self._last_frame = None
        self._fail_count = 0
        self._ffmpeg_cmd = None
        self._stdout_fd: int | None = None
        self._last_returncode_logged: int | None = None
        self._read_timeout = 3.0

        self._logger.info("%s initializing camera worker", self._log_prefix)

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
                ]

            # ช่วยให้วิเคราะห์สตรีมในเครือข่ายที่หน่วง (ออปชันเก่า ๆ ก็มี)
            cmd += ["-probesize", "32M", "-analyzeduration", "5M"]

            cmd += [
                "-fflags", "+discardcorrupt",  # ทิ้งเฟรม/แพ็กเก็ตเสีย
                "-flags", "low_delay",  # ลดการหน่วงของสตรีม
                "-fflags", "nobuffer",  # ไม่สะสมเฟรมในบัฟเฟอร์
                "-i", str(src),
                "-map", "0:v:0",
                "-an",
            ]

            if self.width and self.height:
                cmd += ["-vf", f"scale={int(self.width)}:{int(self.height)}:flags=lanczos"]

            cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

            self._ffmpeg_cmd = cmd
            with silent():
                self._proc = subprocess.Popen(
                    self._ffmpeg_cmd,
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
                if self._proc and self._proc.stdout:
                    try:
                        self._stdout_fd = self._proc.stdout.fileno()
                    except Exception as exc:
                        self._logger.warning(
                            "%s failed to obtain stdout fd: %s", self._log_prefix, exc
                        )

        elif backend == "opencv":
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

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ---------- helpers ----------

    @staticmethod
    def _is_rtsp(src: str) -> bool:
        s = str(src).lower()
        return s.startswith("rtsp://") or s.startswith("rtsps://")

    def _probe_resolution(self, src: str) -> Tuple[Optional[int], Optional[int]]:
        """ใช้ ffprobe หา width/height พร้อม RTSP (ไม่ใช้ rw_timeout/stimeout)"""
        cmd = ["ffprobe", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self._is_rtsp(src):
            cmd += ["-rtsp_transport", "tcp", "-rtsp_flags", "prefer_tcp"]
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

    def _restart_backend(self) -> None:
        """พยายามเชื่อมต่อสตรีมใหม่เมื่ออ่านไม่ได้"""
        self._logger.warning("%s restarting backend after failures", self._log_prefix)
        with silent():
            if self.backend == "ffmpeg":
                if self._proc is not None:
                    try:
                        if self._proc.stdout:
                            self._proc.stdout.close()
                        if self._proc.stderr:
                            self._proc.stderr.close()
                        self._proc.terminate()
                        self._proc.wait(timeout=0.5)
                    except Exception:
                        try:
                            self._proc.kill()
                        except Exception:
                            pass
                if self._ffmpeg_cmd is not None:
                    self._proc = subprocess.Popen(
                        self._ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10**8,
                        preexec_fn=os.setsid,
                    )
                    if self._proc and self._proc.stderr:
                        self._stderr_thread = threading.Thread(
                            target=self._drain_stderr, daemon=True
                        )
                        self._stderr_thread.start()
                    if self._proc and self._proc.stdout:
                        try:
                            self._stdout_fd = self._proc.stdout.fileno()
                        except Exception as exc:
                            self._logger.warning(
                                "%s failed to obtain stdout fd on restart: %s",
                                self._log_prefix,
                                exc,
                            )
            elif self.backend == "opencv":
                if self._cap is not None and self._cap.isOpened():
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                self._cap = cv2.VideoCapture(self.src)
                with silent():
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.width:
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
                    if self.height:
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        self._fail_count = 0

    # ---------- lifecycle ----------

    def start(self) -> bool:
        if self.backend == "ffmpeg":
            if self._proc is None or self._proc.poll() is not None:
                return False
        elif self.backend == "opencv":
            if not self._cap or not self._cap.isOpened():
                return False
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            frame = None
            if self.backend == "ffmpeg":
                if self._proc is None or self._proc.poll() is not None:
                    self._fail_count += 1
                    if self._proc is not None:
                        returncode = self._proc.poll()
                        if (
                            returncode is not None
                            and returncode != self._last_returncode_logged
                        ):
                            self._last_returncode_logged = returncode
                            self._logger.error(
                                "%s ffmpeg exited with code %s; last stderr: %s",
                                self._log_prefix,
                                returncode,
                                self.last_ffmpeg_stderr(),
                            )
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.05)
                    continue

                if not (self.width and self.height and np is not None):
                    time.sleep(0.03)
                    continue
                frame_size = int(self.width) * int(self.height) * 3

                stdout = self._proc.stdout
                if stdout is None:
                    self._fail_count += 1
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.05)
                    continue

                buffer = bytearray()
                start_wait = time.monotonic()
                fd = self._stdout_fd or stdout.fileno()
                while len(buffer) < frame_size and not self._stop_evt.is_set():
                    remaining = frame_size - len(buffer)
                    wait_time = max(0.1, min(0.5, self._read_timeout))
                    try:
                        ready, _, _ = select.select([fd], [], [], wait_time)
                    except Exception as exc:
                        self._logger.warning(
                            "%s select on ffmpeg stdout failed: %s",
                            self._log_prefix,
                            exc,
                        )
                        break
                    if not ready:
                        if time.monotonic() - start_wait > self._read_timeout:
                            self._logger.warning(
                                "%s no video data for %.1fs (collected %d/%d bytes); last stderr: %s",
                                self._log_prefix,
                                self._read_timeout,
                                len(buffer),
                                frame_size,
                                self.last_ffmpeg_stderr(),
                            )
                            break
                        continue
                    try:
                        chunk = os.read(fd, remaining)
                    except BlockingIOError:
                        if time.monotonic() - start_wait > self._read_timeout:
                            self._logger.warning(
                                "%s stdout temporarily unavailable for %.1fs; last stderr: %s",
                                self._log_prefix,
                                self._read_timeout,
                                self.last_ffmpeg_stderr(),
                            )
                            break
                        continue
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    start_wait = time.monotonic()

                if len(buffer) != frame_size:
                    self._fail_count += 1
                    if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                        self._logger.warning(
                            "%s received incomplete frame (%d/%d bytes); consecutive failures=%d",
                            self._log_prefix,
                            len(buffer),
                            frame_size,
                            self._fail_count,
                        )
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.01)
                    continue

                try:
                    frame = np.frombuffer(memoryview(buffer), np.uint8).reshape(
                        (int(self.height), int(self.width), 3)
                    )
                except Exception:
                    self._fail_count += 1
                    if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                        self._logger.warning(
                            "%s failed to reshape frame; consecutive failures=%d",
                            self._log_prefix,
                            self._fail_count,
                        )
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.01)
                    continue

            elif self.backend == "opencv":
                if self._cap is None or not self._cap.isOpened():
                    self._fail_count += 1
                    if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                        self._logger.warning(
                            "%s OpenCV capture not opened; consecutive failures=%d",
                            self._log_prefix,
                            self._fail_count,
                        )
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.05)
                    continue
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._fail_count += 1
                    if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                        self._logger.warning(
                            "%s OpenCV read returned empty frame; consecutive failures=%d",
                            self._log_prefix,
                            self._fail_count,
                        )
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.01)
                    continue
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            self._fail_count = 0
            self._last_returncode_logged = None
            self._last_frame = frame
            frame_copy = frame
            if self._q.full():
                with silent():
                    _ = self._q.get_nowait()
            with silent():
                self._q.put_nowait(frame_copy)

            if self.read_interval > 0:
                time.sleep(self.read_interval)

    async def read(self, timeout: float = 0.1):
        """อ่านเฟรมล่าสุดแบบ non-blocking; คืน ``None`` เมื่อรอเกินกำหนด"""
        try:
            return self._q.get_nowait()
        except Empty:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
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
            self._stdout_fd = None

        elif self.backend == "opencv":
            with silent():
                if self._cap is not None and self._cap.isOpened():
                    await asyncio.to_thread(self._cap.release)
            await asyncio.sleep(0.05)
            self._cap = None
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

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
