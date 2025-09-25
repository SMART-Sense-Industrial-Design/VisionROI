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
import re
from pathlib import Path
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
    backend='ffmpeg' : ใช้ ffmpeg ถ่ม bgr24/rawvideo ผ่าน stdout (ปรับให้ robust กับ RTSP)
    """

    def __init__(
        self,
        src,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        read_interval: float = 0.0,
        backend: str = "opencv",
        # === ตัวเลือก robust ===
        robust: bool = True,        # เปิด watchdog + fallback transport
        low_latency: bool = False,  # โหมดหน่วงต่ำ (ยอม jitter/เฟรมเสียได้มากขึ้น)
        loglevel: str = "error",    # ลดสแปม log จาก ffmpeg
    ) -> None:
        self.src = src
        self.loop = loop
        self.width = width
        self.height = height
        self.read_interval = read_interval
        self.backend = backend

        self.robust = robust
        self.low_latency = low_latency
        self._loglevel = loglevel

        self._cap = None
        self._proc: subprocess.Popen | None = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._last_stderr = deque(maxlen=200)  # เก็บบรรทัด stderr ล่าสุดไว้ดีบัก
        self._logger = logging.getLogger("camera_worker")
        self._log_prefix = f"[{backend}:{src}]"

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q: Queue = Queue(maxsize=1)
        self._last_frame = None
        self._fail_count = 0
        self._ffmpeg_cmd = None
        self._ffmpeg_pix_fmt: str | None = None
        self._stdout_fd: int | None = None
        self._last_returncode_logged: int | None = None
        self._read_timeout = 4.0  # ทน jitter/collection ช้าบ้าง
        self._next_resolution_probe = 0.0
        self._image_path: Path | None = None
        self._image_frame = None
        now = time.monotonic()
        self._last_frame_at = now
        self._last_restart_at: float | None = None
        self._stall_restart_secs = max(6.0, self._read_timeout * 2.5)
        self._min_restart_interval = 1.5

        # robust state
        self._err_window = deque(maxlen=300)
        self._err_window_secs = 5.0
        self._err_threshold = 25
        self._last_err_prune = 0.0
        self._restart_backoff = 0.0  # exponential backoff ก่อน restart ffmpeg

        # วน fallback: transport (tcp <-> udp) ถ้า build รองรับ
        self._rtsp_transport_cycle = ["tcp", "udp"] if robust else ["tcp"]
        self._rtsp_transport_idx = 0

        # cache capability probes (ตรวจครั้งเดียว)
        self._ff_caps_checked = False
        self._ff_rtsp_opts: set[str] = set()

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

            # probe ความสามารถของ ffmpeg ก่อนสร้างคำสั่ง
            self._probe_ffmpeg_caps()

            cmd = self._build_ffmpeg_cmd(str(src))
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
            self._image_path = self._maybe_image_path(src)
            if self._image_path is not None:
                self._image_frame = cv2.imread(str(self._image_path), cv2.IMREAD_COLOR)
                if self._image_frame is None:
                    self._logger.error(
                        "%s unable to read image file %s", self._log_prefix, self._image_path
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

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _maybe_image_path(self, src) -> Path | None:
        if isinstance(src, (str, os.PathLike)):
            path = Path(src)
            if path.is_file() and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            }:
                return path
        return None

    # ---------- ffmpeg capabilities probe ----------

    def _probe_ffmpeg_caps(self) -> None:
        """เช็คว่า ffmpeg build นี้รองรับ RTSP options อะไรบ้าง (ครั้งเดียว)"""
        if self._ff_caps_checked:
            return
        self._ff_caps_checked = True

        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-h", "protocol=rtsp"],
                capture_output=True, text=True, check=False
            )
            text = (out.stdout or "") + (out.stderr or "")
            for opt in ("stimeout", "rw_timeout", "timeout", "rtsp_flags", "rtsp_transport"):
                if f" {opt} " in text or f"\n{opt} " in text or f"{opt}=" in text:
                    self._ff_rtsp_opts.add(opt)
        except Exception:
            pass

        self._logger.info(
            "%s ffmpeg caps: rtsp_opts=%s",
            self._log_prefix, sorted(self._ff_rtsp_opts)
        )

    # ---------- ffmpeg command builder ----------

    def _build_ffmpeg_cmd(self, src: str) -> list[str]:
        """
        สร้างคำสั่ง ffmpeg แบบ robust สำหรับ RTSP (ทุกกล้อง):
        - บังคับ TCP ก่อน ถ้าแตกเยอะจะ fallback ไป UDP อัตโนมัติ (ถ้า build รองรับ)
        - genpts/max_delay เพื่อทน jitter
        - timeout (เฉพาะที่รองรับจริง) ป้องกันแฮงก์
        - ไม่ตั้ง -c:v ใด ๆ (กันไปชนเอาต์พุต)
        """
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", self._loglevel, "-nostdin"]

        transport = self._rtsp_transport_cycle[self._rtsp_transport_idx]
        if self._is_rtsp(src):
            # transport
            if "rtsp_transport" in self._ff_rtsp_opts:
                cmd += ["-rtsp_transport", transport]
            # flags
            if "rtsp_flags" in self._ff_rtsp_opts:
                cmd += ["-rtsp_flags", "prefer_tcp" if transport == "tcp" else "none"]
            # timeouts (เฉพาะที่มีจริงใน build นี้)
            if "rw_timeout" in self._ff_rtsp_opts:
                cmd += ["-rw_timeout", "5000000"]  # 5s
            elif "stimeout" in self._ff_rtsp_opts:
                cmd += ["-stimeout", "5000000"]    # 5s
            elif "timeout" in self._ff_rtsp_opts:
                cmd += ["-timeout", "5000000"]

        # วิเคราะห์สตรีมพอประมาณ
        cmd += ["-probesize", "16M", "-analyzeduration", "2M"]

        # โปรไฟล์ robust/low_latency
        if self.low_latency:
            cmd += ["-fflags", "+discardcorrupt", "-flags", "low_delay", "-fflags", "nobuffer"]
        else:
            cmd += ["-fflags", "+discardcorrupt+genpts"]
            cmd += ["-max_delay", "500000"]  # 500ms
            cmd += ["-use_wallclock_as_timestamps", "1"]

        # *** สำคัญ: input option ต้องมาก่อน -i ***
        cmd += ["-thread_queue_size", "512"]

        # เปิด input
        cmd += ["-i", src, "-map", "0:v:0", "-an"]

        # สเกล/สี (ให้ ffmpeg แปลงเป็น bgr24 ออกมา)
        filters: list[str] = []
        if self.width and self.height:
            filters.append(f"scale={int(self.width)}:{int(self.height)}:flags=lanczos")
        filters.append("setsar=1")
        filters.append("format=bgr24")
        if filters:
            cmd += ["-vf", ",".join(filters)]

        # เอาต์พุตเป็น rawvideo/bgr24 ผ่าน stdout
        cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

        self._ffmpeg_pix_fmt = "bgr24"
        self._logger.info("%s ffmpeg cmd: %s", self._log_prefix, " ".join(cmd))
        return cmd

    # ---------- helpers ----------

    @staticmethod
    def _is_rtsp(src: str) -> bool:
        s = str(src).lower()
        return s.startswith("rtsp://") or s.startswith("rtsps://")

    def _probe_resolution(self, src: str) -> Tuple[Optional[int], Optional[int]]:
        """ใช้ ffprobe หา width/height พร้อม RTSP (ไม่ใช้ timeout option)"""
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

    def _maybe_update_resolution(self) -> bool:
        """พยายามหาความกว้าง/สูงแบบขี้เกียจในระหว่างรัน (ffmpeg เท่านั้น)."""
        if self.width and self.height:
            return True

        if self._maybe_parse_resolution_from_logs():
            return True

        now = time.monotonic()
        if now < self._next_resolution_probe:
            return False

        self._next_resolution_probe = now + 15.0

        src = str(self.src)
        w, h = self._probe_resolution(src)
        if not (w and h):
            w, h = self._probe_with_opencv_once(src)
        if w and h:
            self.width, self.height = w, h
            self._logger.info(
                "%s inferred resolution %sx%s during runtime", self._log_prefix, w, h
            )
            return True
        return False

    def _maybe_parse_resolution_from_logs(self) -> bool:
        if self.width and self.height:
            return True
        if not self._last_stderr:
            return False

        pattern = re.compile(r"(\d{2,5})x(\d{2,5})")
        for line in reversed(self._last_stderr):
            if "x" not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            try:
                w = int(match.group(1))
                h = int(match.group(2))
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            if w < 32 or h < 32:
                continue
            if w > 16384 or h > 16384:
                continue
            self.width, self.height = w, h
            self._logger.info(
                "%s parsed resolution %sx%s from ffmpeg stderr", self._log_prefix, w, h
            )
            return True
        return False

    def _drain_stderr(self) -> None:
        """ดูด stderr กันบล็อก และเก็บบรรทัดท้าย ๆ ไว้ดีบัก + ป้อนให้ watchdog"""
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
                    self._track_ffmpeg_errors(line)
        except Exception:
            pass

    def _flush_partial_ffmpeg_frame(self, fd: int, remaining: int) -> bool:
        """พยายามอ่านและทิ้ง byte ที่เหลือของเฟรมที่ไม่ครบเพื่อรักษาการจัดแนว"""
        if remaining <= 0:
            return True
        deadline = time.monotonic() + max(0.2, min(self._read_timeout, 1.5))
        while remaining > 0 and not self._stop_evt.is_set():
            timeout = min(0.2, max(0.05, self._read_timeout / 2))
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
            except Exception:
                return False
            if not ready:
                if time.monotonic() > deadline:
                    return False
                continue
            try:
                chunk = os.read(fd, min(remaining, 65536))
            except BlockingIOError:
                if time.monotonic() > deadline:
                    return False
                continue
            except Exception:
                return False
            if not chunk:
                return False
            remaining -= len(chunk)
        return remaining <= 0

    def _maybe_restart_on_stall(self) -> bool:
        """ตรวจจับกรณีไม่มีเฟรมสำเร็จเป็นเวลานานแล้วบังคับ restart"""
        now = time.monotonic()
        elapsed = now - self._last_frame_at if self._last_frame_at is not None else float("inf")
        if elapsed < self._stall_restart_secs:
            return False
        if self._last_restart_at is not None and now - self._last_restart_at < self._min_restart_interval:
            return False
        self._logger.warning(
            "%s no successful frame for %.1fs; forcing backend restart",
            self._log_prefix,
            elapsed,
        )
        self._restart_backend()
        return True

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
                self._stdout_fd = None
                if self._ffmpeg_cmd is not None:
                    # exponential backoff (สูงสุด ~2s) ช่วยหลบช่วง jitter หนัก
                    if self._restart_backoff > 0:
                        time.sleep(min(self._restart_backoff, 2.0))
                    self._restart_backoff = min(self._restart_backoff * 2 + 0.1, 2.0) if self._restart_backoff else 0.2
                    # สร้างคำสั่งใหม่ (เผื่อมีการสลับ transport)
                    self._ffmpeg_cmd = self._build_ffmpeg_cmd(str(self.src))
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
                if self._image_path is not None:
                    self._image_frame = cv2.imread(str(self._image_path), cv2.IMREAD_COLOR)
                    if self._image_frame is None:
                        self._logger.error(
                            "%s unable to read image file %s on restart",
                            self._log_prefix,
                            self._image_path,
                        )
                else:
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
        self._err_window.clear()
        self._last_returncode_logged = None
        self._next_resolution_probe = 0.0
        self._last_restart_at = time.monotonic()

    # ---------- lifecycle ----------

    def start(self) -> bool:
        if self.backend not in {"ffmpeg", "opencv"}:
            raise ValueError(f"Unknown backend: {self.backend}")

        def _backend_ready() -> bool:
            if self.backend == "ffmpeg":
                return self._proc is not None and self._proc.poll() is None
            if self._image_path is not None:
                return self._image_frame is not None
            return bool(self._cap and self._cap.isOpened())

        if not _backend_ready():
            self._logger.warning(
                "%s backend not ready during start(); attempting single restart",
                self._log_prefix,
            )
            self._restart_backend()

        if not _backend_ready():
            self._logger.error(
                "%s unable to start backend; last stderr: %s",
                self._log_prefix,
                self.last_ffmpeg_stderr() if self.backend == "ffmpeg" else "",
            )
            return False

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
                    if self._maybe_restart_on_stall():
                        time.sleep(0.1)
                        continue
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.05)
                    continue

                if np is None:
                    time.sleep(0.03)
                    continue

                if not (self.width and self.height):
                    if not self._maybe_update_resolution():
                        self._fail_count += 1
                        if self._fail_count in (1, 25) or self._fail_count % 50 == 0:
                            self._logger.warning(
                                "%s awaiting resolution discovery; consecutive failures=%d; last stderr: %s",
                                self._log_prefix,
                                self._fail_count,
                                self.last_ffmpeg_stderr(),
                            )
                        if self._maybe_restart_on_stall():
                            time.sleep(0.1)
                            continue
                        time.sleep(0.05)
                        continue

                if not (self.width and self.height):
                    time.sleep(0.03)
                    continue
                frame_size = self._expected_frame_size()
                if frame_size is None:
                    self._fail_count += 1
                    if self._fail_count > 10 and self._fail_count % 10 == 0:
                        self._logger.warning(
                            "%s unable to determine frame size for pix_fmt=%s; consecutive failures=%d",
                            self._log_prefix,
                            self._ffmpeg_pix_fmt,
                            self._fail_count,
                        )
                    if self._maybe_restart_on_stall():
                        time.sleep(0.1)
                        continue
                    time.sleep(0.05)
                    continue

                stdout = self._proc.stdout
                if stdout is None:
                    self._fail_count += 1
                    if self._maybe_restart_on_stall():
                        time.sleep(0.1)
                        continue
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.05)
                    continue

                buffer = bytearray()
                start_wait = time.monotonic()
                fd = self._stdout_fd or stdout.fileno()
                read_failed = False
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
                    except OSError as exc:
                        self._logger.warning(
                            "%s read from ffmpeg stdout failed: %s; restarting backend",
                            self._log_prefix,
                            exc,
                        )
                        self._restart_backend()
                        time.sleep(0.1)
                        read_failed = True
                        break
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    start_wait = time.monotonic()

                if len(buffer) != frame_size:
                    if not read_failed and 0 < len(buffer) < frame_size:
                        leftover = frame_size - len(buffer)
                        if not self._flush_partial_ffmpeg_frame(fd, leftover):
                            self._logger.warning(
                                "%s unable to flush %d bytes of partial frame; restarting backend",
                                self._log_prefix,
                                leftover,
                            )
                            self._restart_backend()
                            time.sleep(0.1)
                            continue
                    self._fail_count += 1
                    if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                        self._logger.warning(
                            "%s received incomplete frame (%d/%d bytes); consecutive failures=%d",
                            self._log_prefix,
                            len(buffer),
                            frame_size,
                            self._fail_count,
                        )
                    if self._maybe_restart_on_stall():
                        time.sleep(0.1)
                        continue
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.01)
                    continue

                frame = None
                try:
                    frame = self._reshape_ffmpeg_frame(buffer)
                except Exception:
                    self._fail_count += 1
                    if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                        self._logger.warning(
                            "%s failed to reshape frame; consecutive failures=%d",
                            self._log_prefix,
                            self._fail_count,
                        )
                    if self._maybe_restart_on_stall():
                        time.sleep(0.1)
                        continue
                    if self._fail_count > 100:
                        self._restart_backend()
                    time.sleep(0.01)
                    continue

            elif self.backend == "opencv":
                if self._image_path is not None:
                    if self._image_frame is None:
                        self._fail_count += 1
                        if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                            self._logger.warning(
                                "%s unable to read still image; consecutive failures=%d",
                                self._log_prefix,
                                self._fail_count,
                            )
                        if self._maybe_restart_on_stall():
                            time.sleep(0.1)
                            continue
                        if self._fail_count > 100:
                            self._restart_backend()
                        time.sleep(0.2)
                        continue
                    frame = self._image_frame.copy()
                else:
                    if self._cap is None or not self._cap.isOpened():
                        self._fail_count += 1
                        if self._fail_count in (1, 10) or self._fail_count % 25 == 0:
                            self._logger.warning(
                                "%s OpenCV capture not opened; consecutive failures=%d",
                                self._log_prefix,
                                self._fail_count,
                            )
                        if self._maybe_restart_on_stall():
                            time.sleep(0.1)
                            continue
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
                        if self._maybe_restart_on_stall():
                            time.sleep(0.1)
                            continue
                        if self._fail_count > 100:
                            self._restart_backend()
                        time.sleep(0.01)
                        continue
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            self._fail_count = 0
            self._last_returncode_logged = None
            self._restart_backoff = 0.0
            self._last_frame = frame
            self._last_frame_at = time.monotonic()
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

    def _expected_frame_size(self) -> Optional[int]:
        if not (self.width and self.height):
            return None
        if self._ffmpeg_pix_fmt == "bgr24":
            return int(self.width) * int(self.height) * 3
        return None

    def _reshape_ffmpeg_frame(self, buffer: bytearray):
        if not (self.width and self.height and np is not None):
            raise RuntimeError("width/height or numpy missing")
        if self._ffmpeg_pix_fmt == "bgr24":
            return np.frombuffer(memoryview(buffer), np.uint8).reshape(
                (int(self.height), int(self.width), 3)
            )
        raise RuntimeError(f"unsupported pix_fmt {self._ffmpeg_pix_fmt}")

    # -------- error watchdog / adaptive fallback --------
    def _track_ffmpeg_errors(self, line: str) -> None:
        """
        จับแพตเทิร์น error จาก ffmpeg; ถ้ามี burst ภายในหน้าต่างสั้น ๆ:
            - รีสตาร์ท
            - และถ้า robust เปิดอยู่ ให้ 'สลับ transport' (tcp<->udp) ถ้าบิลด์รองรับ
        """
        now = time.monotonic()
        bad = (
            "corrupt decoded frame" in line
            or "error while decoding" in line
            or "bad cseq" in line
            or "Invalid NAL" in line
            or "non-existing PPS" in line
            or "max delay reached" in line
        )
        if bad:
            self._err_window.append(now)
        # prune window + action
        if now - self._last_err_prune > 0.5:
            self._last_err_prune = now
            while self._err_window and (now - self._err_window[0]) > self._err_window_secs:
                self._err_window.popleft()
            if len(self._err_window) >= self._err_threshold:
                self._logger.warning(
                    "%s burst errors=%d within %.1fs; last stderr: %s",
                    self._log_prefix, len(self._err_window), self._err_window_secs, self.last_ffmpeg_stderr()
                )
                if self.robust and "rtsp_transport" in self._ff_rtsp_opts and len(self._rtsp_transport_cycle) > 1:
                    next_idx = (self._rtsp_transport_idx + 1) % len(self._rtsp_transport_cycle)
                    if next_idx != self._rtsp_transport_idx:
                        self._rtsp_transport_idx = next_idx
                        self._logger.warning(
                            "%s switching RTSP transport -> %s",
                            self._log_prefix, self._rtsp_transport_cycle[self._rtsp_transport_idx]
                        )
                self._restart_backend()
                self._err_window.clear()
