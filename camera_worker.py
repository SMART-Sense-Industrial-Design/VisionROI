import cv2
import asyncio
import threading
import time

class CameraWorker:
    """อ่านภาพจากกล้องในเธรดแยกแล้วส่งผ่าน asyncio.Queue"""

    def __init__(
        self,
        src,
        loop: asyncio.AbstractEventLoop | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        self.src = src
        self.loop = loop or asyncio.get_event_loop()
        self.cap = cv2.VideoCapture(src)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self) -> bool:
        if not self.cap.isOpened():
            # ปล่อยทรัพยากรทันทีหากไม่สามารถเปิดกล้องได้
            self.cap.release()
            return False
        if not self._thread.is_alive():
            self._thread.start()
        return True

    def _reader(self) -> None:
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            def put_frame() -> None:
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                self.queue.put_nowait(frame)

            if self.loop.is_closed():
                break
            try:
                self.loop.call_soon_threadsafe(put_frame)
            except RuntimeError:
                break
            time.sleep(0.04)

    async def read(self):
        return await self.queue.get()

    async def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            try:
                await asyncio.to_thread(self._thread.join)
            except RuntimeError:
                self._thread.join()
        if self.cap.isOpened():
            try:
                await asyncio.to_thread(self.cap.release)
            except RuntimeError:
                self.cap.release()
        try:
            await asyncio.to_thread(cv2.destroyAllWindows)
        except RuntimeError:
            cv2.destroyAllWindows()

    def __del__(self):
        """Ensure resources are released when the worker is garbage collected."""
        try:
            self._stop.set()
            if self._thread.is_alive():
                self._thread.join(timeout=0.1)
            if self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
