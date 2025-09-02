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
        use_gui: bool = False,
    ):
        """สร้างตัว worker สำหรับอ่านภาพจากกล้อง

        Parameters
        ----------
        src : any
            แหล่งที่มาของวิดีโอสำหรับ ``cv2.VideoCapture``
        loop : asyncio.AbstractEventLoop | None, optional
            event loop ที่จะใช้สำหรับส่งเฟรม, by default None
        width, height : int | None, optional
            ปรับขนาดเฟรมหากกำหนด
        use_gui : bool, optional
            หากเป็น ``True`` จะเรียก ``cv2.destroyAllWindows`` เมื่อหยุด
        """

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
        self.use_gui = use_gui

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

            self.loop.call_soon_threadsafe(put_frame)
            time.sleep(0.04)

    async def read(self):
        return await self.queue.get()

    async def stop(self, timeout: float | None = 1.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, timeout)
        if self.cap.isOpened():
            await asyncio.to_thread(self.cap.release)
        if self.use_gui:
            await asyncio.to_thread(cv2.destroyAllWindows)

    def __del__(self):
        """Ensure resources are released when the worker is garbage collected."""
        try:
            self._stop.set()
            if self._thread.is_alive():
                self._thread.join(timeout=0.1)
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
