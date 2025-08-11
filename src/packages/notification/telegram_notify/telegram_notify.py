import time
import requests
import cv2
from datetime import datetime
import threading
import queue
import os
from requests.adapters import HTTPAdapter, Retry

class TelegramNotify:
    def __init__(self, token=None, chat_id=None):
        self.TG_TOKEN = token
        self.CHAT_ID = chat_id
        self.TG_TIME_INTERVAL = 5  # 🕒 Global default interval (sec)

        self.last_send_time = {
            'text': 0,
            'image': 0,
            'file': 0,
            'frame': 0,
            'video': 0
        }

        self.send_queue = queue.Queue(maxsize=100)
        # 🔚 sentinel object สำหรับหยุด worker thread อย่างปลอดภัย
        self._stop_sentinel = object()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

        # ✅ Session + Retry
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def _safe_telegram_post(self, url, data=None, files=None, timeout=30):
        try:
            response = self.session.post(url, data=data, files=files, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if result.get("ok", False):
                    return True
                else:
                    print(f"❌ Telegram ตอบกลับผิดพลาด: {result}")
            else:
                print(f"❌ HTTP Error: {response.status_code} | {response.text}")
        except requests.exceptions.SSLError as e:
            print(f"❌ SSL Error: {e}")
        except Exception as e:
            print(f"❌ General Exception: {e}")
        return False

    def _process_queue(self):
        while True:
            item = self.send_queue.get()
            if item is self._stop_sentinel:
                self.send_queue.task_done()
                break
            func, args = item
            try:
                func(*args)
            except Exception as e:
                print(f"❌ Error in queue processor: {e}")
            finally:
                self.send_queue.task_done()

    def _should_send(self, key, time_interval=None):
        now = time.time()
        interval = time_interval if time_interval is not None else self.TG_TIME_INTERVAL
        if now - self.last_send_time[key] >= interval:
            self.last_send_time[key] = now
            return True
        return False

    # ========== Actual Send Methods ==========
    def tg_send_text(self, msg):
        url = f"https://api.telegram.org/bot{self.TG_TOKEN}/sendMessage"
        data = {'chat_id': self.CHAT_ID, 'text': msg}
        self._safe_telegram_post(url, data=data)

    def tg_send_image(self, msg, image):
        try:
            ret, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                img_bytes = buffer.tobytes()
                url = f"https://api.telegram.org/bot{self.TG_TOKEN}/sendPhoto"
                files = {'photo': ('image.jpg', img_bytes, 'image/jpeg')}
                data = {'chat_id': self.CHAT_ID, 'caption': msg}
                self._safe_telegram_post(url, files=files, data=data)
        except Exception as e:
            print(f"❌ Failed to send image: {e}")

    def tg_frame_send_file(self, msg, frame):
        try:
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                img_bytes = buffer.tobytes()
                filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                url = f"https://api.telegram.org/bot{self.TG_TOKEN}/sendDocument"
                files = {'document': (filename, img_bytes, 'image/jpeg')}
                data = {'chat_id': self.CHAT_ID, 'caption': msg}
                self._safe_telegram_post(url, files=files, data=data)
        except Exception as e:
            print(f"❌ Failed to send frame: {e}")

    def tg_send_file(self, msg, path_file):
        try:
            if not os.path.exists(path_file):
                print(f"❌ File not found: {path_file}")
                return
            filename = os.path.basename(path_file)
            url = f"https://api.telegram.org/bot{self.TG_TOKEN}/sendDocument"
            with open(path_file, 'rb') as myfile:
                files = {'document': (filename, myfile)}
                data = {'chat_id': self.CHAT_ID, 'caption': msg}
                self._safe_telegram_post(url, files=files, data=data)
        except Exception as e:
            print(f"❌ Failed to send file: {e}")

    def tg_send_video(self, msg, path_file):
        try:
            if not os.path.exists(path_file):
                print(f"❌ Video file not found: {path_file}")
                return
            filename = os.path.basename(path_file)
            url = f"https://api.telegram.org/bot{self.TG_TOKEN}/sendVideo"
            with open(path_file, 'rb') as myfile:
                files = {'video': (filename, myfile)}
                data = {'chat_id': self.CHAT_ID, 'caption': msg}
                self._safe_telegram_post(url, files=files, data=data)
        except Exception as e:
            print(f"❌ Failed to send video: {e}")

    # ========== Public Send Methods ==========
    def start_send_text(self, msg, time_interval=None):
        if self._should_send('text', time_interval):
            try:
                self.send_queue.put((self.tg_send_text, (msg,)), timeout=1)
            except queue.Full:
                print("⚠️ Send queue is full. Skipping text.")

    def start_send_image(self, msg, image, time_interval=None):
        if self._should_send('image', time_interval):
            try:
                self.send_queue.put((self.tg_send_image, (msg, image)), timeout=1)
            except queue.Full:
                print("⚠️ Send queue is full. Skipping image.")

    def start_frame_send_file(self, msg, frame, time_interval=None):
        if self._should_send('frame', time_interval):
            try:
                self.send_queue.put((self.tg_frame_send_file, (msg, frame)), timeout=1)
            except queue.Full:
                print("⚠️ Send queue is full. Skipping frame.")

    def start_send_file(self, msg, path_file, time_interval=None):
        if self._should_send('file', time_interval):
            try:
                self.send_queue.put((self.tg_send_file, (msg, path_file)), timeout=1)
            except queue.Full:
                print("⚠️ Send queue is full. Skipping file.")

    def start_send_video(self, msg, path_file, time_interval=None):
        if self._should_send('video', time_interval):
            try:
                self.send_queue.put((self.tg_send_video, (msg, path_file)), timeout=1)
            except queue.Full:
                print("⚠️ Send queue is full. Skipping video.")

    def close(self):
        """ส่งสัญญาณหยุดและรอ thread ก่อนปิด session"""
        # ส่ง sentinel เพื่อหยุด worker thread
        self.send_queue.put(self._stop_sentinel)
        # รอให้ thread ทำงานเสร็จ
        self.worker_thread.join()
        self.session.close()
