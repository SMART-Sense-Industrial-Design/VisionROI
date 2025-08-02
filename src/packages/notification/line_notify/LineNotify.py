
import time
import requests
import cv2
from datetime import datetime
import threading
# from concurrent.futures import ThreadPoolExecutor
import requests



class LineNotify:

    shared_session = requests.Session()

    def __init__(self, token= None, reciever_id= None):
        cfg_line = AppConfig().load_toml_config()['line_notify']
        self.START_TIME = None
        self.LINE_API = cfg_line['api']
        self.LINE_TOKEN = token or cfg_line['token']
        self.RECIEVER_ID = reciever_id
        self.LINE_TIME_INTERVAL = cfg_line['time_interval_sec']
        self.LINE_CLASS_INTEREST = cfg_line['class_interest']
        self.LINE_MAX_VALUE = cfg_line['max_value']
        self.LINE_STATUS_RUNNING = cfg_line['status_running']
        self.session = LineNotify.shared_session

    def send_line_notify_text(self, msg):
        
        url = self.LINE_API
        # data = {'message': msg}
        data = {"to": self.RECIEVER_ID, "messages":[{
            "type":"text",
            "text":msg
        }]}
        headers = {'Authorization':'Bearer ' + self.LINE_TOKEN, 'Content-Type':'application/json'}
        session_post = self.session.post(url, headers=headers, json=data)
        # print(session_post.text)

    def send_line_notify_image(self, msg, image):
        
        url = self.LINE_API
        try:
            ret, img_buf_arr = cv2.imencode(".jpg", image)
            if ret:
                image = img_buf_arr.tobytes()
                img = {'imageFile': image}
                data = {'message': msg}
                headers = {'Authorization':'Bearer ' + self.LINE_TOKEN}
                session_post = self.session.post(url, headers=headers, files=img, data =data)
                # print(session_post.text)
        except Exception:
            pass

    def start_line_notify_image(self, dictData, image, time_interval_sec= None):
        if time_interval_sec != None:
            self.LINE_TIME_INTERVAL = time_interval_sec
        current_time = datetime.fromtimestamp(time.time())
  
        if not self.START_TIME or int((current_time - self.START_TIME).total_seconds()) > self.LINE_TIME_INTERVAL:
            msg_alert = f"{dictData}"
            self.START_TIME = datetime.fromtimestamp(time.time())
            threading.Thread(target= self.send_line_notify_image, args= (msg_alert, image,)).start()
    
    def start_line_notify_text(self, msg, time_interval_sec= None):
        if time_interval_sec != None:
            self.LINE_TIME_INTERVAL = time_interval_sec
        current_time = datetime.fromtimestamp(time.time())
    
        if not self.START_TIME or int((current_time - self.START_TIME).total_seconds()) > self.LINE_TIME_INTERVAL:
            self.START_TIME = datetime.fromtimestamp(time.time())
            threading.Thread(target= self.send_line_notify_text, args= (msg,)).start()


        
