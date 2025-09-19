# VisionROI

โปรเจ็กต์ **VisionROI** เป็นแอป Quart สำหรับทดสอบ OCR และการจัดการ ROI

## สารบัญ
- [การเริ่มต้นอย่างรวดเร็ว](#การเริ่มต้นอย่างรวดเร็ว)
- [ฟีเจอร์](#ฟีเจอร์)
- [โครงสร้างโปรเจ็กต์](#โครงสร้างโปรเจ็กต์)
- [Backend การสตรีม](#backend-การสตรีม)
- [ข้อกำหนดระบบ](#ข้อกำหนดระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [การรันโปรเจ็กต์](#การรันโปรเจ็กต์)
- [ตัวอย่าง: ทดสอบด้วยไฟล์ภาพนิ่ง (`example.jpg`)](#ตัวอย่าง-ทดสอบด้วยไฟล์ภาพนิ่ง-examplejpg)
- [โฟลว์การทำงานจากการสร้าง Source ถึงการรัน Inference Group](#โฟลว์การทำงานจากการสร้าง-source-ถึงการรัน-inference-group)
- [การตั้งค่าการแจ้งเตือน](#การตั้งค่าการแจ้งเตือน)
- [หน้าต่างต่าง ๆ](#หน้าต่างต่าง-ๆ)
- [API/Endpoints](#apiendpoints)
- [การพัฒนาและการทดสอบ](#การพัฒนาและการทดสอบ)
- [การกู้คืนงานอัตโนมัติ](#การกู้คืนงานอัตโนมัติ)
- [โครงสร้าง `data_sources/`](#โครงสร้าง-data_sources)
- [ข้อมูลเพิ่มเติม](#ข้อมูลเพิ่มเติม)
- [การร่วมพัฒนา](#การร่วมพัฒนา)
- [การทดสอบ](#การทดสอบ)
- [License](#license)

## การเริ่มต้นอย่างรวดเร็ว

1. สร้าง virtual environment แล้ว activate:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # บน Windows ใช้ .venv\\Scripts\\activate
   ```

2. ติดตั้งแพ็กเกจของโปรเจ็กต์และ dependencies หลัก (ถ้าต้องการรันผ่าน **uvicorn** หรือใช้งานโมเดลเพิ่มเติม ให้ติดตั้ง extras ตามความต้องการ):

   ```bash
   pip install "."
   pip install ".[server]"      # (ตัวเลือก) ติดตั้ง uvicorn + websockets
   pip install ".[onnx]"        # (ตัวเลือก) ใช้งานโมเดล ONNX
   pip install ".[torch]"       # (ตัวเลือก) ใช้งานโมเดลที่พึ่งพา PyTorch/Transformers
   pip install ".[tflite]"      # (ตัวเลือก) ใช้งานโมเดล TensorFlow Lite
   pip install ".[tensorrt]"    # (ตัวเลือก) ใช้งาน TensorRT backend
   ```

   > หากยังไม่ได้ติดตั้งไลบรารีระบบ เช่น `tesseract` หรือแพ็กเกจที่เกี่ยวข้อง โปรดดูหัวข้อ [ติดตั้ง System Dependencies ที่จำเป็น](#ติดตั้ง-system-dependencies-ที่จำเป็น)

3. ตรวจสอบให้แน่ใจว่าเครื่องมีคำสั่ง `ffmpeg` (จำเป็นเมื่อเลือก backend แบบ `ffmpeg`).
   บน Ubuntu สามารถติดตั้งได้ด้วย `sudo apt install ffmpeg` ส่วน Windows สามารถดาวน์โหลดจาก [FFmpeg.org](https://ffmpeg.org) แล้วเพิ่มลงใน PATH.
   สามารถตรวจสอบได้ด้วยคำสั่ง `ffmpeg -version` หากใช้เฉพาะ backend `opencv` สามารถข้ามขั้นตอนนี้ได้

4. รันแอปพลิเคชัน (ค่าเริ่มต้นใช้พอร์ต `5000` หากต้องการกำหนดพอร์ตเองให้เพิ่ม `--port`).
   หากต้องการใช้เซิร์ฟเวอร์ **uvicorn** (แนะนำสำหรับการใช้งานจริง) ให้เพิ่ม `--use-uvicorn`:

   ```bash
   python app.py                       # ใช้เซิร์ฟเวอร์ภายใน
   python app.py --port 12000          # ระบุพอร์ตเอง
   python app.py --use-uvicorn         # รันด้วย uvicorn
   ```

5. เปิดเบราว์เซอร์ไปที่ `http://localhost:5000/` หรือ `/home` (หรือพอร์ตที่กำหนด)

### พารามิเตอร์บรรทัดคำสั่งของ `app.py`
- `--port <int>` – กำหนดพอร์ตที่เว็บเซิร์ฟเวอร์จะรับคำขอ (ค่าเริ่มต้น 5000)
- `--use-uvicorn` – สลับไปใช้เซิร์ฟเวอร์ `uvicorn` พร้อมค่าตั้งต้นที่เหมาะสำหรับการใช้งานจริง เช่น `timeout_keep_alive=2`

ค่าดีฟอลต์ของแอปจะรับคำขอบน `0.0.0.0` เพื่อให้เข้าถึงได้จากเครื่องอื่นในเครือข่ายเดียวกัน หากต้องการผูกกับอินเทอร์เฟซเฉพาะสามารถรันผ่าน `uvicorn` และเพิ่มอาร์กิวเมนต์ `--host` เอง:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

## ฟีเจอร์
- เลือกและบันทึกตำแหน่ง ROI จากกล้องหรือวิดีโอ
- รันโมเดลเพื่อตรวจจับข้อความหรือวัตถุใน ROI ตามกลุ่มที่เลือก (Inference Group)
- ตรวจจับหน้ากระดาษและ ROI แบบเพจผ่านหน้า Inference Page (`/inference_page`)
- รองรับการแจ้งเตือนผ่าน Telegram
- มาพร้อมโมดูลตัวอย่าง `typhoon_ocr`, `yolo`, `easy_ocr`, `rapid_ocr`, `trocr` และ `tesseract_ocr` เพื่อเริ่มต้นทดลองใช้งาน
- มี `CameraWorker` สำหรับดึงเฟรมจากสตรีมวิดีโอทั้งผ่าน OpenCV และ `ffmpeg`
- รวมแพ็กเกจโมเดลเสริมใน `src/packages/models` เช่น `rtdetr` และ `yolov8`
- บันทึกสถานะกล้องและงาน inference ลงไฟล์ `service_state.json` พร้อมข้อมูล `source`, ความละเอียด, backend, group, ค่า `interval` และสถานะการรัน เพื่อให้ `restore_service_state()` เริ่มงานเดิมให้อัตโนมัติหลังรีสตาร์ทเซิร์ฟเวอร์
- มี endpoint `GET /_healthz` สำหรับตรวจสอบสถานะ และ `POST /_quit` สำหรับสั่งปิดเซิร์ฟเวอร์อย่างนุ่มนวล

### การกู้คืนงานอัตโนมัติ

ทุกครั้งที่มีการเริ่มหรือหยุดงาน ROI หรือ inference ระบบจะบันทึกสถานะล่าสุดไว้ในไฟล์ `service_state.json`
ซึ่งรวมถึงชื่อ source, ความละเอียด, backend, ค่า `interval`, กลุ่ม ROI ที่กำลังประมวลผล และสถานะการรัน
เมื่อรีสตาร์ทแอป ฟังก์ชัน `restore_service_state()` ใน `app.py` จะโหลดไฟล์ดังกล่าวเพื่อเริ่มงานเดิมให้อัตโนมัติ
หากไม่ต้องการให้ระบบกู้คืนงาน สามารถลบไฟล์ `service_state.json` ก่อนสตาร์ทแอปใหม่ได้

## โครงสร้างโปรเจ็กต์

```
VisionROI/
├── app.py                # แอปหลัก
├── camera_worker.py      # worker จัดการสตรีมวิดีโอ
├── data_sources/         # คอนฟิกและไฟล์ ROI ของแต่ละ source
├── inference_modules/    # โมดูล inference เช่น typhoon_ocr, yolo, easy_ocr, rapid_ocr, trocr, tesseract_ocr
├── src/
│   └── packages/
│       ├── models/       # โมเดลเสริม เช่น rtdetr, yolov8
│       └── notification/ # ฟังก์ชันแจ้งเตือน
├── static/               # ไฟล์ front-end
├── templates/            # เทมเพลต HTML
└── tests/                # เทสต์ด้วย pytest
```

## Backend การสตรีม

ระบบรองรับ backend การอ่านสตรีมวิดีโอ 2 แบบหลัก ๆ:

- `opencv` – ใช้ `cv2.VideoCapture` เหมาะสำหรับการทดสอบทั่วไปและอุปกรณ์ที่รองรับ OpenCV โดยตรง
- `ffmpeg` – ใช้คำสั่ง `ffmpeg` เพื่อส่งเฟรมแบบ `bgr24` ผ่าน `stdout` ให้ความเสถียรกับสตรีม RTSP และสามารถปรับขนาดภาพล่วงหน้าได้ (จำเป็นต้องติดตั้งคำสั่ง `ffmpeg` ให้เรียกได้จาก PATH)

สามารถกำหนด backend ได้จากคีย์ `stream_type` ในไฟล์ `data_sources/<name>/config.json`, ผ่านหน้าเว็บ `/create_source` หรือ API `PATCH /update_stream_type/<name>` หากไม่ระบุจะใช้ `opencv` เป็นค่าเริ่มต้น

## ข้อกำหนดระบบ
- Python ≥3.10
- dependencies ถูกกำหนดใน `pyproject.toml`

### Dependencies หลัก
- `Quart`
- `opencv-python`
- `numpy`
- `Pillow`
- `easyocr`
- `rapidocr`
- `pytesseract`
- `paho-mqtt`
- `requests`

### Dependencies เพิ่มเติม (Extras)
- `server` – รวม `uvicorn`, `websockets` สำหรับการรันเซิร์ฟเวอร์ผ่าน Uvicorn และรองรับการจัดการ WebSocket client
- `onnx` – รวม `onnx`, `onnxruntime` สำหรับโหลดและรันโมเดล ONNX
- `torch` – รวม `torch`, `torchvision`, `transformers` สำหรับโมเดลที่อาศัย PyTorch หรือ Hugging Face
- `tflite` – รวม `tensorflow` สำหรับโมเดล TensorFlow Lite
- `tensorrt` – รวม `tensorrt`, `pycuda`, `torch`, `torchvision`, `onnx` เพื่อใช้งาน TensorRT backend (รวมส่วนพึ่งพาที่จำเป็น)

## การติดตั้ง
แนะนำให้สร้าง virtual environment ก่อน:

```bash
python -m venv .venv
```

หลังจากสร้างแล้ว ให้ activate environment:

- **Linux/macOS**

  ```bash
  source .venv/bin/activate
  ```

- **Windows**

  ```powershell
  .venv\\Scripts\\activate
  ```

เมื่อ environment ถูก activate แล้วให้ติดตั้งแพ็กเกจหลัก และเลือกเสริมส่วนที่ต้องใช้:

```bash
pip install "."              # ติดตั้งแพ็กเกจหลัก
pip install ".[server]"      # (ทางเลือก) ใช้ uvicorn + websockets
pip install ".[onnx,torch]"  # (ตัวอย่าง) ติดตั้ง extras หลายชุดพร้อมกัน
```

หากต้องการติดตั้งแบบ editable เพื่อแก้ไขซอร์สโค้ดโดยตรง:

```bash
pip install -e "."
```

### ติดตั้ง System Dependencies ที่จำเป็น
แพ็กเกจบางตัวต้องพึ่งพาไลบรารีระดับระบบปฏิบัติการ โดยเฉพาะเมื่อต้องใช้งานโมดูล OCR (`pytesseract`) และการประมวลผลภาพผ่าน OpenCV:

- **Ubuntu / Debian**

  ```bash
  sudo apt update
  sudo apt install ffmpeg tesseract-ocr libtesseract-dev libgl1 libglib2.0-0
  ```

- **macOS (ผ่าน Homebrew)**

  ```bash
  brew install ffmpeg tesseract
  ```

- **Windows**

  1. ดาวน์โหลดและติดตั้ง [FFmpeg](https://ffmpeg.org/download.html) แล้วเพิ่มลงใน `PATH`.
  2. ดาวน์โหลดและติดตั้ง [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) แล้วเพิ่มโฟลเดอร์ที่มี `tesseract.exe` ลงใน `PATH`.

หลังจากติดตั้งระบบเรียบร้อยสามารถตรวจสอบว่า `ffmpeg` และ `tesseract` ใช้งานได้ด้วยคำสั่ง `ffmpeg -version` และ `tesseract --version` เพื่อยืนยันว่า backend `ffmpeg` และโมดูล `pytesseract` พร้อมใช้งาน

### ตัวแปรสภาพแวดล้อมที่สำคัญ
- `INFERENCE_RESULT_TIMEOUT` – กำหนดเวลารอผลลัพธ์จากโมดูล inference ต่อ ROI หน่วยเป็นวินาที (ค่าเริ่มต้น `10.0`). หากต้องการลดความถี่ในการตัดการเชื่อมต่อของ WebSocket ในงานที่ใช้เวลาประมวลผลยาว ควรกำหนดค่าให้อยู่เหนือระยะเวลาประมวลผลสูงสุดของแต่ละโมดูล


## ไฟล์และโฟลเดอร์สำคัญ
ตารางต่อไปนี้สรุปหน้าที่ของไฟล์และโฟลเดอร์หลักในโปรเจ็กต์เพื่อให้ง่ายต่อการอ้างอิงขณะพัฒนา:

| เส้นทาง | รายละเอียด |
| --- | --- |
| `app.py` | จุดเริ่มต้นของแอป Quart กำหนดเส้นทางหลักและการตั้งค่าเซิร์ฟเวอร์ |
| `camera_worker.py` | โมดูลสำหรับบริหาร worker ที่ดึงภาพจากกล้องหรือสตรีม |
| `data_sources/` | เก็บไฟล์คอนฟิกของแต่ละ source รวมถึง ROI ที่บันทึกไว้ |
| `inference_modules/` | รวมโมดูล inference ย่อย เช่น `typhoon_ocr`, `yolo`, `easy_ocr`, `rapid_ocr`, `trocr`, `tesseract_ocr` |
| `src/packages/` | ยูทิลิตีและแพ็กเกจเสริม เช่น โมดูลโมเดลใน `models/` และการแจ้งเตือนใน `notification/` |
| `static/` | ไฟล์ front-end เช่น JavaScript, CSS และรูปภาพ |
| `templates/` | เทมเพลต HTML ของหน้า UI ต่าง ๆ |
| `tests/` | ชุดเทสต์ที่รันด้วย `pytest` สำหรับตรวจสอบพฤติกรรมสำคัญ |
| `pyproject.toml` | ไฟล์กำหนด dependencies และเมทาดาตาของแพ็กเกจ |

## การรันโปรเจ็กต์
1. รัน `python app.py` (ระบุพอร์ตเองได้ด้วย `--port`; ระหว่างพัฒนาอาจใช้ `quart --app app:app run --reload --port 12000`)
2. เปิดเบราว์เซอร์ไปที่ `http://localhost:5000/` หรือพอร์ตที่กำหนด (ระบบจะรีไดเรกต์ไปหน้า `/home`)

### ตัวอย่าง: ทดสอบด้วยไฟล์ภาพนิ่ง (`example.jpg`)
ในกรณีที่ต้องการทดสอบฟีเจอร์ของ VisionROI โดยไม่มีสตรีมวิดีโอหรือกล้องจริง สามารถใช้ไฟล์ภาพนิ่งเพื่อจำลองแหล่งภาพได้ตามขั้นตอนนี้ (สมมติว่ามีไฟล์ `example.jpg` อยู่ในโฟลเดอร์โปรเจ็กต์แล้ว):

1. รันเซิร์ฟเวอร์ด้วย `python app.py` ตามปกติ
2. เปิดเทอร์มินัลอีกหน้าหนึ่ง แล้วลงทะเบียน source ชื่อ `demo_image` (สั่งครั้งแรกครั้งเดียว หากสร้างไว้แล้วสามารถข้ามได้)

   ```bash
   curl -X POST "http://localhost:5000/create_source" \
        -F "name=demo_image" \
        -F "source=/absolute/path/to/example.jpg" \
        -F "stream_type=opencv"
   ```

   > **หมายเหตุ**: แทนที่ `/absolute/path/to/example.jpg` ด้วยพาธแบบ *absolute* ของไฟล์ในเครื่องคุณ (บน Windows สามารถใช้รูปแบบ `C:\\path\\to\\example.jpg`)

3. สั่งให้ worker อ่านไฟล์ภาพนิ่งเพื่อจำลองสตรีมวิดีโอ

   ```bash
   curl -X POST "http://localhost:5000/start_roi_stream/demo_image" \
        -H "Content-Type: application/json" \
        -d '{
              "name": "demo_image",
              "source": "/absolute/path/to/example.jpg",
              "stream_type": "opencv"
            }'
   ```

4. ขอภาพตัวอย่างจาก worker เพื่อยืนยันว่าอ่านภาพนิ่งได้แล้ว (ไฟล์ `snapshot.jpg` จะมีเนื้อหาเดียวกับ `example.jpg`)

   ```bash
   curl "http://localhost:5000/ws_snapshot/demo_image" --output snapshot.jpg
   ```

5. เมื่อทดลองเสร็จให้หยุดการอ่านภาพนิ่ง

   ```bash
   curl -X POST "http://localhost:5000/stop_roi_stream/demo_image"
   ```

เมื่อมีการสร้าง source `demo_image` แล้ว สามารถเปิดหน้า `/roi` หรือ `/inference` เพื่อทดลองเลือก ROI หรือเรียกใช้โมดูลต่าง ๆ กับภาพนิ่งดังกล่าวได้ทันที (ระบบจะใช้เฟรมเดิมจากไฟล์ `example.jpg` ซ้ำ)

## โฟลว์การทำงานจากการสร้าง Source ถึงการรัน Inference Group
1. ไปที่หน้า `/create_source` เพื่อสร้าง source ใหม่ โดยกรอกชื่อและแหล่งกล้อง
2. เลือก source ที่ต้องการในหน้า UI (ระบบจะตั้งค่ากล้องให้อัตโนมัติเมื่อเริ่มสตรีมภาพหรือทำ inference)
3. เปิดหน้า `/roi` แล้วเลือกตำแหน่ง ROI ที่ต้องการ จากนั้นกดบันทึก (ROI แต่ละจุดสามารถระบุโมดูลของตัวเองได้) – เรียก `POST /save_roi`
4. หากต้องการตรวจจับหน้ากระดาษ ให้เข้า Inference Page ที่ `/inference_page` เพื่อดูผลลัพธ์การตรวจจับเพจ
5. เข้า `/inference` แล้วเลือก source กล้องและกลุ่ม ROI เพื่อเริ่มประมวลผล (หากเลือกค่า `All` จะบังคับประมวลผล ROI ทุกตัว) ผลลัพธ์แต่ละ ROI จะถูกส่งกลับผ่าน `/ws_roi_result/<cam_id>` โดยที่ `cam_id` เป็นคีย์สตริง (เช่น `inf_cam1`)
6. เมื่อเสร็จสิ้นสามารถหยุดงานได้ที่ `POST /stop_inference/<cam_id>`

## การตั้งค่าการแจ้งเตือน
### Telegram Notify
ตัวอย่างการใช้งาน:

```python
from src.packages.notification.telegram_notify import TelegramNotify

tg = TelegramNotify(token="YOUR_BOT_TOKEN", chat_id="YOUR_CHAT_ID")
tg.start_send_text("สวัสดี")
```

### MQTT ต่อ ROI
- สามารถสร้างคอนฟิก MQTT ได้จากหน้า `/create_mqtt` โดยระบุ `host`, `port`, ชื่อผู้ใช้/รหัสผ่าน (ถ้ามี), `base_topic`, `client_id`, `qos`, `retain`, `tls` และ `keepalive`
- เมื่อมีคอนฟิกแล้ว หน้า `/roi` จะให้เลือกคอนฟิก MQTT แยกสำหรับแต่ละ ROI ได้ ทำให้ ROI ที่สำคัญสามารถส่งข้อมูลคนละ broker หรือ broker เดียวกันแต่คนละ topic ได้
- หากเลือกคอนฟิกไว้ ROI นั้น ๆ จะเรียก `publish_roi_to_mqtt()` อัตโนมัติเมื่อมีผลลัพธ์จาก inference
- โครงสร้าง topic ที่ระบบสร้างคือ:

  ```
  <base_topic>/<source>/<group>/<roi_id>
  ```

  แต่ละส่วนจะถูกแปลงให้เป็นตัวอักษร, ตัวเลข, `_` หรือ `-` เท่านั้น และตัดช่องว่างออก เช่น `My Camera` จะกลายเป็น `My_Camera`
- ตัวอย่าง: หากตั้งค่า `base_topic` เป็น `vision`, source ชื่อ `cam_north`, กลุ่ม `group_a`, ROI ลำดับ `3` จะได้ topic `vision/cam_north/group_a/3`
- Payload ที่ส่งเป็น JSON ประกอบด้วยข้อมูล ROI เช่น ชื่อ ROI, ค่า inference, ค่าความเชื่อมั่น, timestamp และอาจมี `duration` หากโมดูลส่งมา

## หน้าต่างต่าง ๆ
### Home (`/home`)
หน้าแรกแสดงแดชบอร์ดสรุปฟีเจอร์หลักของระบบ ได้แก่ **Create Source**, **ROI** และ **Inference Group** เพื่อให้ผู้ใช้เข้าใจขั้นตอนการทำงานโดยรวมตั้งแต่จัดการแหล่งวิดีโอไปจนถึงรันโมเดล AI นอกจากนี้เมื่อเข้าใช้งานครั้งแรกจะมีข้อความต้อนรับแสดงขึ้นมา

### Create Source (`/create_source`)
สร้าง source ใหม่โดยระบุชื่อและแหล่งกล้อง เก็บข้อมูลไว้ใน `data_sources/<name>` เมื่อกดปุ่ม **Create** แล้ว รายการจะปรากฏในตารางบนหน้านั้นทันทีโดยไม่ต้องไปหน้าสถานะอื่น

### ROI Selection (`/roi`)
เลือกและบันทึกตำแหน่ง ROI ตาม source ที่เลือก พร้อมระบุโมดูลให้แต่ละ ROI ได้

### Inference Group (`/inference`)
แสดงผลวิดีโอพร้อม ROI สามารถเลือกกลุ่ม ROI เพื่อเรียกใช้ `custom.py` จากโมดูลที่เลือก ผลลัพธ์จะถูกส่งกลับผ่าน `/ws_roi_result/<cam_id>`

### Inference Page (`/inference_page`)
สตรีมภาพจากกล้องพร้อมแสดงชื่อเพจที่ตรวจจับได้ และผลลัพธ์ของ ROI แต่ละจุดภายในเพจนั้น ใช้สำหรับงานที่ต้องระบุหน้ากระดาษโดยเฉพาะ

## API/Endpoints

- **POST `/start_inference/<cam_id>`** – เริ่มอ่านภาพและประมวลผล ROI ที่ส่งมา (หากไม่ส่ง `rois` จะโหลดจากไฟล์ของ source ที่เลือกไว้ก่อนหน้า) พร้อมตั้งค่ากล้องจากข้อมูลใน body (`name`, `source`, `width`, `height`, `stream_type`) และสามารถระบุ `group` เพื่อบังคับเลือก ROI group; หากส่งค่า `all` จะประมวลผล ROI ทุกตัวโดยไม่สนใจผลการจับหน้ากระดาษ ค่า `interval` (หน่วยวินาที) ใช้กำหนดช่วงเวลาการประมวลผลและจะถูกบันทึกใน `service_state.json` เพื่อใช้ต่อในการรีสตาร์ทครั้งถัดไป โดย `cam_id` สามารถเป็นสตริงใด ๆ เพื่อแยกกล้องแต่ละหน้าตามต้องการ
  ```json
  {
    "name": "cam1",
    "source": "0",
    "stream_type": "ffmpeg",
    "group": "p1",
    "interval": 0.5,
    "rois": [
      {
        "id": "1",
        "module": "typhoon_ocr",
        "points": [
          {"x": 10, "y": 20},
          {"x": 110, "y": 20},
          {"x": 110, "y": 100},
          {"x": 10, "y": 100}
        ]
      }
    ]
  }
  ```
  ตอบกลับสำเร็จจะอยู่ในรูป `{"status": "started", "cam_id": "<id>"}` หรือ `"already_running"` หากสตรีมกำลังทำงานอยู่ ส่วนกรณีที่หน้า ROI ยังทำงานอยู่จะได้ `{"status": "roi_running", "cam_id": "<id>"}` พร้อม HTTP 400 และหากเปิดกล้องไม่สำเร็จจะคืน `{"status": "error", "message": "open_failed"}`

- **POST `/stop_inference/<cam_id>`** – หยุดงาน inference ส่งสัญญาณให้ WebSocket ปิดตัวและล้างสถานะกล้องกับ queue ที่เกี่ยวข้อง

- **POST `/start_roi_stream/<cam_id>`** – เริ่มส่งภาพสดสำหรับหน้าเลือก ROI พร้อมตั้งค่ากล้องจากข้อมูลใน body (`name`, `source`, `width`, `height`, `stream_type`) หากงาน inference กำลังทำงานอยู่จะได้รับ `{"status": "inference_running"}` พร้อม HTTP 400

- **POST `/stop_roi_stream/<cam_id>`** – หยุดส่งภาพ ROI และคืนทรัพยากรของกล้องเมื่อไม่มีงานอื่นใช้งานอยู่

- **GET `/roi_stream_status/<cam_id>`** – ตรวจสอบว่างาน ROI stream กำลังทำงานอยู่หรือไม่ (ตอบกลับประกอบด้วย `running`, `cam_id` และ `source` ล่าสุดของสตรีมนั้น)

- **GET `/inference_status/<cam_id>`** – ตรวจสอบว่างาน inference กำลังทำงานอยู่หรือไม่ (ตอบกลับประกอบด้วย `running` และ `cam_id`)
  ```json
  {"running": true, "cam_id": "inf_cam1"}
  ```

- **POST `/save_roi`** – บันทึก ROI ลงไฟล์ของ source หรือพาธที่กำหนด
  ```json
  {
    "source": "cam1",
    "rois": [
      {
        "id": "1",
        "module": "typhoon_ocr",
        "points": [
          {"x": 10, "y": 20},
          {"x": 110, "y": 20},
          {"x": 110, "y": 100},
          {"x": 10, "y": 100}
        ]
      }
    ]
  }
  ```

- **GET `/load_roi/<name>`** – โหลด ROI ล่าสุดของ source นั้น
  ```bash
  GET /load_roi/cam1
  ```

- **GET `/load_roi_file`** – โหลด ROI จากพาธที่ระบุในพารามิเตอร์ `path`
  ```bash
  GET /load_roi_file?path=data_sources/cam1/rois.json
  ```

- **GET `/read_log`** – อ่านบรรทัดล่าสุดจากไฟล์ `custom.log` ของ source ที่ระบุผ่านพารามิเตอร์ `source` (สามารถกำหนดจำนวนบรรทัดด้วย `lines`, ค่าเริ่มต้น 40)

- **GET `/ws_snapshot/<cam_id>`** – คืนรูป JPEG หนึ่งเฟรมจากกล้อง (ต้องมี worker ของกล้องนั้นทำงานอยู่ มิฉะนั้นจะตอบกลับข้อผิดพลาด)

- **GET `/data_sources`** – รายชื่อ source ทั้งหมดในระบบ

- **GET `/inference_modules`** – รายชื่อโมดูล inference ที่มีในระบบ
- **GET `/groups`** – รายชื่อ group หรือ page ที่พบใน ROI ของทุก source
- **GET `/source_list`** – รายชื่อ source พร้อมรายละเอียดของแต่ละตัว

- **GET `/source_config`** – คืนค่าคอนฟิกของ source ตามชื่อที่ระบุผ่านพารามิเตอร์ `name`

- **DELETE `/delete_source/<name>`** – ลบโฟลเดอร์และไฟล์ที่เกี่ยวกับ source นั้น
- **PATCH `/update_stream_type/<name>`** – ปรับชนิดการสตรีม (`opencv` หรือ `ffmpeg`) ของ source ที่ระบุ

- **GET `/_healthz`** – ตรวจสอบสถานะเซิร์ฟเวอร์
- **POST `/_quit`** – สั่งปิดเซิร์ฟเวอร์อย่างนุ่มนวล

- **WebSocket `/ws`** – ส่งภาพ JPEG แบบไบนารีต่อเนื่องสำหรับหน้า `/inference`

- **WebSocket `/ws_roi`** – ส่งภาพ JPEG แบบไบนารีต่อเนื่องสำหรับหน้า `/roi`

- **WebSocket `/ws_roi_result/<cam_id>`** – ส่งผลลัพธ์ ROI ขณะรัน inference โดยมี 2 รูปแบบข้อความ: (1) ข้อมูลการจับหน้ากระดาษ `{"group": "<ชื่อเพจ>", "scores": [{"page": "...", "score": 0.87}, ...]}` และ (2) ผลลัพธ์ ROI รายตัว `{"id": "1", "image": "<base64>", "text": "...", "frame_time": 1700000000.123, "result_time": 1700000000.456}`


## การพัฒนาและการทดสอบ

เพื่อช่วยให้งานพัฒนาเป็นระบบมากขึ้น สามารถใช้คำสั่งด้านล่างในการตั้งค่าและตรวจสอบโค้ด:

1. ติดตั้งแพ็กเกจแบบ editable พร้อม extras ที่จำเป็นสำหรับการทดสอบโมดูลเสริม:

   ```bash
   pip install -e ".[extras]"
   ```

2. รันชุดทดสอบอัตโนมัติด้วย `pytest` เพื่อยืนยันว่าโค้ดยังทำงานถูกต้องหลังแก้ไข:

   ```bash
   pytest
   ```

3. เมื่อต้องการพัฒนา UI สามารถเปิดเซิร์ฟเวอร์แบบ reload อัตโนมัติ:

   ```bash
   quart --app app:app run --reload --port 12000
   ```

   โหมดนี้เหมาะสำหรับการปรับเทมเพลตหรือสคริปต์ front-end เพราะจะแสดงผลการแก้ไขทันทีโดยไม่ต้องรีสตาร์ทเซิร์ฟเวอร์


## โครงสร้าง `data_sources/`
```
data_sources/
└── <name>/
    ├─ config.json
    └─ rois.json
```

ไฟล์ `rois.json` เก็บข้อมูล ROI โดยมีคีย์ `type` แยกชนิดระหว่าง ROI ปกติ (`roi`) และเพจ (`page`):

```json
[
  {
    "id": "1",
    "group": "p1",
    "module": "",
    "points": [
      {"x": 10, "y": 20},
      {"x": 110, "y": 20},
      {"x": 110, "y": 100},
      {"x": 10, "y": 100}
    ],
    "type": "roi"
  },
  {
    "id": "roi_123",
    "page": "page1",
    "points": [
      {"x": 0, "y": 0},
      {"x": 100, "y": 0},
      {"x": 100, "y": 80},
      {"x": 0, "y": 80}
    ],
    "image": "<base64>",
    "type": "page"
  }
]
```

ROI ปกติสามารถกำหนด `group` สำหรับเลือก group id ได้ ส่วนค่า `module` จะเริ่มต้นเป็นค่าว่าง
หากเว้นว่างจะไม่ประมวลผล ROI นั้น ส่วน ROI ที่มี `type` เป็น `page` จะบันทึกชื่อ (`page`) และรูปภาพที่ครอปไว้ในฟิลด์ `image`


โมดูลสำหรับประมวลผลจะเก็บไว้ในโฟลเดอร์ `inference_modules/<module_name>/custom.py`

ตัวอย่างโมดูลที่มีให้:
- `typhoon_ocr` – OCR เอกสารทั่วไป
- `yolo` – ตรวจจับวัตถุพื้นฐานด้วย YOLOv8
- `easy_ocr` – OCR ด้วยไลบรารี EasyOCR
- `rapid_ocr` – OCR ด้วยไลบรารี RapidOCR
- `trocr` – OCR ด้วยโมเดล TrOCR จาก HuggingFace Transformers
- `tesseract_ocr` – OCR ด้วยไลบรารี Tesseract (pytesseract)

### สร้างโมดูล inference เพิ่มเติม

เมธอด `process` ของโมดูลจะถูกเรียกพร้อมอาร์กิวเมนต์อย่างน้อย `[frame, roi_id, save_flag]` และหากฟังก์ชันรองรับพารามิเตอร์ชื่อ `source`, `cam_id` หรือ `interval` ระบบจะส่งค่าเหล่านี้ไปด้วยโดยอัตโนมัติ ข้อมูลที่คืนค่าควรเป็นสตริงหรือดิกชันนารีที่มีคีย์ `text` เพื่อให้ UI แสดงผลได้ อีกทั้งควรมีฟังก์ชัน `cleanup()` สำหรับคืนทรัพยากรเมื่อหยุดงาน

ตัวอย่างการใช้งานคลาส OCR:

```python
from inference_modules.easy_ocr.custom import EasyOCR

ocr = EasyOCR()
text = ocr.process(frame, roi_id="1", save=True, source="demo")
```

## ข้อมูลเพิ่มเติม
- `config.json` เก็บข้อมูล source และไฟล์ ROI
- `service_state.json` จะถูกสร้างอัตโนมัติเมื่อมีการเริ่มงาน ROI หรือ inference เพื่อเก็บสถานะล่าสุดของกล้องแต่ละตัว
- `custom.log` ภายใต้ `data_sources/<name>/` ใช้สำหรับบันทึกผลลัพธ์หรือข้อความจากโมดูล `custom.py`

## การร่วมพัฒนา
- หากพบปัญหาหรือมีข้อเสนอแนะ สามารถเปิด issue ได้
- ก่อนส่ง pull request กรุณารัน `pytest` เพื่อให้แน่ใจว่าเทสต์ทั้งหมดผ่าน
- อธิบายรายละเอียดการเปลี่ยนแปลงใน PR เพื่อให้รีวิวได้ง่ายขึ้น

## การทดสอบ
โปรเจ็กต์มีเทสต์ตัวอย่างใช้ `pytest`
หากยังไม่มี `pytest` ให้ติดตั้งก่อน:

```bash
pip install pytest
```

จากนั้นรันเทสต์ทั้งหมดได้ด้วยคำสั่ง:

```bash
pytest
```

## License

โค้ดนี้เผยแพร่ภายใต้เงื่อนไขที่ต้องอ้างอิงแหล่งที่มา ดูรายละเอียดเพิ่มเติมในไฟล์ [LICENSE](LICENSE)

