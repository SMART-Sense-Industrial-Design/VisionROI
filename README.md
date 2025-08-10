# VisionROI

โปรเจ็กต์ **VisionROI** เป็นแอป Quart สำหรับทดสอบ OCR และการจัดการ ROI

## ฟีเจอร์
- เลือกและบันทึกตำแหน่ง ROI จากกล้องหรือวิดีโอ
- รันโมเดลเพื่อตรวจจับข้อความหรือวัตถุใน ROI
- รองรับการแจ้งเตือนผ่าน Telegram
- มาพร้อมโมดูลตัวอย่าง `typhoon_ocr`, `yolo` และ `easy_ocr` เพื่อเริ่มต้นทดลองใช้งาน

## ข้อกำหนดระบบ
- Python ≥3.10
- dependencies ถูกกำหนดใน `pyproject.toml`

### Dependencies หลัก
- `Quart`
- `opencv-python`
- `numpy`
- `Pillow`
- `easyocr`

### Dependencies เพิ่มเติม (Extras)
- `onnxruntime` – สำหรับรันโมเดล ONNX
- `requests` – ใช้ส่งการแจ้งเตือนผ่าน Telegram เท่านั้น (หากต้องการรองรับ Line จำเป็นต้องพัฒนาโมดูลเพิ่มเติม)
- `tensorflow` – สำหรับโมเดลที่ใช้ TFLite
- `torch` – สำหรับฟังก์ชันที่ใช้ PyTorch
- `websockets` – สำหรับการเชื่อมต่อผ่าน WebSocket แบบไคลเอนต์
- `onnx` – เครื่องมือสำหรับจัดการโมเดล ONNX
- `torchvision` – ยูทิลิตีเสริมสำหรับ PyTorch
- `pycuda` – จำเป็นสำหรับ TensorRT backend
- `tensorrt` – เร่งความเร็วโมเดลด้วย TensorRT

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

เมื่อ environment ถูก activate แล้วจึงติดตั้งแพ็กเกจและ dependencies ด้วย `pip`:

```bash
pip install "."
```

หากต้องการ dependencies เพิ่มเติม เช่น `onnxruntime`, `requests`, `tensorflow`, `torch`, `websockets`, `onnx`, `torchvision`, `pycuda` หรือ `tensorrt` สามารถติดตั้งผ่าน extras ได้:

```bash
pip install ".[extras]"
```

หรือหากต้องการติดตั้งแบบ editable:

```bash
pip install -e "."
```

### การตั้งค่า Info.plist บน macOS
หากคุณแพ็กเกจโปรเจ็กต์นี้เป็นแอปสำหรับ macOS (เช่น ผ่าน PyInstaller หรือ py2app) จำเป็นต้องกำหนดคีย์ `NSCameraUsageDescription` และระบุชนิดกล้องที่จะใช้ในไฟล์ `Info.plist` มิฉะนั้นระบบจะไม่อนุญาตให้เข้าถึงกล้อง

```xml
<!-- Info.plist -->
<plist version="1.0">
  <dict>
    <key>NSCameraUsageDescription</key>
    <string>แอปต้องการใช้กล้องเว็บแคมเพื่ออ่านภาพสำหรับ OCR</string>
  </dict>
</plist>
```

ปรับข้อความใน `<string>` ให้สอดคล้องกับชนิดกล้องที่คุณต้องการ เช่น กล้องในตัวของเครื่อง หรือกล้อง USB ภายนอก

## โครงสร้างโปรเจ็กต์
โครงสร้างไฟล์หลักของโปรเจ็กต์:

```
VisionROI/
├─ app.py
├─ camera_worker.py
├─ data_sources/
├─ inference_modules/
│  ├─ easy_ocr/
│  ├─ typhoon_ocr/
│  └─ yolo/
├─ src/
├─ static/
├─ templates/
├─ tests/
└─ pyproject.toml
```

## การรันโปรเจ็กต์
1. รัน `python app.py --port 12000` (หรือใช้ `quart --app app:app run --reload --port 12000` เพื่อรีโหลดอัตโนมัติระหว่างพัฒนา)
2. เปิดเบราว์เซอร์ไปที่ `http://localhost:12000/` (หากไม่ระบุ `--port` ค่าเริ่มต้นคือ `5000` และระบบจะรีไดเรกต์ไปหน้า `/home`)

## โฟลว์การทำงานจากการสร้าง Source ถึงการรัน Inference
1. ไปที่หน้า `/create_source` เพื่อสร้าง source ใหม่ โดยกรอกชื่อและแหล่งกล้อง
2. ตั้งค่ากล้องและเลือก source (พร้อมกำหนดโมดูลเริ่มต้นหากต้องการ) ด้วย `POST /set_camera/<cam_id>` หรือผ่านหน้า UI
3. เปิดหน้า `/roi` แล้วเลือกตำแหน่ง ROI ที่ต้องการ จากนั้นกดบันทึก (ROI แต่ละจุดสามารถระบุโมดูลของตัวเองได้) – เรียก `POST /save_roi`
4. เข้า `/inference` แล้วเลือก source กล้องเพื่อเริ่มประมวลผล ผลลัพธ์แต่ละ ROI จะถูกส่งกลับผ่าน `/ws_roi_result/<cam_id>`
5. เมื่อเสร็จสิ้นสามารถหยุดงานได้ที่ `POST /stop_inference/<cam_id>`

## การตั้งค่าการแจ้งเตือน
### Telegram Notify
ตัวอย่างการใช้งาน:

```python
from src.packages.notification.telegram_notify import TelegramNotify

tg = TelegramNotify(token="YOUR_BOT_TOKEN", chat_id="YOUR_CHAT_ID")
tg.start_send_text("สวัสดี")
```

## หน้าต่างต่าง ๆ
### Create Source (`/create_source`)
สร้าง source ใหม่โดยระบุชื่อและแหล่งกล้อง เก็บข้อมูลไว้ใน `data_sources/<name>`

### ROI Selection (`/roi`)
เลือกและบันทึกตำแหน่ง ROI ตาม source ที่เลือก พร้อมระบุโมดูลให้แต่ละ ROI ได้

### Inference (`/inference`)
แสดงผลวิดีโอพร้อม ROI และเรียกใช้ `custom.py` จากโมดูลที่เลือก ผลลัพธ์จะถูกส่งกลับผ่าน `/ws_roi_result/<cam_id>`

## API/Endpoints

- **POST `/set_camera/<cam_id>`** – ตั้งค่ากล้อง เลือก source และกำหนดโมดูลเริ่มต้นได้
  ```json
  {"name": "cam1", "source": "0", "module": "yolo"}
  ```

- **POST `/start_inference/<cam_id>`** – เริ่มอ่านภาพและประมวลผล ROI ที่ส่งมา (หากไม่ส่ง `rois` จะโหลดจากไฟล์ของ source)
  ```json
  {
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

- **POST `/stop_inference/<cam_id>`** – หยุดงาน inference

- **POST `/start_roi_stream/<cam_id>`** – เริ่มส่งภาพสดสำหรับหน้าเลือก ROI

- **POST `/stop_roi_stream/<cam_id>`** – หยุดส่งภาพ ROI

- **GET `/roi_stream_status/<cam_id>`** – ตรวจสอบว่างาน ROI stream กำลังทำงานอยู่หรือไม่

- **GET `/inference_status/<cam_id>`** – ตรวจสอบว่างาน inference กำลังทำงานอยู่หรือไม่
  ```json
  {"running": true}
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

- **GET `/ws_snapshot/<cam_id>`** – คืนรูป JPEG หนึ่งเฟรมจากกล้อง

- **GET `/data_sources`** – รายชื่อ source ทั้งหมดในระบบ

- **GET `/inference_modules`** – รายชื่อโมดูล inference ที่มีในระบบ

- **GET `/source_list`** – รายชื่อ source พร้อมรายละเอียดของแต่ละตัว

- **GET `/source_config`** – คืนค่าคอนฟิกของ source ตามชื่อที่ระบุผ่านพารามิเตอร์ `name`

- **DELETE `/delete_source/<name>`** – ลบโฟลเดอร์และไฟล์ที่เกี่ยวกับ source นั้น

- **WebSocket `/ws`** – ส่งภาพ JPEG แบบไบนารีต่อเนื่องสำหรับหน้า `/inference`

- **WebSocket `/ws_roi`** – ส่งภาพ JPEG แบบไบนารีต่อเนื่องสำหรับหน้า `/roi`

- **WebSocket `/ws_roi_result/<cam_id>`** – ส่งผลลัพธ์ ROI (รูปภาพและข้อความ) ขณะรัน inference


## โครงสร้าง `data_sources/`
```
data_sources/
└── <name>/
    ├─ config.json
    └─ rois.json
```

ไฟล์ `rois.json` เก็บ ROI ในรูปแบบ:

```json
[
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
```

โมดูลสำหรับประมวลผลจะเก็บไว้ในโฟลเดอร์ `inference_modules/<module_name>/custom.py`

ตัวอย่างโมดูลที่มีให้:
- `typhoon_ocr` – OCR เอกสารทั่วไป
- `yolo` – ตรวจจับวัตถุพื้นฐานด้วย YOLOv8
- `easy_ocr` – OCR ด้วยไลบรารี EasyOCR

## ข้อมูลเพิ่มเติม
- `config.json` เก็บข้อมูล source และไฟล์ ROI

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

