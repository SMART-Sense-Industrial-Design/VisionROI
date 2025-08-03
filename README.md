# web_ocrroi

โปรเจ็กต์นี้เป็นแอป Quart สำหรับทดสอบ OCR และการจัดการ ROI

## ฟีเจอร์
- เลือกและบันทึกตำแหน่ง ROI จากกล้องหรือวิดีโอ
- รันโมเดลเพื่อตรวจจับข้อความหรือวัตถุใน ROI
- รองรับการแจ้งเตือนผ่าน Telegram

## ข้อกำหนดระบบ
- Python ≥3.10
- dependencies ถูกกำหนดใน `pyproject.toml`

### Dependencies เพิ่มเติม
- `onnxruntime` – สำหรับรันโมเดล ONNX
- `requests` – ใช้ส่งการแจ้งเตือนผ่าน Telegram เท่านั้น (หากต้องการรองรับ Line จำเป็นต้องพัฒนาโมดูลเพิ่มเติม)
- `tensorflow` – สำหรับโมเดลที่ใช้ TFLite
- `torch` – สำหรับฟังก์ชันที่ใช้ PyTorch

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

หากต้องการ dependencies เพิ่มเติม เช่น `onnxruntime`, `requests`, `tensorflow`, `torch` สามารถติดตั้งผ่าน extras ได้:

```bash
pip install ".[extras]"
```

หรือหากต้องการติดตั้งแบบ editable:

```bash
pip install -e "."
```

## โครงสร้างโปรเจ็กต์
โครงสร้างไฟล์หลักของโปรเจ็กต์:

```
web_ocrroi/
├─ app.py
├─ data_sources/
├─ src/
├─ static/
├─ templates/
└─ tests/
```

## การรันโปรเจ็กต์
1. รัน `python app.py`
2. เปิดเบราว์เซอร์ไปที่ `http://localhost:5000/home`

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
สร้าง source ใหม่และอัปโหลดโมเดล/label เพื่อเตรียมไฟล์ใน `data_sources/<name>`

### ROI Selection (`/roi`)
เลือกและบันทึกตำแหน่ง ROI ตาม source ที่เลือก

### Inference (`/inference`)
แสดงผลวิดีโอพร้อม ROI และเรียกฟังก์ชัน `custom.py` ถ้ามี

## API/Endpoints

- **POST `/set_camera`** – ตั้งค่ากล้องและเลือก source ที่จะใช้
  ```json
  {"name": "cam1", "source": "0"}
  ```

- **POST `/start_inference`** – เริ่มอ่านภาพและประมวลผล ROI ที่ส่งมา
  ```json
  {"rois": [{"x": 10, "y": 20, "width": 100, "height": 80}]}
  ```

- **POST `/stop_inference`** – หยุดงาน inference

- **POST `/start_roi_stream`** – เริ่มส่งภาพสดสำหรับหน้าเลือก ROI

- **POST `/stop_roi_stream`** – หยุดส่งภาพ ROI

- **POST `/save_roi`** – บันทึก ROI ลงไฟล์ของ source หรือพาธที่กำหนด
  ```json
  {"source": "cam1", "rois": [{"x": 10, "y": 20, "width": 100, "height": 80}]}
  ```

- **GET `/load_roi/<name>`** – โหลด ROI ล่าสุดของ source นั้น
  ```bash
  GET /load_roi/cam1
  ```

- **GET `/load_roi_file`** – โหลด ROI จากพาธที่ระบุในพารามิเตอร์ `path`
  ```bash
  GET /load_roi_file?path=data_sources/cam1/rois.json
  ```

- **GET `/ws_snapshot`** – คืนรูป JPEG หนึ่งเฟรมจากกล้อง

- **WebSocket `/ws`** – ส่งภาพ base64 ต่อเนื่องสำหรับหน้า `/inference`

- **WebSocket `/ws_roi`** – ส่งภาพ base64 ต่อเนื่องสำหรับหน้า `/roi`

- **GET `/inference_status`** – ตรวจสอบว่างาน inference กำลังทำงานอยู่หรือไม่
  ```json
  {"running": true}
  ```

## โครงสร้าง `data_sources/`
```
data_sources/
└── <name>/
    ├─ model.onnx
    ├─ classes.txt
    ├─ config.json
    ├─ rois.json
    └─ custom.py
```

ไฟล์ `custom.py` ต้องมีฟังก์ชัน `process(frame)` เพื่อประมวลผลเฟรมหรือ ROI ตามต้องการ

## ข้อมูลเพิ่มเติม
- สร้าง `custom.py` ตามตัวอย่างข้างต้นภายใน `data_sources/<name>/`
- `config.json` เก็บข้อมูล source, โมเดล และไฟล์ ROI

## การทดสอบ
ขณะนี้โปรเจ็กต์ยังไม่มีชุดทดสอบอย่างเป็นทางการ
แต่รองรับการเขียนเทสต์ด้วย `pytest` หากต้องการเพิ่มขึ้นมาเอง ตัวอย่างง่าย ๆ:

```python
# tests/test_app.py
def test_example():
    assert True
```

เมื่อสร้างไฟล์เทสต์แล้ว สามารถรันทดสอบได้ด้วยคำสั่ง:

```bash
pytest
```

