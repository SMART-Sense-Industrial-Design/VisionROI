# web_ocrroi

โปรเจ็กต์นี้เป็นแอป Quart สำหรับทดสอบ OCR และการจัดการ ROI

## ข้อกำหนดระบบ
- Python ≥3.10
- dependencies ถูกกำหนดใน `pyproject.toml`

## การติดตั้ง
ติดตั้งแพ็กเกจและ dependencies ด้วย `pip` โดยไม่จำเป็นต้องใช้ `requirements.txt`:

```bash
pip install .
```

หรือหากต้องการติดตั้งแบบ editable:

```bash
pip install -e .
```

## การรันโปรเจ็กต์
1. รัน `python app.py`
2. เปิดเบราว์เซอร์ไปที่ `http://localhost:5000/home`

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
