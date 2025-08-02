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
สร้าง source ใหม่และอัปโหลดโมเดล/label เพื่อเตรียมไฟล์ใน `sources/<name>`

### ROI Selection (`/roi`)
เลือกและบันทึกตำแหน่ง ROI ตาม source ที่เลือก

### Inference (`/inference`)
แสดงผลวิดีโอพร้อม ROI และเรียกฟังก์ชัน `custom.py` ถ้ามี

## โครงสร้าง `sources/`
```
sources/
└── <name>/
    ├─ model.onnx
    ├─ classes.txt
    ├─ config.json
    ├─ rois.json
    └─ custom.py
```

ไฟล์ `custom.py` ต้องมีฟังก์ชัน `process(frame)` เพื่อประมวลผลเฟรมหรือ ROI ตามต้องการ

## ข้อมูลเพิ่มเติม
- สร้าง `custom.py` ตามตัวอย่างข้างต้นภายใน `sources/<name>/`
- `config.json` เก็บข้อมูล source, โมเดล และไฟล์ ROI
