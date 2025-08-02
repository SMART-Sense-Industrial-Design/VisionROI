# web_ocrroi

โปรเจ็กต์นี้เป็นแอป Quart สำหรับทดสอบ OCR และการจัดการ ROI

## โครงสร้าง `sources/`

```
sources/
└── <name>/
    ├── model.onnx
    ├── classes.txt
    ├── config.json
    └── rois.json
```

เมื่อใช้หน้า **Create Source** (ที่ `/create_source`) ระบบจะรับชื่อ, ค่า source, ไฟล์ model และ label แล้วจะสร้างโฟลเดอร์ใหม่ใน `sources/<name>` พร้อมบันทึกไฟล์และไฟล์ `config.json` ที่เก็บข้อมูล:

```
{
  "name": "...",
  "source": "...",
  "model": "model.onnx",
  "label": "classes.txt",
  "rois": "rois.json"
}
```

หน้า **ROI** จะอ่านและบันทึก ROI จากไฟล์ `rois.json` ในโฟลเดอร์ของ source นั้น ๆ

## วิธีใช้งานหน้า Create Source

1. เปิด `/create_source` ในเบราว์เซอร์
2. กรอก `Name` และ `Source`
3. เลือกไฟล์โมเดล (`model.onnx`) และไฟล์ label (`classes.txt`)
4. กด **Create** ระบบจะสร้างโฟลเดอร์ใหม่ใน `sources/` พร้อมไฟล์ทั้งหมด

