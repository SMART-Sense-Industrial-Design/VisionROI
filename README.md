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

เมื่อใช้หน้า **Create Source** (ที่ `/create_source`) ระบบจะรับชื่อและค่า source พร้อมไฟล์ model และ label (ถ้ามี) แล้วจะสร้างโฟลเดอร์ใหม่ใน `sources/<name>` พร้อมบันทึกไฟล์และไฟล์ `config.json` ที่เก็บข้อมูล:

```
{
  "name": "...",
  "source": "...",
  "model": "model.onnx",  // หรือ "" ถ้าไม่ได้อัปโหลด
  "label": "classes.txt", // หรือ "" ถ้าไม่ได้อัปโหลด
  "rois": "rois.json"
}
```

หน้า **ROI** จะอ่านและบันทึก ROI จากไฟล์ `rois.json` ในโฟลเดอร์ของ source นั้น ๆ

## วิธีใช้งานหน้า Create Source

1. เปิด `/create_source` ในเบราว์เซอร์
2. กรอก `Name` และ `Source`
3. (ไม่บังคับ) เลือกไฟล์โมเดล (`model.onnx`) และไฟล์ label (`classes.txt`)
4. กด **Create** ระบบจะสร้างโฟลเดอร์ใหม่ใน `sources/` พร้อมไฟล์ที่อัปโหลด

