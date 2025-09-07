# VisionROI

โปรเจ็กต์ **VisionROI** เป็นแอป Quart สำหรับทดสอบ OCR และการจัดการ ROI

## การเริ่มต้นอย่างรวดเร็ว

1. สร้าง virtual environment แล้ว activate:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # บน Windows ใช้ .venv\\Scripts\\activate
   ```

2. ติดตั้งแพ็กเกจของโปรเจ็กต์:

   ```bash
   pip install "."
   ```

3. รันแอปพลิเคชัน (ค่าเริ่มต้นใช้พอร์ต `5000` หากต้องการกำหนดพอร์ตเองให้เพิ่ม `--port`).
   หากต้องการใช้เซิร์ฟเวอร์ **uvicorn** (แนะนำสำหรับการใช้งานจริง) ให้เพิ่ม `--use-uvicorn`:

   ```bash
   python app.py                       # ใช้เซิร์ฟเวอร์ภายใน
   python app.py --port 12000          # ระบุพอร์ตเอง
   python app.py --use-uvicorn         # รันด้วย uvicorn
   ```

4. เปิดเบราว์เซอร์ไปที่ `http://localhost:5000/` หรือ `/home` (หรือพอร์ตที่กำหนด)

## ฟีเจอร์
- เลือกและบันทึกตำแหน่ง ROI จากกล้องหรือวิดีโอ
- รันโมเดลเพื่อตรวจจับข้อความหรือวัตถุใน ROI ตามกลุ่มที่เลือก (Inference Group)
- ตรวจจับหน้ากระดาษและ ROI แบบเพจผ่านหน้า Inference Page (`/inference_page`)
- รองรับการแจ้งเตือนผ่าน Telegram
- มาพร้อมโมดูลตัวอย่าง `typhoon_ocr`, `yolo`, `easy_ocr` และ `rapid_ocr` เพื่อเริ่มต้นทดลองใช้งาน
- บันทึกสถานะกล้องและงาน inference ลงไฟล์ `service_state.json` เพื่อกู้คืนได้หลังรีสตาร์ทเซิร์ฟเวอร์
- มี endpoint `GET /_healthz` สำหรับตรวจสอบสถานะ และ `POST /_quit` สำหรับสั่งปิดเซิร์ฟเวอร์อย่างนุ่มนวล

ตัวอย่างไฟล์ `vision-roi.service` มีให้สำหรับรันเป็น **systemd service** ซึ่งเรียก `/_quit` เพื่อหยุดแอปอย่างปลอดภัย

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
│  ├─ rapid_ocr/
│  ├─ typhoon_ocr/
│  └─ yolo/
├─ src/
├─ static/
├─ templates/
├─ tests/
└─ pyproject.toml
```

## การรันโปรเจ็กต์
1. รัน `python app.py` (ระบุพอร์ตเองได้ด้วย `--port`; ระหว่างพัฒนาอาจใช้ `quart --app app:app run --reload --port 12000`)
2. เปิดเบราว์เซอร์ไปที่ `http://localhost:5000/` หรือพอร์ตที่กำหนด (ระบบจะรีไดเรกต์ไปหน้า `/home`)

## โฟลว์การทำงานจากการสร้าง Source ถึงการรัน Inference Group
1. ไปที่หน้า `/create_source` เพื่อสร้าง source ใหม่ โดยกรอกชื่อและแหล่งกล้อง
2. เลือก source ที่ต้องการในหน้า UI (ระบบจะตั้งค่ากล้องให้อัตโนมัติเมื่อเริ่มสตรีมภาพหรือทำ inference)
3. เปิดหน้า `/roi` แล้วเลือกตำแหน่ง ROI ที่ต้องการ จากนั้นกดบันทึก (ROI แต่ละจุดสามารถระบุโมดูลของตัวเองได้) – เรียก `POST /save_roi`
4. หากต้องการตรวจจับหน้ากระดาษ ให้เข้า Inference Page ที่ `/inference_page` เพื่อดูผลลัพธ์การตรวจจับเพจ
5. เข้า `/inference` แล้วเลือก source กล้องและกลุ่ม ROI เพื่อเริ่มประมวลผล ผลลัพธ์แต่ละ ROI จะถูกส่งกลับผ่าน `/ws_roi_result/<cam_id>` โดยที่ `cam_id` เป็นคีย์สตริง (เช่น `inf_cam1`)
6. เมื่อเสร็จสิ้นสามารถหยุดงานได้ที่ `POST /stop_inference/<cam_id>`

## การตั้งค่าการแจ้งเตือน
### Telegram Notify
ตัวอย่างการใช้งาน:

```python
from src.packages.notification.telegram_notify import TelegramNotify

tg = TelegramNotify(token="YOUR_BOT_TOKEN", chat_id="YOUR_CHAT_ID")
tg.start_send_text("สวัสดี")
```

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

- **POST `/start_inference/<cam_id>`** – เริ่มอ่านภาพและประมวลผล ROI ที่ส่งมา (หากไม่ส่ง `rois` จะโหลดจากไฟล์ของ source) พร้อมตั้งค่ากล้องจากข้อมูลใน body (`name`, `source`, `width`, `height`) โดย `cam_id` สามารถเป็นสตริงใด ๆ เพื่อแยกกล้องแต่ละหน้าตามต้องการ
  ```json
  {
    "name": "cam1",
    "source": "0",
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

- **POST `/start_roi_stream/<cam_id>`** – เริ่มส่งภาพสดสำหรับหน้าเลือก ROI พร้อมตั้งค่ากล้องจากข้อมูลใน body (`name`, `source`, `width`, `height`)

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

- **GET `/read_log`** – อ่านบรรทัดล่าสุดจากไฟล์ `custom.log` ของ source ที่ระบุผ่านพารามิเตอร์ `source` (สามารถกำหนดจำนวนบรรทัดด้วย `lines`, ค่าเริ่มต้น 20)

- **GET `/ws_snapshot/<cam_id>`** – คืนรูป JPEG หนึ่งเฟรมจากกล้อง

- **GET `/data_sources`** – รายชื่อ source ทั้งหมดในระบบ

- **GET `/inference_modules`** – รายชื่อโมดูล inference ที่มีในระบบ
- **GET `/groups`** – รายชื่อ group หรือ page ที่พบใน ROI ของทุก source
- **GET `/source_list`** – รายชื่อ source พร้อมรายละเอียดของแต่ละตัว

- **GET `/source_config`** – คืนค่าคอนฟิกของ source ตามชื่อที่ระบุผ่านพารามิเตอร์ `name`

- **DELETE `/delete_source/<name>`** – ลบโฟลเดอร์และไฟล์ที่เกี่ยวกับ source นั้น

- **GET `/_healthz`** – ตรวจสอบสถานะเซิร์ฟเวอร์
- **POST `/_quit`** – สั่งปิดเซิร์ฟเวอร์อย่างนุ่มนวล

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

## License

โค้ดนี้เผยแพร่ภายใต้เงื่อนไขที่ต้องอ้างอิงแหล่งที่มา ดูรายละเอียดเพิ่มเติมในไฟล์ [LICENSE](LICENSE)

