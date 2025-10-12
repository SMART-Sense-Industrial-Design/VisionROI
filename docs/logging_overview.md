# โครงสร้างใหม่ของโฟลเดอร์ log

ระบบได้ย้ายไฟล์ log ทั้งหมดมารวมกันภายใต้โฟลเดอร์ `logs/` ที่ root ของโปรเจกต์ โดยทุกชุด log จะอยู่ในโฟลเดอร์ย่อยที่ขึ้นต้นด้วย `log_` ตามชื่อที่เกี่ยวข้อง เช่น log ของแหล่งสัญญาณกล้อง `webcam` จะถูกเก็บใน `logs/log_webcam/` ส่วน log ของโมดูลที่ไม่มี source ก็จะถูกจัดเก็บใน `logs/log_<ชื่อโมดูล>/` แทนการแยกกระจายอยู่ใน `data_sources/` หรือ `inference_modules/` เหมือนเดิม

ฟังก์ชัน `get_logger` ใน `src/utils/logger.py` จะเป็นผู้สร้างโครงสร้างโฟลเดอร์ใหม่นี้อัตโนมัติ โดยยังคงใช้ `TimedRotatingFileHandler` เพื่อหมุนไฟล์ log รายวันและเก็บย้อนหลัง 7 วันเหมือนเดิม ดังนั้นเมื่อโค้ดเรียก `get_logger` ด้วยชื่อโมดูลหรือแหล่งข้อมูลใด จะพบว่าไฟล์ log ถูกเขียนลงในโฟลเดอร์ `logs/log_<ชื่อที่เกี่ยวข้อง>/` ตามรูปแบบใหม่นี้

## รายการ log ที่มีอยู่ปัจจุบัน

| หมวด log | โฟลเดอร์ปลายทาง | รูปแบบไฟล์/หมายเหตุ | คำอธิบาย |
| --- | --- | --- | --- |
| **Aggregated ROI** | `logs/log_<source>/` | `custom.log` (แยกตาม source) | บันทึกผลรวมการตรวจจับ ROI รายกล้องจากงาน background ของ `aggregated_roi`. |
| **MQTT** | `logs/log_<source>/mqtt/` | `log_<roi>.log` ตามชื่อ ROI | จัดเก็บผลการเผยแพร่ข้อความ MQTT ของแต่ละ ROI เพื่อไล่ปัญหาได้ละเอียดถึงระดับพื้นที่สนใจ. |
| **Inference Queue** | `logs/log_inference_queue/` | `custom.log` | เตือนเมื่อคิวประมวลผลเต็ม ช้า หรือมีข้อผิดพลาดระหว่างดึงงานเข้า worker. |
| **Websocket Stream** | `logs/log_websocket_stream/` | `custom.log` | แจ้งเตือนการส่งเฟรมขึ้น WebSocket ล่าช้า หรือหลุดการเชื่อมต่อ. |
| **Camera Startup** | `logs/log_camera_startup/` | `custom.log` | ใช้ติดตามการ warm-up ของ ffmpeg/avfoundation และเหตุผลที่กล้องเริ่มไม่สำเร็จ. |
| **Inference Modules** | `logs/log_<source>/` | `custom.log` | โมดูล OCR/YOLO (`rapid_ocr`, `easy_ocr`, `typhoon_ocr`, `light_button`, `yolo`, ฯลฯ) จะบันทึกผลไว้ใต้โฟลเดอร์ของ source เดียวกับกล้อง. |
| **Fallback/ไม่มี source** | `logs/log_<module>/` | `custom.log` | หากไม่มีการระบุ source จะตกไปที่ชื่อโมดูล เช่น `log_base_ocr`, `log_aggregated_roi`. |

> หมายเหตุ: `_sanitize_component` และ `_normalize_log_filename` จะทำความสะอาดชื่อ source/ROI ให้อยู่ในรูปแบบที่ปลอดภัยต่อระบบไฟล์ โดยแทนที่อักขระพิเศษด้วย `_` และเติมนามสกุล `.log` ให้โดยอัตโนมัติหากไม่ได้ระบุไว้
