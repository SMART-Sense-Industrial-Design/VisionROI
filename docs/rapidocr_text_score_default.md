# ค่า default ของ RapidOCR `text_score`

เอกสารนี้บันทึก snapshot ของค่าตั้งต้นจากไฟล์ `config.yaml` ภายในแพ็กเกจ `rapidocr_onnxruntime` เพื่อใช้อ้างอิงเวลา debug ค่า confidence

> ปัจจุบัน VisionROI override ค่า `text_score` เป็น `0.3` ระหว่างสร้าง RapidOCR reader เสมอ เพื่อผ่อน threshold จากค่าดีฟอลต์ 0.5【F:inference_modules/rapid_ocr/custom.py†L37-L80】

```yaml
Global:
  text_score: 0.5
```
