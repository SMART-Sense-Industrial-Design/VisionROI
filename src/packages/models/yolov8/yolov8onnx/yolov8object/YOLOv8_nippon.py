import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8:
    """YOLOv8 object detection model class for handling inference on video frames."""

    def __init__(self, onnx_model, conf_thres=0.5, iou_thres=0.5):
        self.onnx_model = onnx_model
        self.confidence_thres = conf_thres
        self.iou_thres = iou_thres

        # Load ONNX model
        self.session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        model_inputs = self.session.get_inputs()
        self.input_width = model_inputs[0].shape[2]
        self.input_height = model_inputs[0].shape[3]

    def __call__(self, frame):
        return self.detect_objects(frame)

    def letterbox(self, img, new_shape=(640, 640)):
        """Resizes image with padding."""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left)

    def preprocess(self, frame):
        """Preprocess a frame for inference."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, pad = self.letterbox(img, (self.input_width, self.input_height))
        img = np.array(img) / 255.0
        img = np.transpose(img, (2, 0, 1))  # Channel first
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img, pad

    def postprocess(self, frame, output, pad):
        """Post-process the model output and draw detections."""
        outputs = np.squeeze(output[0]).T
        rows = outputs.shape[0]

        boxes, scores, class_ids = [], [], []
        r = min(self.input_height / frame.shape[0], self.input_width / frame.shape[1])
        new_unpad = (int(round(frame.shape[1] * r)), int(round(frame.shape[0] * r)))
        dw, dh = (self.input_width - new_unpad[0]) / 2, (self.input_height - new_unpad[1]) / 2

        for i in range(rows):
            class_scores = outputs[i][4:]
            max_score = np.amax(class_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(class_scores)
                x, y, w, h = outputs[i][:4]

                # Convert to original scale
                left = int((x - w / 2 - dw) * frame.shape[1] / new_unpad[0])
                top = int((y - h / 2 - dh) * frame.shape[0] / new_unpad[1])
                width = int(w * frame.shape[1] / new_unpad[0])
                height = int(h * frame.shape[0] / new_unpad[1])

                left = max(0, min(left, frame.shape[1] - 1))
                top = max(0, min(top, frame.shape[0] - 1))
                width = min(width, frame.shape[1] - left)
                height = min(height, frame.shape[0] - top)

                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        if indices is not None and len(indices) > 0:
            indices = indices.astype(np.int32).flatten()  # แปลงเป็น NumPy int32 เพื่อลด overhead
            boxes = np.array(boxes)[indices]  # ดึง bounding boxes ด้วย NumPy indexing
            scores = np.array(scores)[indices]  # ดึง scores
            class_ids = np.array(class_ids)[indices]  # ดึง class IDs

        return boxes, scores, class_ids


    def detect_objects(self, frame):
        """Perform object detection on a single frame."""
        img_data, pad = self.preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        detections = self.postprocess(frame, outputs, pad)
        return detections