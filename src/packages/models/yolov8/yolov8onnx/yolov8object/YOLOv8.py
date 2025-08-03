import time
import cv2
import numpy as np
import onnxruntime

from models.common.box_ops import multiclass_nms


def xywh_to_xyxy(boxes):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes_xyxy



class YOLOv8:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=[
            # "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ])
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    def detect_objects(self, image):
        self.original_shape = image.shape[:2]
        input_tensor, self.new_unpad, self.r = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        boxes, scores, class_ids = self.process_output(
            outputs, self.original_shape, self.new_unpad, self.r
        )
        return boxes, scores, class_ids

    def prepare_input(self, image):
        h0, w0 = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, r, (dw, dh) = self.letterbox(img, (self.input_width, self.input_height))
        input_tensor = (img / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        return input_tensor, new_unpad, r

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_output(self, output, original_shape, new_unpad, r):
        preds = np.squeeze(output[0]).T
        scores = np.max(preds[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        preds, scores = preds[mask], scores[mask]

        if len(scores) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([])

        class_ids = np.argmax(preds[:, 4:], axis=1)
        boxes_xywh = self.extract_boxes(preds[:, :4], original_shape, new_unpad)
        boxes_xywh = np.array(boxes_xywh)

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        indices = multiclass_nms(boxes_xyxy, scores, class_ids, self.iou_threshold)
        if len(indices) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([])

        indices = np.array(indices).flatten()
        boxes_nms = np.array([[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes_xywh])[indices]
        return boxes_nms, scores[indices], class_ids[indices]

    def extract_boxes(self, xywh, original_shape, new_unpad):
        h0, w0 = original_shape
        r = min(self.input_height / h0, self.input_width / w0)
        dw, dh = (self.input_width - new_unpad[0]) / 2, (self.input_height - new_unpad[1]) / 2
        out = []
        for cx, cy, w, h in xywh:
            x = (cx - w / 2 - dw) / r
            y = (cy - h / 2 - dh) / r
            w /= r; h /= r
            x, y = max(0, x), max(0, y)
            w, h = min(w0 - x, w), min(h0 - y, h)
            out.append([x, y, w, h])
        return np.array(out)

    def letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = image.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (dw, dh)
