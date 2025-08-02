import cv2
import numpy as np
import onnxruntime as ort

def multiclass_nms(boxes, scores, labels, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) for multiple classes."""
    final_boxes = []
    final_scores = []
    final_labels = []

    unique_labels = np.unique(labels)
    for cls in unique_labels:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        # Convert to [x, y, w, h] for OpenCV NMS input
        boxes_xywh = np.stack([
            cls_boxes[:, 0],
            cls_boxes[:, 1],
            cls_boxes[:, 2] - cls_boxes[:, 0],
            cls_boxes[:, 3] - cls_boxes[:, 1]
        ], axis=1)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(),
            scores=cls_scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )

        if len(indices) > 0:
            for idx in indices.flatten():
                # ⬇️ Convert to [x, y, w, h] here before storing
                x1, y1, x2, y2 = cls_boxes[idx]
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                final_boxes.append([x, y, w, h])
                final_scores.append(cls_scores[idx])
                final_labels.append(cls)

    return np.array(final_boxes), np.array(final_scores), np.array(final_labels)


class RTDETRv2:
    def __init__(self, onnx_file, conf_thres=0.6, iou_thres=0.5):
        self.onnx_file = onnx_file
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.session = ort.InferenceSession(
            self.onnx_file,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider", 
                "CPUExecutionProvider"
            ]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.orig_size_name = self.session.get_inputs()[1].name

    def preprocess(self, frame, size=(640, 640)):
        self.original_shape = frame.shape[:2]  # (h, w)
        img = cv2.resize(frame, size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return img

    def postprocess(self, labels, boxes, scores):
        nms_boxes, nms_scores, nms_labels = [], [], []
        for i in range(len(scores)):
            score_mask = scores[i] > self.conf_thres
            filtered_boxes = boxes[i][score_mask]
            filtered_scores = scores[i][score_mask]
            filtered_labels = labels[i][score_mask]

            if len(filtered_scores) == 0:
                continue

            # Apply multi-class NMS
            nms_boxes, nms_scores, nms_labels = multiclass_nms(
                filtered_boxes, filtered_scores, filtered_labels, iou_threshold=self.iou_thres)

        return nms_boxes, nms_labels, nms_scores

    def __call__(self, frame):
        input_tensor = self.preprocess(frame)
        w, h = self.original_shape[1], self.original_shape[0]
        orig_size = np.array([[w, h]], dtype=np.int64)

        outputs = self.session.run(None, {
            self.input_name: input_tensor,
            self.orig_size_name: orig_size
        })
        labels, boxes, scores = outputs
        nms_boxes, nms_labels, nms_scores = self.postprocess(labels, boxes, scores)
        return nms_boxes, nms_labels, nms_scores