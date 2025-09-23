import cv2
import numpy as np
import onnxruntime as ort


def _nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """ทำ Non-Maximum Suppression สำหรับกรณีที่เป็นคลาสเดียว"""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep = []
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[idx] + areas[rest] - inter
        valid = union > 0
        ious = np.zeros_like(union)
        ious[valid] = inter[valid] / union[valid]

        order = rest[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def multiclass_nms(boxes, scores, labels, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) for multiple classes ด้วย numpy ล้วน"""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels)

    if boxes.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=labels.dtype),
        )

    final_boxes: list[np.ndarray] = []
    final_scores: list[np.ndarray] = []
    final_labels: list[np.ndarray] = []

    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        if cls_boxes.size == 0:
            continue

        keep = _nms_single_class(cls_boxes, cls_scores, iou_threshold)
        if keep.size == 0:
            continue

        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_labels.append(np.full(keep.shape, cls, dtype=labels.dtype))

    if not final_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=labels.dtype),
        )

    selected_boxes = np.concatenate(final_boxes, axis=0)
    selected_scores = np.concatenate(final_scores, axis=0)
    selected_labels = np.concatenate(final_labels, axis=0)

    boxes_xywh = np.empty_like(selected_boxes)
    boxes_xywh[:, 0] = selected_boxes[:, 0]
    boxes_xywh[:, 1] = selected_boxes[:, 1]
    boxes_xywh[:, 2] = selected_boxes[:, 2] - selected_boxes[:, 0]
    boxes_xywh[:, 3] = selected_boxes[:, 3] - selected_boxes[:, 1]

    return boxes_xywh, selected_scores, selected_labels


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
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for label_set, box_set, score_set in zip(labels, boxes, scores):
            score_mask = score_set > self.conf_thres
            if not np.any(score_mask):
                continue

            filtered_boxes = box_set[score_mask]
            filtered_scores = score_set[score_mask]
            filtered_labels = label_set[score_mask]

            selected_boxes, selected_scores, selected_labels = multiclass_nms(
                filtered_boxes, filtered_scores, filtered_labels, iou_threshold=self.iou_thres
            )

            if selected_scores.size == 0:
                continue

            all_boxes.append(selected_boxes)
            all_scores.append(selected_scores)
            all_labels.append(selected_labels)

        if not all_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=labels.dtype if hasattr(labels, "dtype") else np.int64),
                np.empty((0,), dtype=np.float32),
            )

        return (
            np.concatenate(all_boxes, axis=0),
            np.concatenate(all_labels, axis=0),
            np.concatenate(all_scores, axis=0),
        )

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