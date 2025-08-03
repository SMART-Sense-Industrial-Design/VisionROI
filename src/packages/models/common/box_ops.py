import numpy as np


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single box and an array of boxes."""
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Perform Non-Maximum Suppression."""
    sorted_indices = np.argsort(scores)[::-1]
    keep = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        remaining = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[remaining + 1]
    return keep


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float) -> list[int]:
    """Apply NMS separately per class and return kept indices."""
    keep = []
    for cid in np.unique(class_ids):
        cls_inds = np.where(class_ids == cid)[0]
        cls_boxes = boxes[cls_inds, :]
        cls_scores = scores[cls_inds]
        cls_keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep.extend(cls_inds[cls_keep])
    return keep


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert boxes from center-based ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
