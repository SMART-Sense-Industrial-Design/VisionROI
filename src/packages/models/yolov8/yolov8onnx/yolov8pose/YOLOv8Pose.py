import os
import numpy as np
import cv2
import onnxruntime
from typing import List, Tuple, Dict
import base64

from models.common.box_ops import compute_iou, xywh2xyxy
# from utils import file_decode

class YOLOv8Pose:
    ''' yolov8-keypoints onnxruntime inference
    '''

    def __init__(self,
                 onnx_path: str,
                 input_size: Tuple[int],
                 box_score=0.25,
                 kpt_score=0.5,
                 nms_thr=0.2
                 ) -> None:
        assert onnx_path.endswith('.onnx'), f"invalid onnx model: {onnx_path}"
        assert os.path.exists(onnx_path), f"model not found: {onnx_path}"
        # binary_str = file_decode(onnx_path)
        self.sess = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        # print("input info: ", self.sess.get_inputs()[0])
        # print("output info: ", self.sess.get_outputs()[0])
        self.input_size = input_size
        self.box_score = box_score
        self.kpt_score = kpt_score
        self.nms_thr = nms_thr

    def __call__(self, image):
        return self.detect(image)

    def _preprocess(self, img: np.ndarray):
        ''' preprocess image for model inference
        '''
        input_w, input_h = self.input_size
        if len(img.shape) == 3:
            padded_img = np.ones((input_w, input_h, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114
        r = min(input_w / img.shape[0], input_h / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        # (H, W, C) BGR -> (C, H, W) RGB
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def _postprocess(self, output: List[np.ndarray], ratio) -> Dict:
        predict = output[0].squeeze(0).T
        predict = predict[predict[:, 4] > self.box_score, :]
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / ratio
        boxes = xywh2xyxy(boxes)
        kpts = predict[:, 5:]
        # print(kpts)
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:
                    kpts[i, 3*j: 3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j] /= ratio
                    kpts[i, 3*j+1] /= ratio 
        idxes = nms_process(boxes, scores, self.nms_thr)
        result = {'boxes': boxes[idxes,: ].astype(int).tolist(),
                  'kpts': kpts[idxes,: ].astype(float).tolist(),
                  'scores': scores[idxes].tolist()}
        return result

    def detect(self, img: np.ndarray) -> Dict:
        img, ratio = self._preprocess(img)
        ort_input = {self.sess.get_inputs()[0].name: img[None, :]/255}
        output = self.sess.run(None, ort_input)
        result = self._postprocess(output, ratio)
        boxes, scores, kpts = np.array(result['boxes']), np.array(result['scores']), np.array(result['kpts'])
        return boxes, scores, kpts
    
    
    def draw_result(self, img: np.ndarray, result: Dict, with_label=False, object_detect_list=[]) -> np.ndarray:
        original_img = img.copy()
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        for box, kpt, score in zip(boxes, kpts, scores):
            # result_fall = yolo_module.human_fall(box, kpt, score)
            x1, y1, x2, y2 = box

            crop_img = original_img[y1:y2, x1:x2]
            ret, jpeg = cv2.imencode('.jpg', crop_img)
            img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            # object_detect_list.append([img_b64, result_fall, score])

            label_str = "{:.0f}%".format(score*100)
            label_size, baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            if with_label:
                cv2.rectangle(img, (x1, y1), (x1+label_size[0], y1+label_size[1]+baseline),
                    (0, 0, 255), -1)
                cv2.putText(img, label_str, (x1, y1+label_size[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
            for idx in range(len(kpt) // 3):
                # print(idx)
                x, y, score = kpt[3*idx: 3*(idx+1)]
                if score > 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    center_point = (int(x), int(y))
                    fontScale = 0.5
                    color = (0, 255, 255)
                    thickness = 1
                    name_point = list(KEYPOINT_DICT.keys())[idx]
                    cv2.putText(img, '{}'.format(idx), center_point, font, fontScale, color, thickness, cv2.LINE_AA)
                    cv2.circle(img, (int(x), int(y)), 3, COLOR_LIST[idx], -1)

            # plot skeleton
            palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
            palette_fall = np.array([[0, 0, 255], [0, 0, 255], [0, 0, 255],
                    [0, 0, 255], [0, 0, 255], [0, 0, 255],
                    [0, 0, 255], [0, 0, 255], [0, 0, 255],
                    [0, 0, 255], [0, 0, 255], [0, 0, 255],
                    [0, 0, 255], [0, 0, 255], [0, 0, 255],
                    [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255],
                    [0, 0, 255]])
            skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

            steps = 3
            pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
            
            for sk_id, sk in enumerate(skeleton):
                r, g, b = pose_limb_color[sk_id]
                # print(sk)
                kpts_51 = kpt
                pos1 = (int(kpts_51[(sk[0] - 1) * steps]), int(kpts_51[(sk[0] - 1) * steps + 1]))
                pos2 = (int(kpts_51[(sk[1] - 1) * steps]), int(kpts_51[(sk[1] - 1) * steps + 1]))
                conf1 = kpts_51[(sk[0] - 1) * steps + 2]
                conf2 = kpts_51[(sk[1] - 1) * steps + 2]
                if conf1 > 0.5 and conf2 > 0.5:  # For a limb, both the keypoint confidence must be greater than 0.5
                    cv2.line(img, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
            
        return img

COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                   [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                   [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def nms_process(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    sorted_idx = np.argsort(scores)[::-1]
    keep_idx = []
    while sorted_idx.size > 0:
        idx = sorted_idx[0]
        keep_idx.append(idx)
        ious = compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
        rest_idx = np.where(ious < iou_thr)[0]
        sorted_idx = sorted_idx[rest_idx+1]
    return keep_idx

