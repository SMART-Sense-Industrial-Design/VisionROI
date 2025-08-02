# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
# from tflite_runtime import interpreter as tflite
import tensorflow as tf
from .ops import *
import torch


class Yolov8TFLite:
    def __init__(self, model_path, conf_thres=0.7, iou_thres=0.5):

        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

    def detect_objects(self, frame, conf):
        orig_imgs = [frame]

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']

        # TODO check shape of input_shape and frame.shape
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        _, w, h, _ = input_shape

        # check width and height
        if frame.shape[0] != h or frame.shape[1] != w:
            input_img = cv2.resize(frame, (w, h))
        else:
            input_img = frame
        input_img = input_img[np.newaxis, ...]  # add batch dim
        input_img = input_img.astype(np.float32) / 255.  # change to float img
        self.interpreter.set_tensor(input_details[0]['index'], input_img)

        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(output_details[0]['index'])
        
        ######################################################################
        # borrowed from ultralytics\models\yolo\detect\predict.py #postprocess

        # convert to torch to use ops.non_max_suppression
        # ultralytics is working on none-deeplearning based non_max_suppression
        # https://github.com/ultralytics/ultralytics/issues/1777
        # maybe someday, but for now, just workaround
        preds = torch.from_numpy(preds)
        preds = non_max_suppression(preds,
                                        conf,
                                        0.7,  # todo, make into arg
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)  # hack. just copied values from execution of yolov8n.pt

        # results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]

            # tflite result are in [0, 1]
            # scale them by width (w == h)
            pred[:, :4] *= w

            pred[:, :4] = scale_boxes(input_img.shape[1:], pred[:, :4], orig_img.shape)
            # img_path = ""
            # results.append(Results(orig_img, path=img_path, names=yolo_default_label_names, boxes=pred))
            boxes = pred[:, :4]
            scores = pred[:, -2]
            class_ids = pred[:, -1]

        return boxes, scores, class_ids
    
    def __call__(self, image, conf=0.5):
        return self.detect_objects(image, conf)

    

