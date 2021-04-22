#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.14'
__copyright__ = 'Copyright 2021, PI'


# python tools/deployment/pytorch2onnx.py ./configs/yolo/yolov3_d53_mstrain-608_273e_makeup.py
#                                         /media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup2/latest.pth
#                                         --output-file work_dir/makeup2/makeup.onnx
#                                         --input-img /media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/38.bmp


import cv2
import numpy as np
import onnxruntime
import time


onnx_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup3/makeup.onnx'
img_path = '/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/8.bmp'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(src=img, dsize=(608, 608))

img_np = np.array([np.float32(img) / 255.]).transpose(0, 3, 1, 2)
session = onnxruntime.InferenceSession(onnx_file)
res = session.run(['boxes'], {'input': img_np})[0]
valid_obj_id = res[:, 4] > 0.5
found = res[valid_obj_id]

# onnx_file = '../work_dir/yolov3_sample.onnx'
# img_path = '../demo/demo.jpg'
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img = cv2.resize(src=img, dsize=(608, 608))
#
# img_np = np.array([np.float32(img) / 255.]).transpose(0, 3, 1, 2)
# session = onnxruntime.InferenceSession(onnx_file)
# res = session.run(['boxes'], {'input': img_np})[0]
# valid_obj_id = res[:, 4] > 0.7
# found = res[valid_obj_id]
#
for obj in found:
    cv2.rectangle(img=img, pt1=(int(obj[0]), int(obj[1])), pt2=(int(obj[2]), int(obj[3])), color=(0,255,0), thickness=1)

cv2.imshow('img', img)
cv2.waitKey(0)