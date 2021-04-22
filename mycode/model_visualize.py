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

from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../')
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import numpy as np
import onnxruntime
import time

writer = SummaryWriter(log_dir='/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup2/model')
config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_makeup.py'
checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup2/latest.pth'

onnx_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup2/makeup.onnx'
# writer.add_onnx_graph(onnx_file)

img_path = '/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/38.bmp'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
session = onnxruntime.InferenceSession(onnx_file)
for i in session.get_inputs():
    print(i.name)

print('---')

for o in session.get_outputs():
    print(o.name)

img = cv2.resize(src=img, dsize=(1216, 800))
img_np = np.array([np.float32(img) / 255.]).transpose(0, 3, 1, 2)
t0 = time.time()
res = session.run(['boxes'], {'input': img_np})[0]
print(res)
valid_obj = res[:, 4] > 0.7
print(valid_obj)