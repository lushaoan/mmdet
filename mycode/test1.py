#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2020.10.25'
__copyright__ = 'Copyright 2020, LSA'


import sys
sys.path.append('../')
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv

config_file = '../configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/media/lsa/MobileDisk2/open-mmlab/mmdetection_model/yolov3/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# print(model)
# test a single image
img = 'demo.jpg'
result = inference_detector(model, img)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")