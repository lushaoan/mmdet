#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2020.10.25'
__copyright__ = 'Copyright 2020, LSA'


import sys
sys.path.append('../')
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/antenna/latest.pth'
checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/tiny_coco/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# print(model)
# test a single image
# img = '/media/lsa/MobileDisk3/dataset/PieProject/antenna/raw/1.png'
img = '/media/lsa/ssdMobileDisk/dataset/COCO2017/train2017/train2017/000000524291.jpg'
result = inference_detector(model, img)

show_result_pyplot(model=model, img=img, result=result, score_thr=0.3, title='result')