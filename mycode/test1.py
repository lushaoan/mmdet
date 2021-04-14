#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2020.10.25'
__copyright__ = 'Copyright 2020, LSA'


import sys
sys.path.append('../')
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2

# config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person.py'
config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_makeup.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/antenna/latest.pth'
checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup2/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# print(model)
# test a single image
# img = '/media/lsa/MobileDisk3/dataset/PieProject/antenna/raw/1.png'
img_path = '/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/38.bmp'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
result = inference_detector(model, img)   #返回的是原图的xyxy，每个类别

show_result_pyplot(model=model, img=img, result=result, score_thr=0.7, title='result')