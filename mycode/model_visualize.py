#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.14'
__copyright__ = 'Copyright 2021, PI'


from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../')
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2

writer = SummaryWriter()
config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_makeup.py'
checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup2/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img_path = '/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/38.bmp'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
model(img)
