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


# config_file = '../configs/centernet/centernet_resnet18_140e_tinycoco.py'
# checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/centernet/line/latest.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = '/media/lsa/MobileDisk3/dataset/PieProject/line/alldata/left_2.bmp'
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# result = inference_detector(model, img)   #返回的是原图的xyxy，每个类别
# show_result_pyplot(model=model, img=img, result=result, score_thr=0.5, title='result')

config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_line2.py'
checkpoint_file = '../work_dir/line/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
for i in range(0, 49):
    img_path = f'/media/lsa/MobileDisk3/dataset/PieProject/line/alldata/right_{i}.bmp'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    result = inference_detector(model, img)[0]   #返回的是原图的xyxy，每个类别的分数

    for ele in result:
        if ele[4] > 0.5:
            cv2.rectangle(img=img, pt1=(int(ele[0]), int(ele[1])), pt2=(int(ele[2]), int(ele[3])), thickness=2, color=(0,255,0))

    cv2.imshow('img', img)
    cv2.waitKey(0)
