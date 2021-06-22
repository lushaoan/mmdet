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


# config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_makeup.py'
# checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup3/latest.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = '/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/8.bmp'
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# result = inference_detector(model, img)   #返回的是原图的xyxy，每个类别
# show_result_pyplot(model=model, img=img, result=result, score_thr=0.5, title='result')


# config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person.py'
# checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/fasterrcnn_tiny_coco_person_cat_car/latest.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = '/media/lsa/ssdMobileDisk/dataset/COCO2017/train2017/train2017/000000262145.jpg'
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# result = inference_detector(model, img)   #返回的是list，每个类别占一个
# show_result_pyplot(model=model, img=img, result=result, score_thr=0.7, title='result')


# config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
# checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection_model/yolov3/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/demo/demo.jpg'
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# result = inference_detector(model, img)
# show_result_pyplot(model=model, img=img, result=result, score_thr=0.7, title='result')

# config_file = '../configs/retinanet/retinanet_r50_fpn_1x_makeup.py'
# checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup_retina/latest.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = '/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/2.bmp'
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# result = inference_detector(model, img)   #返回的是原图的xyxy，每个类别
# show_result_pyplot(model=model, img=img, result=result, score_thr=0.5, title='result')

config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_line.py'
checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/line/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
for i in range(0, 39):
    img_path = f'/media/lsa/MobileDisk3/dataset/PieProject/line/alldata/rightimg_{i}.bmp'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    result = inference_detector(model, img)[0]   #返回的是原图的xyxy，每个类别的分数

    for ele in result:
        if ele[4] > 0.5:
            cv2.rectangle(img=img, pt1=(int(ele[0]), int(ele[1])), pt2=(int(ele[2]), int(ele[3])), thickness=2, color=(0,255,0))

    cv2.imshow('img', img)
    cv2.waitKey(0)
