#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.20'
__copyright__ = 'Copyright 2021, PI'


import numpy as np
import cv2
import torch
from mmdet.models.backbones.darknet import Darknet
from mmdet.models.necks.yolo_neck import YOLOV3Neck
from mmdet.models.dense_heads.yolo_head import YOLOV3Head
from mmdet.core.anchor.anchor_generator import YOLOAnchorGenerator


# yolo_backbone = Darknet(53)            #yolo v3已经返回(y1, y2, y3)三种大小的featuremap
# yolo_backbone.eval()
# inputs = torch.rand(1,3,416,416)
# back_outputs = yolo_backbone.forward(inputs)
# # for level_out in back_outputs:
# #     print(tuple(level_out.shape))
# yolo_neck = YOLOV3Neck(num_scales=3, in_channels=[1024, 512, 256], out_channels=[512, 256, 128])
# yolo_neck.eval()
# neck_outputs = yolo_neck(back_outputs)
# # for lay in neck_outputs:
# #     print(tuple(lay.shape))
# yolo_head = YOLOV3Head(num_classes=5, in_channels=[512, 256, 128])
# yolo_head.eval()
# head_outputs = yolo_head(neck_outputs)

anchor_generator = YOLOAnchorGenerator(base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                                   [(30, 61), (62, 45), (59, 119)],
                                                   [(10, 13), (16, 30), (33, 23)]],
                                       strides=[32, 16, 8])
all_anchors = anchor_generator.grid_anchors(featmap_sizes=[(13, 13), (26, 26), (52, 52)], device='cpu')

for anchors in all_anchors:
    for anchor in anchors:
        canvas = np.zeros(shape=(416, 416), dtype=np.uint8)
        pt = anchor.numpy()
        cv2.rectangle(img=canvas, pt1=(int(pt[0]), int(pt[1])), pt2=(int(pt[2]), int(pt[3])), color=255, thickness=1)
        cv2.imshow('img', canvas)
        cv2.waitKey(100)