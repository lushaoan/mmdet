#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.14'
__copyright__ = 'Copyright 2021, PI'


# 记得指定--shape
# python tools/deployment/pytorch2onnx.py ./configs/yolo/yolov3_d53_mstrain-608_273e_makeup.py
#                                         /media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup2/latest.pth
#                                         --output-file work_dir/makeup2/makeup.onnx
#                                         --input-img /media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/38.bmp
#                                         --shape 512 608


import cv2
import numpy as np
import onnxruntime
import time
import torch
from mmdet.datasets.pipelines import Compose
from einops import rearrange


# onnx_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/makeup3/makeup.onnx'
# img_path = '/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear/8.bmp'
#
# # onnx_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/plug/plug.onnx'
# # img_path = '/media/lsa/MobileDisk3/dataset/PieProject/plug/alldata/rightImg_0.bmp'
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#
# img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(608, 608),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', mean=[0, 0, 0], std=[255., 255., 255.]),
#             dict(type='Pad', size_divisor=32),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# pipeline = Compose(test_pipeline)
# data = pipeline(dict(img_info=dict(filename=img_path), img_prefix=None))
# img_np = data['img'][0].data.numpy()
#
# session = onnxruntime.InferenceSession(onnx_file)
#
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
#
# res = session.run([output_name], {input_name: rearrange(img_np, 'c h w -> 1 c h w')})[0][0]
# print(res.shape)
# valid_obj_id = res[:, 4] > 0.1
# found = res[valid_obj_id]


onnx_file = '/media/rr/ssdMobileDisk/open-mmlab/mmdetection/work_dir/line/line.onnx'
img_path = '/media/rr/MobileDisk3/dataset/PieProject/line/alldata/left_0.bmp'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img = cv2.resize(src=img, dsize=(608, 512))
img = cv2.resize(src=img, dsize=(608, 608))

img_np = np.array([np.float32(img) / 255.]).transpose(0, 3, 1, 2) #以前的失败了应该是这里引起的，成功的时候
                                                                  #config.py: img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
                                                                   #这里记得要做同样的处理
session = onnxruntime.InferenceSession(onnx_file)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
res = session.run([output_name], {input_name: img_np})[0][0]
valid_obj_id = res[:, 4] > 0.5
found = res[valid_obj_id]

for obj in found:
    cv2.rectangle(img=img, pt1=(int(obj[0]), int(obj[1])), pt2=(int(obj[2]), int(obj[3])), color=(0,255,0), thickness=1)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)