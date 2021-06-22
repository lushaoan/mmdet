#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.06.21'
__copyright__ = 'Copyright 2021, PI'


import sys
sys.path.append('../')
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
from pymvcam.pymvcam import MVCam
import numpy as np
import onnxruntime
import torch
import copy
from einops import rearrange, reduce


if __name__ == '__main__':
    config_file = '../configs/yolo/yolov3_d53_mstrain-608_273e_line.py'
    checkpoint_file = '/media/pi/ssdMobileDisk/open-mmlab/mmdetection/work_dir/line/latest.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')

    leftcam = MVCam(index=0)
    leftcam.start()
    leftcam.setAeState(False)
    leftcam.setAnalogGain(64)
    leftcam.setExposureTime(100000)
    leftcam.setContrast(100)
    leftcam.setGamma(100)

    rightcam = MVCam(index=1)
    rightcam.start()
    rightcam.setAeState(False)
    rightcam.setAnalogGain(64)
    rightcam.setExposureTime(100000)
    rightcam.setContrast(100)
    rightcam.setGamma(100)
    stop = False

    cv2.namedWindow('left_img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('right_img', cv2.WINDOW_NORMAL)
    while not stop:
        left_image = leftcam.readImage()
        right_image = rightcam.readImage()
        left_image = cv2.cvtColor(src=left_image, code=cv2.COLOR_GRAY2BGR)
        right_image = cv2.cvtColor(src=right_image, code=cv2.COLOR_GRAY2BGR)
        left_result = inference_detector(model, left_image)[0]
        right_result = inference_detector(model, right_image)[0]

        for ele in left_result:
            if ele[4] > 0.5:
                cv2.rectangle(img=left_image, pt1=(int(ele[0]), int(ele[1])), pt2=(int(ele[2]), int(ele[3])), thickness=2, color=(0,255,0))

        for ele in right_result:
            if ele[4] > 0.5:
                cv2.rectangle(img=right_image, pt1=(int(ele[0]), int(ele[1])), pt2=(int(ele[2]), int(ele[3])), thickness=2, color=(0,255,0))

        cv2.imshow('left_img', left_image)
        cv2.imshow('right_img', right_image)

        key = cv2.waitKey(50)
        if key == 113:
            stop = True

    cv2.destroyAllWindows()
    