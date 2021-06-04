#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.06.02'
__copyright__ = 'Copyright 2021, PI'


import torch
from mmdet.apis import init_detector
from mmdet.core.export import (build_model_from_cfg,
                               generate_inputs_and_wrap_model,
                               preprocess_example_input)
import numpy as np
import cv2


config_file = '../../configs/yolo/yolov3_d53_mstrain-608_273e_plug.py'
checkpoint_file = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/plug/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

input_size = [1,3,512,608]
one_img = torch.randn(input_size)
input_config = {
        'input_shape': (1, 3, 512, 608),
        'input_path': '/media/lsa/MobileDisk3/dataset/PieProject/plug/alldata/leftImg_0.bmp',
        'normalize_cfg': {'mean': [0,0,0], 'std': [255,255,255]}
    }

modelp, tensor_data = generate_inputs_and_wrap_model(
        config_file, checkpoint_file, input_config, cfg_options=None)

torch.onnx.export(model, tensor_data, './test.onnx')