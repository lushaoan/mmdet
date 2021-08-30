#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.08.16'
__copyright__ = 'Copyright 2021, PI'


import torch
import onnx

from mmcv.tensorrt import (TRTWrapper, onnx2trt, save_trt_engine,
                           is_tensorrt_plugin_loaded)
import tensorrt as trt


assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

onnx_file = '../../checkpoints/yolo/yolov3.onnx'
trt_file = 'sample.trt'
onnx_model = onnx.load(onnx_file)

# Model input
inputs = torch.rand(1, 3, 608, 608).cuda()
# Model input shape info
opt_shape_dict = {
    'input': [list(inputs.shape),
              list(inputs.shape),
              list(inputs.shape)]
}

# Create TensorRT engine
max_workspace_size = 1 << 30
trt_engine = onnx2trt(
    onnx_model,
    opt_shape_dict,
    max_workspace_size=max_workspace_size)

# Save TensorRT engine
save_trt_engine(trt_engine, trt_file)

# Run inference with TensorRT
trt_model = TRTWrapper(trt_file, ['input'], ['output'])

with torch.no_grad():
    trt_outputs = trt_model({'input': inputs})
    output = trt_outputs['output']