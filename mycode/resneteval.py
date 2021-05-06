#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.22'
__copyright__ = 'Copyright 2021, PI'


import torch
from mmdet.models.backbones.resnet import ResNet


net = ResNet(depth=50, num_stages=4, out_indices=(0,1,2,3))
net.eval()
inputs = torch.rand(1, 3, 416, 416)
level_outputs = net.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))