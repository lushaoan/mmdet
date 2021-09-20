#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.08.19'
__copyright__ = 'Copyright 2021, PI'

import torch

from mmdet.core.export import build_model_from_cfg, preprocess_example_input
import numpy as np
from functools import partial
from torch2trt import torch2trt


if __name__ == '__main__':
    config_path = '../../configs/centernet/centernet_resnet18_140e_tinycoco.py'
    checkpoint_path = '../../work_dir/centernet/cocotiny5_1/latest.pth'
    model = build_model_from_cfg(config_path=config_path,
                                 checkpoint_path=checkpoint_path)
    input_img = np.random.random((512, 512, 3))
    input_config = {
        'input_shape': (1, 3, 512, 512),
        'input_path': input_img,
        'normalize_cfg': {'mean': (123.675, 116.28, 103.53),
                          'std': (58.395, 57.12, 57.375)}
    }
    # prepare input
    one_img, one_meta = preprocess_example_input(input_config)
    one_meta['border'] = np.array([11., 438., 16., 656.])
    one_meta['scale_factor'] = np.array([1., 1., 1., 1.])
    img_list, img_meta_list = [one_img], [[one_meta]]
    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=False)
    x = torch.ones((1,3,512,512)).cuda()
    model_trt = torch2trt(model, img_list)
