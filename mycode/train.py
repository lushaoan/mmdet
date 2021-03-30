#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.03.27'
__copyright__ = 'Copyright 2021, LSA'

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

if __name__ == '__main__':
    # cfg = Config.fromfile('../configs/rpn/rpn_r50_fpn_1x_coco_toy.py')
    # cfg.work_dir = '/media/lsa/ssdMobileDisk/open-mmlab/mmdetection/work_dir/coco_toy_rpn'
    cfg = Config.fromfile('../configs/yolo/yolov3_d53_mstrain-608_273e_coco.py')
    cfg.work_dir = '../work_dir/tiny_coco'
    cfg.gpu_ids = range(1)
    cfg.seed = None

    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False)
