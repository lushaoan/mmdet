#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.20'
__copyright__ = 'Copyright 2021, PI'


from mmdet.datasets.pipelines.loading import LoadImageFromFile
from mmdet.datasets.pipelines.transforms import Resize


data_result = {}
data_result['img_info'] = {}
data_result['img_prefix'] = None
data_result['img_info']['filename'] = '/media/lsa/MobileDisk3/dataset/nut/images/0.jpg'
loader = LoadImageFromFile(to_float32=True)
load_result = loader(data_result)

resizer = Resize(img_scale=[(608, 608)], keep_ratio=True) #当scale给两个tuple 如img_scale=[(320, 320),(608, 608)]的时候，其实不是说最终是这两个size，
                                                          # 而是会在这个范围里面random一些大小出来，若keep ratio，
                                                          # 则会对random出来的size再裁剪一下，一般是长边不变
                                                          # 若给的是一组数，比如(608, 608)，若keep ratio=True，则会长边是608,
                                                          # 短边保持比例缩放
resizer_result = resizer(load_result)
pass