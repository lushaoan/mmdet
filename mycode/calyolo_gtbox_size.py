#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.06.23'
__copyright__ = 'Copyright 2021, PI'


import os
import sys
import json
import numpy as np


data_root = '/media/lsa/MobileDisk3/dataset/PieProject/line/alldata'
all_file = os.listdir(data_root)
all_gt_box_wh = []
for f in all_file:
    if f.endswith('.json'):
        data = json.load(open(os.path.join(data_root, f), 'r'))
        for obj in data['shapes']:
            if obj['shape_type'] == 'rectangle':
                bbox = [int(obj['points'][0][0]), int(obj['points'][0][1]),
                        int(obj['points'][1][0]), int(obj['points'][1][1])]   #xyxy
                all_gt_box_wh.append([bbox[2]-bbox[0], bbox[3]-bbox[1]])

all_gt_box_wh = np.array(all_gt_box_wh)
res = all_gt_box_wh.mean(axis=0)
print(res)
