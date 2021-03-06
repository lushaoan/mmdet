#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.05.26'
__copyright__ = 'Copyright 2021, PI'


import os
import json
import numpy as np
import einops
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PlugDataset(CustomDataset):
    CLASSES = ('obj',)

    def load_annotations(self, ann_file):
        data_infos = []
        all_files = os.listdir(ann_file)
        img_ids = []
        for file in all_files:
            idx, post_fix = file.split('.')
            if post_fix == 'bmp':
                img_ids.append(idx)
        for img_id in img_ids:
            filename = f'{img_id}.bmp'
            json_path = os.path.join(self.img_prefix, f'{img_id}.json')
            data = json.load(open(json_path))
            del data['imageData']
            data_infos.append(dict(id=img_id, filename=filename, data=data, width=1280, height=1024))

        return data_infos

    def get_ann_info(self, idx):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in self.data_infos[idx]['data']['shapes']:
            if obj['shape_type'] == 'rectangle':
                bbox = [int(obj['points'][0][0]), int(obj['points'][0][1]),
                        int(obj['points'][1][0]), int(obj['points'][1][1])]

                bboxes.append(bbox)
                labels.append(0)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    # def evaluate(self, results, iorThresh=0.5):
