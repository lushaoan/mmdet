#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.13'
__copyright__ = 'Copyright 2021, PI'


import torch
from datasets.plug import PlugDataset
from mmdet.core.bbox.iou_calculators import bbox_overlaps


if __name__ == '__main__':
    # my_dataset = MakeupDataset(ann_file='/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear',
    #                            pipeline=[],
    #                            img_prefix='/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear')
    #
    # print(len(my_dataset.CLASSES))

    plugdataset = PlugDataset(ann_file='/media/lsa/MobileDisk3/dataset/PieProject/plug/alldata',
                              pipeline=[],
                              img_prefix='/media/lsa/MobileDisk3/dataset/PieProject/plug/alldata')

    a = plugdataset[0]
    pass

    # bboxes1 = torch.FloatTensor([[0, 0, 10, 10],
    #                              [10, 10, 20, 20],
    #                              [32, 32, 38, 42]])
    # bboxes2 = torch.FloatTensor([[0, 0, 10, 20],
    #                              [0, 10, 10, 19]])
    # overlaps = bbox_overlaps(bboxes1, bboxes2)
    # print(overlaps)