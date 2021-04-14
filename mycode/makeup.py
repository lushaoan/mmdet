#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.04.13'
__copyright__ = 'Copyright 2021, PI'


from mmdet.datasets.makeup import MakeupDataset


if __name__ == '__main__':
    my_dataset = MakeupDataset(ann_file='/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear',
                               pipeline=[],
                               img_prefix='/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/clear')

    print(len(my_dataset.CLASSES))