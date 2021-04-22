#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.03.26'
__copyright__ = 'Copyright 2021, PI'


from pycocotools.coco import COCO
import cv2
import os
import json
import numpy as np
import random

annfile = '/media/lsa/MobileDisk3/dataset/COCO2017/annotations/instances_train2017.json'
coco = COCO(annotation_file=annfile)

need_cat = ['person']
each_cat_num = 5
category_ids = coco.getCatIds(catNms=need_cat)

tiny_coco = {}
desired_images = []
desired_categories = []
desired_annotations = []

for cat in category_ids:
    c = coco.loadCats(ids=cat)[0]
    desired_categories.append(c)
    img_ids = coco.getImgIds(catIds=cat)
    print('---------')
    for i in range(0, each_cat_num):
        info = coco.loadImgs(img_ids[i])[0]
        if i < 10:
            print(info['file_name'])
        all_ann_in_single_img = coco.getAnnIds(imgIds=info['id'])
        desired_images.append(info)
        img = cv2.imread(filename=os.path.join('/media/lsa/MobileDisk3/dataset/COCO2017/train2017', info['file_name']), flags=cv2.IMREAD_COLOR)
        for ann_id in all_ann_in_single_img:
            single_anno = coco.loadAnns(ids=ann_id)[0]
            if single_anno['category_id'] == cat and single_anno['iscrowd'] == 0:
                desired_annotations.append(single_anno)
                # print(single_anno['bbox'])
#                 cv2.rectangle(img=img, pt1=(int(single_anno['bbox'][0]), int(single_anno['bbox'][1])),
#                               pt2=(int(single_anno['bbox'][0]+single_anno['bbox'][2]), int(single_anno['bbox'][1]+single_anno['bbox'][3])),
#                               color=(0,255,0), thickness=1)
#         cv2.namedWindow('img', cv2.WINDOW_NORMAL)
#         cv2.imshow('img', img)
#         cv2.waitKey(300)
# #
# tiny_coco['info'] = coco.dataset['info']
# tiny_coco['licenses'] = coco.dataset['licenses']
# tiny_coco['images'] = desired_images
# tiny_coco['annotations'] = desired_annotations
# tiny_coco['categories'] = desired_categories
# json.dump(tiny_coco, open('tiny_coco.json', 'w'))