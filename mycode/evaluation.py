#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.06.24'
__copyright__ = 'Copyright 2021, PI'


# python tools/test.py configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
#                      ../mmdetection_model/yolov3/yolov3_d53_mstrain-608_273e_coco-139f5633.pth
#                      --out results.pkl
#                      --eval mAP   #eval能有哪些key，由对应的dataset的evaluate()决定

# python tools/analysis_tools/analyze_results.py configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
#                                                results.pkl
#                                                work_dir/