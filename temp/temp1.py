#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.03.27'
__copyright__ = 'Copyright 2021, LSA'

import cv2

img = cv2.imread('/media/lsa/MobileDisk3/dataset/Makeup/bottom/raw/raw/7.bmp')
reimg = cv2.resize(img, (416, 320))
cv2.imwrite('temp.bmp', reimg)