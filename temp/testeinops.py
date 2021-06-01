#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.05.25'
__copyright__ = 'Copyright 2021, PI'


import numpy as np
import cv2
from einops import rearrange, repeat
import torch

imgs = np.load('./test_images.npy')
print(imgs.shape)

new = repeat(imgs[0], 'h w c -> (h 2) (w 2) c')

cv2.imshow('img', imgs[0])
cv2.imshow('new', new)
cv2.waitKey(0)