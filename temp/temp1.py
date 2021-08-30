#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.05.26'
__copyright__ = 'Copyright 2021, PI'


from pymvcam import MVCam
import cv2

leftcam = MVCam(index=0)

leftcam.start()
leftcam.setAeState(False)
leftcam.setAnalogGain(4)
leftcam.setExposureTime(18000)
leftcam.setContrast(100)
leftcam.setGamma(100)

count = 0
stop = False
cv2.namedWindow('limg', cv2.WINDOW_NORMAL)

while not stop:
    l_img = leftcam.readImage()

    cv2.imshow('limg', l_img)
    key = cv2.waitKey(50)
    if key == 113: # q
        stop = True
    elif key == 116:
        cv2.imwrite(f'/media/lsa/MobileDisk3/dataset/PieProject/line3/{count}.bmp', l_img)
        count += 1

leftcam.release()
cv2.destroyAllWindows()