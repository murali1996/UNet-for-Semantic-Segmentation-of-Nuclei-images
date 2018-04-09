# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:05:36 2018

@author: s.jayanthi
"""

import cv2, numpy as np
img = cv2.imread(params.color_transfer_target_label1)
dst = [];
rows,cols = img.shape[0], img.shape[1]
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
for channel in range(img.shape[2]):
    d_img = img[:,:,channel]
    dst.append(cv2.warpAffine(d_img,M,(cols,rows)))
dst = np.stack(dst, axis=-1)
#cv2.imshow('.',img);cv2.imshow('..',dst);

img = cv2.imread(params.here)
cv2.imshow('I', img)
#img_scaled = cv2.resize(img,None,fx=1.5, fy=1.5, interpolation = cv2.INTER_LINEAR)
#cv2.imshow('Scaling - Linear Interpolation', img_scaled)
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
#img_scaled = cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
#cv2.imshow('Scaling - Skewed Size', img_scaled)

img = cv2.imread(params.here_mask)
cv2.imshow('M', img)
#img_scaled = cv2.resize(img,None,fx=1.5, fy=1.5, interpolation = cv2.INTER_LINEAR)
#cv2.imshow('M Scaling - Linear Interpolation', img_scaled)
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('M Scaling - Cubic Interpolation', img_scaled)
img_scaled = cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('M Scaling - Skewed Size', img_scaled)