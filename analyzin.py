# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 15:53:36 2018

@author: s.jayanthi
"""
import os, cv2, numpy as np


file = '8f27ebc74164eddfe989a98a754dcf5a9c85ef599a1321de24bcf097df1814ca.png'
path = '.'

path = os.path.join(os.pardir,'data_gen/m2/clusters/1')
path = os.path.join(os.pardir,'stain_norm/Peter554/data/source')
h_m, h_s, s_m, s_s, v_m, v_s = [], [], [], [], [], [];
for file in os.listdir(path):
    if file[-8:-4]!='mask' and file[-3:]=='png':
        image = cv2.imread(os.path.join(path,file));
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(image_hsv);
        h_m.append(np.mean(h));s_m.append(np.mean(s));v_m.append(np.mean(v));
        h_s.append(np.std(h));s_s.append(np.std(s));v_s.append(np.std(v));
        cv2.imshow('HSV',np.hstack((h,s,v)));
        cv2.imshow('RGB',np.hstack((cv2.split(image))));
        cv2.imshow('YCrCb',np.hstack((cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)))));
        cv2.waitKey(0);


path = os.path.join(os.pardir,'stain_norm/Peter554/data/source')
th_m, th_s, ts_m, ts_s, tv_m, tv_s = [], [], [], [], [], [];
for file in os.listdir(path):
    if file[-8:-4]!='mask':
        try:
            image = cv2.imread(os.path.join(path,file));
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(image_hsv);
            th_m.append(np.mean(h));ts_m.append(np.mean(s));tv_m.append(np.mean(v));
            th_s.append(np.std(h));ts_s.append(np.std(s));tv_s.append(np.std(v));
        except:
            continue;



