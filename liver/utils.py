import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
from functools import reduce
import math



def sortPoints(cnt):
    new_cnt = []
    start = cnt[0]
    del cnt[0]
    new_cnt.append(start)

    while len(cnt)!=1:
        dist = [np.linalg.norm(start - i) for i in cnt]
        x = dist.index(min(dist))
        start = cnt[x]
        del cnt[x]
        new_cnt.append(start)
    new_cnt.append(cnt[0])
    return new_cnt

def greenandred(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv, (40, 70, 30), (255, 255,255))
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask_orange = cv2.inRange(hsv, (10, 100, 20), (25, 255, 255))
    mask_yellow = cv2.inRange(hsv, (21, 39, 64), (40, 255, 255))

    ## slice the red and orange
    imask_red1 = mask_red1>0
    imask_red2 = mask_red2>0
    imask_orange = mask_orange>0
    imask_yellow = mask_yellow>0
    red = np.zeros_like(img, np.uint8)
    red[imask_red1] = img[imask_red1]
    #red[imask_red2] = img[imask_red2]
    #red[imask_orange] = img[imask_orange]
    #red[imask_yellow] = img[imask_yellow]

    ## slice the green
    imask_green = mask_green>0
    green = np.zeros_like(img, np.uint8)
    green[imask_green] = img[imask_green]
    '''
    gray = cv2.cvtColor(red,cv2.COLOR_BGR2GRAY)
    kernel_size = 13
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    (cnt, hierarchy) = cv2.findContours(
        blur_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    new_cnt = []

    for i in cnt:
        area = cv2.contourArea(i)
        if area>200:
            new_cnt.append(i)

    cv2.drawContours(img, new_cnt, -1, (0, 255, 0), 2)
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    '''
    ## save
    # 
    return green,red

def find(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    #cv2.imwrite("gray2.jpg", gray)
    (cnt, hierarchy) = cv2.findContours(
        blur_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    new_cnt = []
    new_img = img
    #cv2.drawContours(img, cnt, -1, (255, 255, 255), 5)
    #cv2.imwrite("contours.jpg", img)
    for i in cnt:
        area = cv2.contourArea(i)
        if area>200:
            new_cnt.append(i)
    #cv2.drawContours(new_img, new_cnt, -1, (255, 255, 255), 5)
    #cv2.imwrite("contours_new.jpg", new_img)

    return new_cnt
def inorout(cnt,cnt_mean,mask):
    new_cnt=[]
    new_cnt_mean=[]
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    for idx in range(len(cnt_mean)):
        #a = np.asanyarray([255,255,255],dtype=np.uint8)
        x = cnt_mean[idx][0][0]
        y = cnt_mean[idx][0][1]
        #print(gray[y][x])
        if gray[y][x]==255:
            new_cnt_mean.append(cnt_mean[idx])
            new_cnt.append(cnt[idx])
    return new_cnt,new_cnt_mean