import cv2
import numpy as np
from utils import sortPoints,greenandred,find,inorout
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mix_dir')
args = parser.parse_args()

if __name__ == '__main__':
    img = cv2.imread(args.mix_dir)
    img1 = cv2.imread(args.mix_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, th1 = cv2.threshold(gray, 249, 250, cv2.THRESH_BINARY)

    #gray = cv2.cvtColor(th1,cv2.COLOR_BGR2GRAY)
    kernel_size = 13
    blur_gray = cv2.GaussianBlur(th1,(kernel_size,kernel_size),0)
    (cnt, hierarchy) = cv2.findContours(
        blur_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_cnt=[]
    for i in cnt:
        mean = np.mean(i,axis=0).astype(np.int32)
        if (mean[0][0]<img.shape[0]*0.95 or mean[0][1]<img.shape[1]*0.95):
            new_cnt.append(mean[0])
            #cv2.circle(img, mean[0], 1, (0, 255, 0), 20)
    new_cnt = sortPoints(new_cnt)
    new_cnt = np.array(new_cnt).astype(np.int32)
    #cv2.drawContours(th1, new_cnt, -1, (255, 255, 255), 10)

    mask = np.zeros_like(img,dtype=np.uint8)

    cv2.fillPoly(mask,[new_cnt],(255,255,255))
    #cv2.imwrite("mask.jpg", mask)

    green,red = greenandred(img)

    green_cnt = find(green)
    green_mean = [np.mean(i,axis=0).astype(np.int32) for i in green_cnt]
    new_green_cnt,new_green_cnt_mean = inorout(green_cnt,green_mean,mask)
    green_mean = np.array(green_mean).astype(np.int32)
    new_green_cnt_mean = np.array(new_green_cnt_mean).astype(np.int32)
    green_mean = [np.mean(i,axis=0).astype(np.int32) for i in green_cnt]

    red_cnt = find(red)
    red_mean = [np.mean(i,axis=0).astype(np.int32) for i in red_cnt]
    red_mean = np.array(red_mean).astype(np.int32)
    new_red_cnt,new_red_cnt_mean = inorout(red_cnt,red_mean,mask)
    red_mean = np.array(red_mean).astype(np.int32)
    new_red_cnt_mean = np.array(new_red_cnt_mean).astype(np.int32)
    print(len(new_red_cnt_mean),len(new_green_cnt_mean))

    cv2.drawContours(red, new_red_cnt, -1, (255, 255, 255), 5)
    cv2.drawContours(img1, new_red_cnt, -1, (255, 255, 255), 5)
    cv2.drawContours(green, new_green_cnt, -1, (255, 255, 255), 5)
    cv2.drawContours(img, new_green_cnt, -1, (255, 255, 255), 5)
    cv2.imwrite("result/red.jpg", red)
    cv2.imwrite("result/red1.jpg", img1)
    cv2.imwrite("result/green.jpg", green)
    cv2.imwrite("result/green1.jpg", img)
    #cv2.imwrite("mask.jpg", mask)
    #cv2.imwrite("white.jpg", th1)
    
    #cv2.imwrite("gray.jpg", img)
    