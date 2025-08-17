import cv2
import numpy as np
from utils import sortPoints,sort_coordinates
green_lower = np.array([36, 25, 25], np.uint8)
green_upper = np.array([100, 255, 255], np.uint8)
img = cv2.imread('2.jpg')
hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsvFrame, green_lower, green_upper)             
output = cv2.bitwise_and(img, img, mask = mask ) 
#output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
new_contours=[]
for contour in contours:
    area = cv2.contourArea(contour)
    if area>10:
        new_contours.append(contour)

allcon = np.vstack(new_contours)

epli = 0.0005*cv2.arcLength(allcon,True)
appr = cv2.approxPolyDP(allcon,epli,True)
cnt=[]
for i in appr:
    cnt.append(i)
indice = sort_coordinates(appr)
ind = []
for i in indice:
    ind.append(i.reshape(-1,2))
new_cnt = sortPoints(cnt)
new_cnt = np.array(new_cnt).astype(np.int32)
ind = np.vstack(ind).reshape(-1,1,2)
#output = cv2.drawContours(output,allcon,-1,[0,255,0])
output = cv2.drawContours(output,[appr],-1,[0,255,0])
#mask = np.zeros_like(img,dtype=np.uint8)

#cv2.fillPoly(mask,[new_cnt],(255,255,255))
# 循環遍歷所有輪廓
for contour in contours:
    # 計算包圍輪廓的最小矩形
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    length = len(hull)
    #for i in range(len(hull)):
        #cv2.line(output, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 2)
    #output = cv2.drawContours(output,contour,-1,[0,255,0])
    # 繪製矩形框
    #cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 輸出寬度和高度
    print(f"寬度: {w}, 高度: {h}")
cv2.imwrite('output.jpg', output)
cv2.waitKey(0)                                   
cv2.destroyAllWindows()