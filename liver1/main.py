import cv2
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mix_dir')
args = parser.parse_args()

if __name__ == '__main__':
    green_lower = np.array([36, 25, 25], np.uint8)
    green_upper = np.array([100, 255, 255], np.uint8)
    img = cv2.imread(args.mix_dir)
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
        if area>20:
            new_contours.append(contour)

    allcon = np.vstack(new_contours)
    allcon1 = sorted(allcon , key=lambda k: [k[0][0], k[0][1]])
    
    x=0
    y1=0
    y2=0
    n_list=[]
    for i in allcon1:
        if x==0 and y1==0:
            x = i[0][0]
            y1 = i[0][1]
            y2 = i[0][1]
        if x==i[0][0] and y2<i[0][1] :
            y2 = i[0][1]
        elif x==i[0][0] and y1>i[0][1] :
            y1 = i[0][1]
        elif x!=i[0][0] and y1!=y2:
            n_list.append([[int(x),int(y1)]])
            n_list.append([[int(x),int(y2)]])
            x = i[0][0]
            y1 = i[0][1]
            y2 = i[0][1]
        elif x!=i[0][0] and y1==y2 :
            n_list.append([[int(x),int(y1)]])
            #n_list.append([[int(x),int(y2)]])
            x = i[0][0]
            y1 = i[0][1]
            y2 = i[0][1]
    import math

    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0][0] - p2[0][0])**2 + (p1[0][1] - p2[0][1])**2)        
    nn_list=[]
    nn_list.append(n_list[0])
    while len(n_list)!=0:
        reference_point = nn_list[-1]
        n_list.remove(reference_point)
        sorted_points = sorted(n_list, key=lambda p: euclidean_distance(p, reference_point))
        if sorted_points!=[]:
            nn_list.append(sorted_points[0])
    X_new = np.array(nn_list)

    output = cv2.drawContours(output,[allcon],-1,[0,255,0],thickness=1)
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