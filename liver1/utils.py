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

def sort_coordinates(list_of_xy_coords):
    list_of_xy_coords = list_of_xy_coords.reshape((-1,2))
    cx,cy= list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(-angles)
    return list_of_xy_coords[indices]