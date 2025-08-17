import os
import pandas as pd
from torchvision.io import read_image
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
from torchvision.transforms import ToTensor

import numpy as np
import cv2

value=100

'''
im = Image.open("./train1/0.tif")
#im.show()
im = TF.rotate(im,120,Image.BILINEAR,False,)
im.show()
transform = transforms.Compose([
    transforms.RandomRotation(30, resample=Image.BICUBIC, expand=False, center=(55, 5))])
'''




class CellDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir+'/'+str(idx)+'.tif'
        image = Image.open(img_path)
        tranform = transforms.Compose([transforms.Pad(12, fill=(0,0,0), padding_mode='constant'),ToTensor()])
        image = tranform(image)
        label = self.img_labels['0'][idx]

        return image, label

class MaskDataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None, target_transform=None):
        self.leng = os.listdir(label_dir)
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.leng)

    def __getitem__(self, idx):
        img_path = self.img_dir+'/'+str(idx)+'.jpg'
        label_path = self.label_dir+'/'+str(idx)+'.jpg'
        image = Image.open(img_path)
        label = Image.open(label_path)
        transform = transforms.Compose([transforms.Pad(12, fill=(0,0,0), padding_mode='constant'),ToTensor()])
        transform_label = transforms.Compose([transforms.Pad(12, fill=(0), padding_mode='constant'),ToTensor()])
        image = transform(image)
        label = transform_label(label)
        #label = self.img_labels['0'][idx]

        return image, label

#dataset = MaskDataset(img_dir='./mask_train',label_dir='./mask_train_label')
#train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#train_features, train_labels = next(iter(train_dataloader))
#x=1
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def circle(red_dir ):
    value=100
    captured_frame = cv2.imread(red_dir)
    captured_frame = cv2.copyMakeBorder(captured_frame,value,value,value,value,cv2.BORDER_CONSTANT,value=(0,0,0))

    #captured_frame2 = cv2.imread('./4/4_hepatocyte membrane blue.tif')
    #captured_frame2 = cv2.copyMakeBorder(captured_frame2,value,value,value,value,cv2.BORDER_CONSTANT,value=(0,0,0))

    captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)

    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 7)

    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)

    captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([0, 160, 160]), np.array([180, 255, 255]))

    captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)

    circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT, 0.01, captured_frame_lab_red.shape[0] / 48, 
                            param1=60, param2=20, minRadius=3, maxRadius=60)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles

def mix_circle(mix_dir ):
    value=100
    captured_frame = cv2.imread(mix_dir)
    captured_frame = cv2.copyMakeBorder(captured_frame,value,value,value,value,cv2.BORDER_CONSTANT,value=(0,0,0))

    #captured_frame2 = cv2.imread('./4/4_hepatocyte membrane blue.tif')
    #captured_frame2 = cv2.copyMakeBorder(captured_frame2,value,value,value,value,cv2.BORDER_CONSTANT,value=(0,0,0))

    captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)

    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 7)

    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)
    #def show_xy(event,x,y,flags,param):            # 顯示繪製後的影像
        #color = captured_frame_lab[y,x]                          # 當滑鼠點擊時
        #print(color)                              # 印出顏色
    #cv2.imshow('oxxostudio', captured_frame_lab)
    #cv2.setMouseCallback('oxxostudio', show_xy)

    captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([70, 70, 70]), np.array([190, 255, 255]))
    #captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([0, 160, 160]), np.array([180, 255, 255]))
    captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)

    circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT_ALT, 1, captured_frame_lab_red.shape[0] / 64, 
                            param1=50, param2=0.8, minRadius=1, maxRadius=60)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    #for i in range(len(circles)):
        #cv2.circle(captured_frame, center=(circles[i,0], circles[i,1]), radius=circles[i][2], color=(0, 255, 0), thickness=2)
    #cv2.imshow('x',captured_frame)
    #cv2.imshow('y',captured_frame_lab_red)
    #cv2.waitKey(0)
    return circles

def cut(dir_mix,circle):
    value=100
    captured_frame1 = cv2.imread(dir_mix)
    captured_frame1 = cv2.copyMakeBorder(captured_frame1,value,value,value,value,cv2.BORDER_CONSTANT,value=(0,0,0))
        #cv2.circle(captured_frame, center=(circles[i, 0], circles[i, 1]), radius=circles[i, 2], color=(0, 255, 0), thickness=2)
    cv2.circle(captured_frame1, center=(circle[0], circle[1]), radius=circle[2], color=(0, 255, 0), thickness=2)
    if(circle[0]>1224-value):
        circle[0]=1224-value
    if(circle[1]>1224-value):
        circle[1]=1224-value
    if(circle[0]<value):
        circle[0]=value
    if(circle[1]<value):
        circle[1]=value
    
    new_image = captured_frame1[circle[1]-value:circle[1]+value,circle[0]-value:circle[0]+value,:]
        #cv2.imwrite('./train1/'+str(i)+'.tif',new_image)
        #new_image = captured_frame2[circles[i, 1]-value:circles[i, 1]+value,circles[i, 0]-value:circles[i, 0]+value,:]
        #cv2.imwrite('./train/'+str(i)+'.tif',new_image)
    return new_image


import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)