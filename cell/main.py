from utils import cut,mix_circle
import cv2
from PIL import Image
import torch
from model import Net,Unet
from torchvision import transforms
from torchvision.transforms import ToTensor
import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import statistics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mix_dir')
parser.add_argument('--all_sample_dir',default='./sample/')
parser.add_argument('--test_dir',default='./test/')
args = parser.parse_args()


if __name__ == '__main__':
    value=100
    #76 705/1024
    um=0.181
    circles = mix_circle(args.mix_dir)
    captured_frame1 = cv2.imread(args.mix_dir)
    captured_frame1 = cv2.copyMakeBorder(captured_frame1,value,value,value,value,cv2.BORDER_CONSTANT,value=(0,0,0))
    captured_frame2 = cv2.imread(args.mix_dir)
    captured_frame2 = cv2.copyMakeBorder(captured_frame2,value,value,value,value,cv2.BORDER_CONSTANT,value=(0,0,0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = Net().to(device).eval()
    m1 = Unet().to(device).eval()
    #checkpoint = torch.load('save.pt')
    m.load_state_dict(torch.load('save1.pt'))
    m1.load_state_dict(torch.load('save_mask.pt'))
    tranform = transforms.Compose([transforms.Pad(12, fill=(0,0,0), padding_mode='constant'),ToTensor()])
    new_circle=[]
    fail = []
    idx=0
    for i in range(len(circles)):
        if i==140:
            x=1
        a = cut(args.mix_dir,circles[i])
        image = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
        cv2.circle(captured_frame1, center=(circles[i,0], circles[i,1]), radius=circles[i,2], color=(0, 255, 0), thickness=2)
        image = tranform(image).to(device).unsqueeze(0)
        output = m(image)
        if output>0.75:
            new_circle.append(circles[i])
            output1 = m1(image)
            x = F.sigmoid(output1.squeeze(1))
            res = x >= 0.9
            y = torchvision.utils.draw_segmentation_masks(image.squeeze(0),res,0.5,(0,255,0))
            #x = T.ToPILImage()(x)
            y = T.ToPILImage()(y)
            #x.save('./mask.jpg')
            y.save(args.all_sample_dir+str(idx)+'.jpg')
            idx+=1
            #x=1
        else:
            fail.append(circles[i])
        
        #cv2.imwrite('output1.jpg', a)
    '''
    x=[]
    y=[]
    raius=[]
    field=[]
    for i in range(len(new_circle)):
        x.append(new_circle[i][0])
        y.append(new_circle[i][1])
        raius.append(new_circle[i][2])
        field.append(math.pi*new_circle[i][2]*new_circle[i][2]*um*um)
    data = {
        "x": x,
        "y": y,
        "radius": raius,
        "area": field
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_dir)
    '''
    raius=[]
    field=[]
    mask_field=[]
    ratio = []
    y = T.ToTensor()(Image.fromarray(cv2.cvtColor(captured_frame2, cv2.COLOR_BGR2RGB)))
    mask_list=[]
    for i in range(len(new_circle)):
        cv2.circle(captured_frame2, center=(new_circle[i][0], new_circle[i][1]), radius=new_circle[i][2], color=(0, 255, 0), thickness=2)
        a = cut(args.mix_dir,new_circle[i])
        image = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
        image = tranform(image).to(device).unsqueeze(0)
        output1 = m1(image)
        x = F.sigmoid(output1.squeeze(1))
        pad = nn.ZeroPad2d((new_circle[i][0]-112,1224-112-new_circle[i][0],new_circle[i][1]-112,1224-112-new_circle[i][1]))
        x = pad(x)
        res = x >= 0.9
        y = torchvision.utils.draw_segmentation_masks(y,res,0.45,(0,255,0))
        positive_pixel_count = res.sum()
        mask_list.append(res.squeeze(0))
        raius.append(new_circle[i][2])
        field.append(math.pi*new_circle[i][2]*new_circle[i][2]*um*um)
        mask_field.append(int(positive_pixel_count.cpu())*um*um)
        ratio.append((math.pi*new_circle[i][2]*new_circle[i][2])/int(positive_pixel_count.cpu()))
        #cv2.imwrite('./train_mask/'+str(i)+'.jpg', a)
    stacked = torch.stack(mask_list, dim=0)
    #y1 = T.ToTensor()(Image.fromarray(cv2.cvtColor(captured_frame2, cv2.COLOR_BGR2RGB)))
    #y = torchvision.utils.draw_segmentation_masks(y1,stacked,0.3,(0,255,0))
    raius.append(statistics.mean(raius))
    field.append(statistics.mean(field))
    mask_field.append(statistics.mean(mask_field))
    ratio.append(statistics.mean(ratio))
    data = {
        "radius": raius,
        "nuclear_area": field,
        "cytoplasmic_area":mask_field,
        "ratio":ratio
    }
    
    df = pd.DataFrame(data)
    #df.loc['mean'] = df.mean()
    df.to_csv(args.test_dir+'test.csv')
    y = T.ToPILImage()(y[:,100:-100,100:-100])
    y.save(args.test_dir+'test.jpg')