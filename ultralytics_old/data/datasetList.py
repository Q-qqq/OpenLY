from torch.utils import data

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def pad_to_square(img,pad_value):
    _,h,w = img.shape
    dim_diff = np.abs(h - w)
    pad1,pad2 = dim_diff//2, dim_diff - dim_diff//2
    pad = (pad1,pad2,0,0) if h >= w else (0,0,pad1,pad2)
    img = torch.nn.functional.pad(img,pad,"constant",pad_value)

    return img,pad

def resize(img,img_size):
    img = torch.nn.functional.interpolate(img.unsqueeze(0),size=img_size,mode = "nearest").squeeze(0)
    return img



class dataset(data.Dataset):
    def __init__(self,data_path,img_size,class_num):
        super(dataset,self).__init__()

        with open(data_path,"r") as f:
            lines = f.readlines()              #读取文件

        self.path_imgs = [line.rstrip() for line in lines if line is not None] #去空格
        hz = self.path_imgs[0].split(".")[1]     #获取图像文件后缀
        self.path_labels = [path_img.replace("images","labels").replace(hz,"txt")
                            for path_img in self.path_imgs]                  #转换为label路径
        self.img_size = img_size
        self.class_num = class_num

    def __getitem__(self, item):
        index = item%len(self.path_imgs)           #获取数据的索引
        img = transforms.ToTensor()(Image.open(self.path_imgs[index]))   #打开图像并转换为Tensor

        img,_ = pad_to_square(img,0)       #填充图像为正方形
        img = resize(img,self.img_size)    #重构图像大小

        label = torch.from_numpy(np.loadtxt(self.path_labels[index]).reshape(-1,self.class_num))     #获取图像对应标签

        return img, label

    def __len__(self):
        return len(self.path_imgs)

        

