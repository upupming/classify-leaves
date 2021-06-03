# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import random

from torchvision.transforms.transforms import Resize

def train_val_split(data_len,val_radio=0.1):
    all_idxs=list(range(0,data_len))
    random.shuffle(all_idxs)
    train_num=int(data_len-val_radio*data_len)
    return all_idxs[0:train_num],all_idxs[train_num:]    


# 继承pytorch的dataset，创建自己的dataset
class LeavesData(Dataset):
    def __init__(self, data_root, mode='train'):
        """
        Args:
            data_root (string): 数据集的根目录 
            mode (string): 训练模式还是测试模式
        """
        self.data_root=data_root
        csv_path=os.path.join(data_root,mode+'.csv')
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path)  #header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index)
        
        if mode == 'train':
            # 第一列包含图像文件的名称
            self.image_arr = np.asarray(self.data_info.iloc[0:, 0])  #self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.label_arr = np.asarray(self.data_info.iloc[0:, 1])
            self.init_info()
        elif mode == 'test':
            self.image_arr = np.asarray(self.data_info.iloc[1:, 0])
            
        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def init_info(self):
        leaves_labels = sorted(list(set(self.data_info['label'])))
        n_classes = len(leaves_labels)
        self.class_to_id = dict(zip(leaves_labels, range(n_classes)))
        self.classes=leaves_labels

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(os.path.join(self.data_root, single_image_name))        
        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            number_label = self.class_to_id[label]
            return img_as_img, number_label  #返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


class SplitDataset(Dataset):
    def __init__(self,leaves_data:LeavesData,idxs:list,trans=None):
        self.raw_data=leaves_data
        self.idxs=idxs
        self.trans=trans
    
    def __getitem__(self, index: int):
        real_idx=self.idxs[index]
        img,label=self.raw_data[real_idx]
        if self.trans is not None:
            img=self.trans(img)
        return img,label
    
    def __len__(self) -> int:
        return len(self.idxs)

#为了后续做TTA做准备
class TestDataset(Dataset):
    def __init__(self,leaves_data:LeavesData,trans=None) -> None:
        self.raw_data=leaves_data
        self.trans=trans
    
    def __len__(self) -> int:
        return len(self.raw_data)
    
    def __getitem__(self, index: int):
        img,label=self.raw_data[index]
        if self.trans is not None:
            img=self.trans(img)
        return img,label


def getData(args,mode='train'):
    train_transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop((224,224),padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    val_transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    all_data=LeavesData(args.data_root,mode)
    if mode=='train':
        train_idxs,val_idxs=train_val_split(len(all_data))
        train_data=SplitDataset(all_data,train_idxs,train_transform)
        val_data=SplitDataset(all_data,val_idxs,val_transform)
        return train_data,val_data
    else:
        test_data=TestDataset(all_data,val_transform)
        return test_data


if __name__=="__main__":
    train_data=LeavesData('../dataset/classify-leaves',"test")
    train_idxs,test_idxs=train_val_split(len(train_data))
    print(len(train_idxs))
    print(len(test_idxs))
    print(len(train_data))
        