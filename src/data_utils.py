# 首先导入包
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from os import path
import matplotlib.pyplot as plt
import random
from d2l import torch as d2l

from torchvision.transforms.transforms import RandomRotation, Resize


def train_val_split(data_len, val_ratio=0.1):
    all_idxs = list(range(0, data_len))
    random.shuffle(all_idxs)
    train_num = int(data_len - val_ratio * data_len)
    return all_idxs[0:train_num], all_idxs[train_num:]


# 继承pytorch的dataset，创建自己的dataset
class LeavesData(Dataset):
    def __init__(self, data_root=path.join(path.dirname(__file__), '../dataset/classify-leaves'), mode='train', transform=None):
        """
        Args:
            data_root (string): 数据集的根目录
            mode (string): 训练模式还是测试模式
        """
        self.data_root = data_root
        csv_path = os.path.join(data_root, mode + '.csv')
        train_csv_path = os.path.join(data_root, 'train.csv')
        self.mode = mode
        self.transform = transform

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path)  # header=None是去掉表头部分
        self.train_data_info = pd.read_csv(train_csv_path)
        # 计算 length
        self.data_len = len(self.data_info.index)

        self.init_info()
        if mode == 'train':
            # 第一列包含图像文件的名称
            # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            self.image_arr = np.asarray(self.data_info.iloc[0:, 0])
            # 第二列是图像的 label
            self.label_arr = np.asarray(self.data_info.iloc[0:, 1])
        elif mode == 'test':
            self.image_arr = np.asarray(self.data_info.iloc[0:, 0])

        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def init_info(self):
        leaves_labels = sorted(list(set(self.train_data_info['label'])))
        n_classes = len(leaves_labels)
        self.class_to_id = dict(zip(leaves_labels, range(n_classes)))
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        self.classes = leaves_labels

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = np.array(Image.open(os.path.join(
            self.data_root, single_image_name)))
        # print(type(img_as_img))
        if self.transform:
            img_as_img = self.transform(image=img_as_img)['image']
        if self.mode == 'test':
            # 返回一个图像名称方便预测生成 csv 用
            return img_as_img, single_image_name
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            number_label = self.class_to_id[label]
            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


class SplitDataset(Dataset):
    def __init__(self, leaves_data: LeavesData, idxs: list, trans=None):
        self.raw_data = leaves_data
        self.idxs = idxs
        self.trans = trans

    def __getitem__(self, index: int):
        real_idx = self.idxs[index]
        img, label = self.raw_data[real_idx]
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        return img, label

    def __len__(self) -> int:
        return len(self.idxs)

# 为了后续做TTA做准备


class TestDataset(Dataset):
    def __init__(self, args, leaves_data: LeavesData, trans=None) -> None:
        self.raw_data = leaves_data
        self.transform_list = None
        self.transform = None
        if args.tta_transform:
            self.transform_list = trans
        else:
            self.transform = trans

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, index: int):
        img, img_name = self.raw_data[index]
        if self.transform_list is not None:
            img_list = []
            for trans in self.transform_list:
                img_list.append(trans(image=img)['image'])
            return img_list, img_name
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, img_name


mean = [0 * 255.0, 0 * 255.0, 0 * 255.0]
std = [1, 1, 1]

train_transform = albu.Compose([
    albu.Resize(224, 224),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),

    # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
    #                       shift_limit=0.1, p=1, border_mode=0),

    # albu.PadIfNeeded(min_height=224, min_width=224,
    #                  always_apply=True, border_mode=0),
    # albu.RandomCrop(height=224, width=224, always_apply=True),

    # albu.GaussNoise(p=0.2),
    # albu.Perspective(p=0.5),

    # albu.OneOf(
    #     [
    #         albu.CLAHE(p=1),
    #         albu.RandomBrightnessContrast(p=1),
    #         albu.RandomGamma(p=1),
    #     ],
    #     p=0.9,
    # ),

    # albu.OneOf(
    #     [
    #         albu.Sharpen(p=1),
    #         albu.Blur(blur_limit=3, p=1),
    #         albu.MotionBlur(blur_limit=3, p=1),
    #     ],
    #     p=0.9,
    # ),

    # albu.OneOf(
    #     [
    #         albu.RandomBrightnessContrast(p=1),
    #         albu.HueSaturationValue(p=1),
    #     ],
    #     p=0.9,
    # ),
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    albu.Normalize(mean, std),
    ToTensorV2(),

])
test_transform = val_transform = albu.Compose([
    albu.Resize(224, 224),
    albu.Normalize(mean, std),
    ToTensorV2(),
])

TTA_transform = [
    albu.Compose(
        [albu.Resize(224, 224), albu.Normalize(mean, std), ToTensorV2(), ]),
    albu.Compose([albu.Resize(224, 224), albu.HorizontalFlip(
        p=1), albu.Normalize(mean, std), ToTensorV2(), ]),
    albu.Compose([albu.Resize(224, 224), albu.VerticalFlip(
        p=1), albu.Normalize(mean, std), ToTensorV2(), ]),
    albu.Compose([albu.Resize(224, 224), albu.RandomBrightnessContrast(
        p=1), albu.Normalize(mean, std), ToTensorV2(), ]),
    albu.Compose([albu.Resize(224, 224), albu.Perspective(
        p=1), albu.Normalize(mean, std), ToTensorV2(), ]),
    albu.Compose([albu.Resize(224, 224), albu.HueSaturationValue(
        p=1), albu.Normalize(mean, std), ToTensorV2(), ]),
]


def getData(args, mode='train'):

    all_data = LeavesData(
        path.join(path.dirname(__file__), args.data_root), mode)
    if mode == 'train':
        train_idxs, val_idxs = train_val_split(len(all_data), args.val_ratio)
        train_data = SplitDataset(all_data, train_idxs, train_transform)
        val_data = SplitDataset(all_data, val_idxs, val_transform)
        return train_data, val_data
    else:
        test_data = TestDataset(args,
            all_data, TTA_transform if args.tta_transform else test_transform)
        return test_data


def load_leaves_data(batch_size, resize=None):
    leaves_train_val = LeavesData(mode='train')
    leaves_test = LeavesData(mode='test')
    num_workers = 4
    return (DataLoader(leaves_train_val, batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(leaves_test, batch_size, shuffle=False,
                       num_workers=num_workers))


if __name__ == "__main__":
    # test_data = LeavesData(path.join(path.dirname(
    #     __file__), '../dataset/classify-leaves'), "test")
    # for idx, img in enumerate(test_data):
    #     print(img.width)
    leaves_train = LeavesData(mode='train', transform=train_transform)
    batch_size = 128
    train_iter = DataLoader(
        leaves_train, batch_size, shuffle=False, num_workers=4)
    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'读取一轮训练数据需要 {timer.stop():.2f} sec')
