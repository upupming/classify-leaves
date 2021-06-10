from genericpath import isfile
from ntpath import join
import os
from os import listdir
import traceback
import sys
import pandas as pd
from pretrainedmodels.models.senet import se_resnext50_32x4d
from data_utils import getData
from options import getArgs
from torch.utils.data import dataloader
from pretrainedmodels import se_resnext101_32x4d
from tqdm import tqdm
import torch
from torch import nn
from os import path
from train import set_parameter_requires_grad
from d2l import torch as d2l
import copy
from data_utils import LeavesData
from data_utils import test_transform


class ResultSaver():
    def __init__(self, args, model_list) -> None:
        self.args = args
        self.model_list = model_list
        test_data = getData(args, mode='test')
        # print(len(test_data))
        # print(test_data[0][0].shape, test_data[0][1])

        self.class_to_id = test_data.raw_data.class_to_id
        self.id_to_class = test_data.raw_data.id_to_class
        column_names = ["image", "label"]
        self.ans = pd.DataFrame(columns=column_names)

        self.test_loader = dataloader.DataLoader(
            test_data, args.batch_size, shuffle=False)

    def pred_one_batch(self, imgs):
        imgs = imgs.to(self.args.device)
        out = torch.zeros(imgs.shape[0], len(
            self.id_to_class)).to(self.args.device)
        for model in self.model_list:
            out += torch.softmax(model(imgs), dim=1)
        return out.argmax(dim=1)

    def start(self):
        for model in self.model_list:
            model.eval()
        with torch.no_grad():
            for idx, (imgs_list, img_names) in enumerate(tqdm(self.test_loader)):
                pred_labels = torch.zeros(imgs.shape[0], len(self.id_to_class)).to(self.args.device)
                for imgs in imgs_list:
                    pred_labels += self.pred_one_batch(imgs)
                pred_labels=pred_labels.argmax(dim=1)
                for i in range(len(imgs)):
                    self.ans = self.ans.append({
                        'image': img_names[i],
                        'label': self.id_to_class[pred_labels[i].item()]
                    }, ignore_index=True)
        self.ans.to_csv(
            path.join(
                path.dirname(__file__), '../', 'submission.csv'), index=False)

    def cal_acc_on_train(self):
        leaves_train = LeavesData(mode='train', data_root=path.join(
            path.dirname(__file__), args.data_root), transform=test_transform)
        self.train_loader = dataloader.DataLoader(
            leaves_train, args.batch_size, shuffle=False)

        for model in self.model_list:
            model.eval()
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(self.train_loader):
                labels = labels.to(self.args.device)
                pred_labels = self.pred_one_batch(imgs, model)
                metric.add(d2l.accuracy(pred_labels, labels), len(labels))
                print(
                    f'{idx+1}/{len(self.train_loader)} 当前训练集上 acc 为 {metric[0] / metric[1]}')


if __name__ == "__main__":
    args = getArgs()
    args.device = d2l.try_gpu()
    num_classes = 176
    if args.model == 'seresnext101':
        model = se_resnext101_32x4d()
    elif args.model == "seresnext50":
        model = se_resnext50_32x4d()
    else:
        print("Unexpected model type")
        exit(-1)
    set_parameter_requires_grad(model, args.freeze, num_classes)
    model = nn.DataParallel(model)
    model = model.to(args.device)
    model_list = []
    try:
        model_dir = path.join(path.dirname(
            __file__), f'../models/')
        model_files = [join(model_dir, f) for f in os.listdir(model_dir) if(isfile(
            join(model_dir, f)) and f.lower().endswith('.pth'))]

        for file in model_files:
            read_dict = torch.load(file)
            model.module.load_state_dict(read_dict['weight'])
            model_list.append(copy.deepcopy(model))
        print(f'已加载 {len(model_list)} 个模型')
    except:
        print(traceback.format_exc())
        print('模型加载失败')
        sys.exit(-1)
    resultSaver = ResultSaver(args, model_list)
    if args.res_train:
        resultSaver.cal_acc_on_train()
    else:
        resultSaver.start()
