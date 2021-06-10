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


class ResultSaver():
    def __init__(self, args, model) -> None:
        self.args = args
        self.model = model
        test_data = getData(args, mode='test')
        # print(len(test_data))
        # print(test_data[0][0].shape, test_data[0][1])

        self.class_to_id = test_data.raw_data.class_to_id
        self.id_to_class = test_data.raw_data.id_to_class
        column_names = ["image", "label"]
        self.ans = pd.DataFrame(columns=column_names)

        self.test_loader = dataloader.DataLoader(
            test_data, args.batch_size, shuffle=False)

    def start(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (imgs_list, img_names) in enumerate(tqdm(self.test_loader)):
                out=torch.zeros(imgs_list[0].shape[0],len(self.id_to_class)).to(args.device)
                for imgs in imgs_list:
                    imgs = imgs.to(self.args.device)
                    out += torch.softmax(self.model(imgs),1)
                labels = out.argmax(dim=1)
                for i in range(len(imgs)):
                    self.ans = self.ans.append({
                        'image': img_names[i],
                        'label': self.id_to_class[labels[i].item()]
                    }, ignore_index=True)
        self.ans.to_csv(
            path.join(
                path.dirname(__file__), '../', 'submission.csv'), index=False)


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
    fold=0
    try:             
        read_dict = torch.load(path.join(path.dirname(
                    __file__), f'../models/fold={fold}-{args.ckpt_path}'))
        model.module.load_state_dict(read_dict['weight'])
    except:
        print('模型加载失败')
        pass
    resultSaver = ResultSaver(args, model)
    resultSaver.start()
