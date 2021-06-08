import pandas as pd
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


class ResultSaver():
    def __init__(self, args, model: nn.DataParallel) -> None:
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
            for idx, (imgs, img_names) in enumerate(tqdm(self.test_loader)):
                imgs = imgs.to(self.args.device)
                out = model(imgs)
                labels = out.argmax(dim=1)
                for i in range(len(imgs)):
                    self.ans = self.ans.append({
                        'image': img_names[i],
                        'label': self.id_to_class[labels[i].item()]
                    }, ignore_index=True)
        self.ans.to_csv(
            path.join(
                path.dirname(__file__),
                '../dataset/classify-leaves', 'submission.csv'), index=False)


if __name__ == "__main__":
    args = getArgs()
    args.device = d2l.try_gpu()
    num_classes = 176
    if args.model == 'seresnext101':
        model = se_resnext101_32x4d()
    else:
        print("Unexpected model type")
        exit(-1)
    set_parameter_requires_grad(model, args.freeze, num_classes)
    model = nn.DataParallel(model)
    model = model.to(args.device)
    try:
        read_dict = torch.load(path.join(path.dirname(
            __file__), f'../models/{args.ckpt_path}'))
        model.module.load_state_dict(read_dict['weight'])
    except:
        print('模型加载失败')
        pass
    resultSaver = ResultSaver(args, model=model)
    resultSaver.start()
