from os import path
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from d2l import torch as d2l
from pretrainedmodels import se_resnext101_32x4d, se_resnext50_32x4d
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from data_utils import LeavesData, getData, test_transform, train_transform
from options import getArgs
from sklearn.model_selection import KFold
from label_smooth import LabelSmoothSoftmaxCEV2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) / batch_size for k in topk]


class ModelUpdater():
    def __init__(self, args, train_loader, val_loader, optimizer) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.optimizer = optimizer
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = LabelSmoothSoftmaxCEV2()

    def train_one_epoch(self, model):
        model.train()
        num_batches = len(self.train_loader)
        metric = d2l.Accumulator(3)
        for idx, (imgs, labels) in enumerate(tqdm(self.train_loader)):
            imgs = imgs.to(self.args.device)
            labels = labels.to(self.args.device)
            self.optimizer.zero_grad()
            out = model(imgs)
            loss = self.loss_fn(out, labels)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                metric.add(
                    loss * imgs.shape[0], d2l.accuracy(out, labels), imgs.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if self.args.verbose and (idx + 1) % 10 == 0:
                print("{}/{} loss:{}  acc:{}".format(idx,
                      num_batches, train_l, train_acc))
        return train_l, train_acc

    def validate(self, model, use_top5=False):
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        pred = []
        ground_truth = []
        if use_top5:
            top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(self.val_loader):
                imgs = imgs.to(self.args.device)
                labels = labels.to(self.args.device)
                out = model(imgs)

                pred.extend((torch.argmax(out, 1)).cpu().numpy().tolist())
                ground_truth.extend(list(labels.cpu().numpy()))

                acc1 = accuracy(out, labels, topk=(1,))
                top1.update(acc1[0].item(), imgs.size(0))
                if use_top5:
                    acc5 = accuracy(out, labels, topk=(5,))
                    top5.update(acc5[0].item(), imgs.size(0))
        if not use_top5:
            return top1.avg
        else:
            return top1.avg, top5.avg


def set_parameter_requires_grad(model: nn.Module, feature_extracting, num_classes):
    model_children = list(model.children())
    if feature_extracting:
        for i in range(len(model_children)):
            if i != 4 or i != len(model_children) - 1:
                layer = model_children[i]
                # print(layer)
                for param in layer.parameters():
                    param.requires_grad = False
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, num_classes)


def train(args):
    args.device = d2l.try_gpu()
    num_classes = 176

    leaves_train = LeavesData(mode='train', data_root=path.join(
        path.dirname(__file__), args.data_root), transform=train_transform)
    leaves_train_no_trans = LeavesData(mode='train', data_root=path.join(
        path.dirname(__file__), args.data_root), transform=test_transform)
    # print(leaves_train[0])
    kFold = KFold(n_splits=args.fold, shuffle=False)
    for fold, (train_ids, val_ids) in enumerate(kFold.split(leaves_train)):
        print(f'Training for fold {fold}/{args.fold}...')
        model_path = path.join(path.dirname(
            __file__), f'../models/fold={fold}-{args.ckpt_path}')
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(
            leaves_train,
            batch_size=args.batch_size, sampler=train_subsampler, num_workers=4)
        val_loader = DataLoader(
            leaves_train_no_trans,
            batch_size=args.batch_size, sampler=val_subsampler, num_workers=4)

        if args.model == 'seresnext101':
            model = se_resnext101_32x4d()
        elif args.model == 'seresnext50':
            model = se_resnext50_32x4d()
        else:
            print("Unexpected model type")
            exit(-1)

        set_parameter_requires_grad(model, args.freeze, num_classes)
        # print(model.last_linear.weight.requires_grad)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epoch, 1e-5)
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

        model = nn.DataParallel(model)
        model = model.to(args.device)
        updater = ModelUpdater(args, train_loader, val_loader, optimizer)
        current_epoch = 0
        best_acc = 0
        if args.resume and path.exists(model_path):
            save_dict = torch.load(model_path)
            current_epoch = save_dict['current_epoch']
            model.module.load_state_dict(save_dict['weight'])
            print(
                f'loaded model from {model_path} for resume training with current_epoch={current_epoch}')
            optimizer.load_state_dict(save_dict['optimizer'])
            scheduler_warmup.load_state_dict(save_dict["scheduler"])
            best_acc = save_dict["best_acc"]

        best_weight = copy.deepcopy(model.module.state_dict())
        writer = SummaryWriter('./logs')
        for i in range(current_epoch, args.epoch):
            print("Epoch {}/{} training...".format(i, args.epoch))
            scheduler_warmup.step()

            loss, acc = updater.train_one_epoch(model)
            writer.add_scalar(f"fold={fold}-loss", loss, i)
            writer.add_scalar(f"fold={fold}-train_acc", acc, i)
            print("train loss:{} acc:{}".format(loss, acc))

            if args.eval_all:
                acc, acc5 = updater.validate(model, use_top5=True)
                print("acc:{} acc5:{}".format(acc, acc5))
                writer.add_scalars(
                    f"fold={fold}-test_acc", {"top1": acc, "top5": acc5}, i)
            # ??????eval_all?????????acc??????????????????acc???????????????????????????acc
            if acc > best_acc:
                best_acc = acc
                best_weight = copy.deepcopy(model.module.state_dict())
                save_dict = {
                    "weight": best_weight,
                    "current_epoch": i,
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler_warmup.state_dict(),
                }
                torch.save(save_dict, model_path)
            plt.savefig(path.join(path.dirname(__file__),
                        f'../figures/fold={fold}-epoch={i+1}.png'))


if __name__ == '__main__':
    args = getArgs()
    train(args)
