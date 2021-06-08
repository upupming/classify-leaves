from os import path
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from d2l import torch as d2l
from pretrainedmodels import se_resnext101_32x4d
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import dataloader
from tqdm import tqdm
from matplotlib import pyplot as plt
from data_utils import LeavesData, getData, test_transform, train_transform
from options import getArgs


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
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class ModelUpdater():
    def __init__(self, args, train_loader, val_loader, optimizer) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss()

    def train_one_epoch(self, model, animator, epoch):
        model.train()
        num_batches = len(train_loader)
        metric = d2l.Accumulator(3)
        for idx, (imgs, labels) in enumerate(tqdm(self.train_loader)):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
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

            if (idx + 1) % (num_batches // 5) == 0 or idx == num_batches - 1:
                animator.add(epoch + (idx + 1) / num_batches,
                             (train_acc * 100.0, train_l, None, None))
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
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
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


def set_parameter_requires_grad(model, feature_extracting, num_classes):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, num_classes)


if __name__ == '__main__':
    args = getArgs()
    args.device = d2l.try_gpu()
    num_classes = 176
    if args.model == 'seresnext101':
        model = se_resnext101_32x4d()
    else:
        print("Unexpected model type")
        exit(-1)

    set_parameter_requires_grad(model, args.freeze, num_classes)
    # print(model.last_linear.weight.requires_grad)

    current_epoch = 0
    if args.resume:
        save_dict = torch.load(args.ckpt_path)
        current_epoch = save_dict['current_epoch']
        model.load_state_dict(save_dict['weight'])

    train_data, val_data = getData(args, mode='train')
    train_loader = dataloader.DataLoader(
        train_data, args.batch_size, shuffle=True)
    val_loader = dataloader.DataLoader(val_data, args.batch_size, shuffle=True)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    schdueler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epoch, 1e-5)
    model = nn.DataParallel(model)
    model = model.to(args.device)
    updater = ModelUpdater(args, train_loader, val_loader, optimizer)

    best_loss = 1e9
    best_weight = copy.deepcopy(model.module.state_dict())
    writer = SummaryWriter('./logs')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, args.epoch],
                            legend=['train acc', 'train loss', 'test acc (top1)', 'test acc (top5)'])
    for i in range(args.epoch):
        print("Epoch {}/{} training...".format(i, args.epoch))
        loss, acc = updater.train_one_epoch(model, animator, i)
        writer.add_scalar("loss", loss, i)
        writer.add_scalar("train_acc", acc,i)
        print("train loss:{} acc:{}".format(loss, acc))
        if loss < best_loss:
            best_loss = loss
            best_weight = copy.deepcopy(model.module.state_dict())
            save_dict = {
                "weight": best_weight,
                "current_epoch": i,
                "best_loss": best_loss,
            }
            torch.save(save_dict, path.join(path.dirname(
                __file__), f'../models/{args.ckpt_path}'))
        if args.eval_all:
            acc, acc5 = updater.validate(model, use_top5=True)
            print("acc:{} acc5:{}".format(acc, acc5))
            writer.add_scalars("test_acc", {"top1": acc, "top5": acc5},i)

            animator.add(i + 1, (None, None, acc, acc5))
        plt.savefig(path.join(path.dirname(__file__),
                    f'../figures/epoch-{i+1}.png'))
