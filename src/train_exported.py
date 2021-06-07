# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# 将 `classify-leaves` 放在根目录的 `dataset` 文件夹下，然后加载一下数据集：

# %%
from data_prepare import LeavesData, train_transform, test_transform

leaves_train = LeavesData(mode='train', transform=train_transform)
leaves_test = LeavesData(mode='test', transform=test_transform)
id_to_class = leaves_train.id_to_class
print('类别数', len(id_to_class))

# %% [markdown]
# 来看看数据长什么样子：

# %%
leaves_train[0][0].shape, leaves_train[0][1], id_to_class[leaves_train[0][1]]


# %%
get_ipython().run_line_magic('matplotlib', 'inline')
from d2l import torch as d2l
import torch


def get_labels(labels):
    """返回Fashion-MNIST数据集的文本标签。"""
    return [id_to_class[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontsize=8)
    return axes


# %%
from torch.utils import data

X, y = next(iter(data.DataLoader(leaves_train, batch_size=18)))
show_images(X.reshape(18, 3, 224, 224).numpy().transpose(
    0, 2, 3, 1), 2, 9, titles=get_labels(y))

# %% [markdown]
# 使用 sklean 进行 k 折交叉验证：

# %%
from sklearn.model_selection import KFold

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True)
for fold, (train_ids, val_ids) in enumerate(kfold.split(leaves_train)):
    print(f'Training for fold {fold}/{n_splits}...')
    print(len(train_ids), len(val_ids))
    train_subsampler = data.SubsetRandomSampler(train_ids)
    val_subsampler = data.SubsetRandomSampler(val_ids)

# %% [markdown]
# 使用预训练模型，然后加上自己的一些层即可:

# %%
from pretrainedmodels import se_resnext101_32x4d
model = se_resnext101_32x4d()
print(model)


# %%
# Copied from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# %%
from torch import nn

num_classes = 176
feature_extract = True
num_ftrs = model.last_linear.in_features
set_parameter_requires_grad(model, feature_extract)
model.last_linear = nn.Linear(num_ftrs, num_classes)
print(model)


# %%
device = d2l.try_gpu()
num_epochs = 10
batch_size = 64
model.to(device)

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()

for fold, (train_ids, val_ids) in enumerate(kfold.split(leaves_train)):
    print(f'Training for fold {fold}/{n_splits}...')
    train_subsampler = data.SubsetRandomSampler(train_ids)
    val_subsampler = data.SubsetRandomSampler(val_ids)
    train_loader = data.DataLoader(
        leaves_train,
        batch_size=batch_size, sampler=train_subsampler)
    val_loader = data.DataLoader(
        leaves_train,
        batch_size=batch_size, sampler=val_subsampler)

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_loader)
    for epoch in range(num_epochs):
        print(f'\tTraining for epoch {epoch}')
        # 训练态
        metric = d2l.Accumulator(3)
        model.train()
        for i, (X, y) in enumerate(train_loader):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        # 测试态
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, '
          f'test loss {test_loss:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# %%
