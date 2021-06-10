# Classify Leaves

目录结构:

```bash
├── dataset
│   └── classify-leaves
├── figures
│   ├── epoch-1.png
│   └── epoch-2.png
├── logs
│   ├── events.out.tfevents.1623141550.qian.5866.0
│   ├── events.out.tfevents.1623142471.qian.8148.0
│   └── events.out.tfevents.1623142967.qian.9325.0
├── Makefile
├── models
│   └── model.pth
├── README.md
└── src
    ├── data_prepare.py
    ├── data_utils.py
    ├── __init__.py
    ├── options.py
    ├── __pycache__
    ├── resnet.py
    ├── result_saver.py
    ├── train_exported.py
    ├── train.ipynb
    └── train.py
```

运行训练代码：

```bash

```

## Result

| 模型描述 | 验证集acc | 测试集分数 |
| :----:| :----: | :----: |
| se-resnext101(freeze) | 75.08（92.53） | 68.59 |
| se-resnext101(freeze)+warmup | 78.23（95.63） | 76.181 |

## To do

1. warm up(√)
2. 标签平滑
3. TTA(Test Time Augmentation)
4. unlabeled data
5. ensemble learning
6. 知识蒸馏
7. Mixup数据增强

## References

1. https://www.kaggle.com/c/classify-leaves/overview
2. https://www.kaggle.com/nekokiku/simple-resnet-baseline
3. https://blog.csdn.net/u011622208/article/details/102485741
4. https://github.com/Cadene/pretrained-models.pytorch
5. https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
6. https://github.com/CoinCheung/pytorch-loss/blob/master/label_smooth.py
