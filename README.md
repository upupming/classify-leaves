# Classify Leaves

目录结构:

```bash
.
├── README.md
├── dataset
│   └── classify-leaves
└── src
    ├── data_prepare.py
    ├── options.py
    ├── train.ipynb
    └── train.py
```

运行训练代码：

```bash

```

## Result

| 模型描述 | 验证集acc | 测试集分数 |
| :----:| :----: | :----: |
| se-resnext101(freeze) | 68.08（92.53） | -- |

## To do

1. warm up
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
