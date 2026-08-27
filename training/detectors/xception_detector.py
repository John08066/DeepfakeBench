# region info
'''
#  author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XceptionDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''
#endregion inf
import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC


logger = logging.getLogger(__name__) #给当前这个 Python 文件拿到一个专属的日志记录器 logger，以后这个文件想打印日志，就通过它来记录。
    # logging 是Python自带的日志系统 其中 getLogger()是：获取一个指定名字的Logger对象  __name__ 是 Python 自动提供的特殊变量，表示当前模块的名字
    # 当它作为模块被导入时，__name__ 很可能类似："training.detectors.xception_detector"因为 Python logging 默认存在**向父 logger 传播（propagation）**机制。
    # 这就是为什么 Detector 自己不需要知道：training.log 到底放在哪？实验目录叫什么？ 时间戳是什么？
@DETECTOR.register_module(module_name='xception')
class XceptionDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config) #对象持有另一个对象的引用，self.backbone = Xception对象
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):# prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']] # 通过 config['backbone_name'] 获取骨干网络的类名，然后从 BACKBONE 注册表中获取对应的类对象。
        model_config = config['backbone_config'] 
        backbone = backbone_class(model_config) # 实例化骨干网络
        state_dict = torch.load(config['pretrained']) # 通过预训练权重路径加载预训练权重 state_dict键是模型参数的名称，值是对应的权重张量{'conv1.weight': Tensor(...), 'bn1.weight': Tensor(...), 'block1.rep.0.pointwise.weight': Tensor(...), 'fc.weight': Tensor(...), 'fc.bias': Tensor(...)}
        for name, weights in state_dict.items(): #修正 pointwise convolution 权重的维度
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1) # 将权重张量的形状从 [out_channels, in_channels] 转换为 [out_channels, in_channels, 1, 1]，以适应卷积层的权重形状要求。
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k} # 只要它学好的特征提取能力，不要原来的分类头，过滤掉全连接层的权重，尤其是在迁移学习或微调时。
        backbone.load_state_dict(state_dict, False) # False 表示不严格匹配，允许加载部分权重，忽略缺失或多余的键。正好和前面删除fc呼应
        logger.info('Load pretrained model successfully!')
        return backbone

    def build_loss(self, config): # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']] 
        loss_func = loss_class() # 类实例化
        return loss_func

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256 data_dict 可能包含 image、label、name，以及 mask、landmark 等字段

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features) #32,2

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:#Detector 决定怎样组成总损失 Trainer 只负责对 overall 执行 backward
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss #有些 Detector可能有多个损失  这里Xception是单任务基线，因此
        loss_dict = {'overall': overall_loss, 'cls': loss,} # overall → 用于 backward，cls → 用于日志和分析
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'] # 这里不复用loss函数的局部变量，因为get_losses()与get_train_metrics()是两个独立接口，由Trainer分别调用
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())  # compute metrics for batch data
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        self.video_names = []   # we dont compute the video-level metrics for training
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict) # get the features by backbone
        pred = self.classifier(features)  # get the prediction by classifier
        prob = torch.softmax(pred, dim=1)[:, 1] 
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}   # build the prediction dict for each output
        return pred_dict