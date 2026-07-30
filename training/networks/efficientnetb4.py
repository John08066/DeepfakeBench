'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is for EfficientNetB4 backbone.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from efficientnet_pytorch import EfficientNet
from metrics.registry import BACKBONE
import os

@BACKBONE.register_module(module_name="efficientnetb4")
class EfficientNetB4(nn.Module):
    def __init__(self, efficientnetb4_config):
        super(EfficientNetB4, self).__init__()
        """ Constructor
        Args:
            efficientnetb4_config: configuration file with the dict format
        """
        self.num_classes = efficientnetb4_config["num_classes"]
        inc = efficientnetb4_config["inc"]
        self.dropout = efficientnetb4_config["dropout"]
        self.mode = efficientnetb4_config["mode"]

        # Load the EfficientNet-B4 model without pre-trained weights
        if efficientnetb4_config['pretrained']:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4',weights_path=efficientnetb4_config['pretrained'])  # FIXME: load the pretrained weights from online
        # self.efficientnet = EfficientNet.from_name('efficientnet-b4')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b4')
        # Modify the first convolutional layer to accept input tensors with 'inc' channels   修改了EfficientNet-B4的输第一层卷积。用于提取输入图像的初始特征，它的默认输入通道数是 3（即常见的 RGB 图像通道数），输出通道数是 48
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)  # 这里的inc是输入通道数，可以指定输入通道数

        # Remove the last layer (the classifier) from the EfficientNet-B4 model  移除分类器
        self.efficientnet._fc = nn.Identity()  # self.efficientnet._fc 是 EfficientNet 模型的全连接层（分类器），用于将特征映射到特定数量的类别。将其替换为 nn.Identity()，表示不进行任何操作，直接返回输入张量。
        # 这样做的目的是移除原始分类头部（避免固定输出类别数），保留特征提取部分（主干网络）。

        if self.dropout:
            # Add dropout layer if specified
            self.dropout_layer = nn.Dropout(p=self.dropout)

        # Initialize the last_layer layer 在移除 EfficientNet-B4 原始分类器后，用一个新的线性层作为分类器。
        self.last_layer = nn.Linear(1792, self.num_classes)  # 输入是1792，是骨干的输出，输出是数据的类别数

        if self.mode == 'adjust_channel':  # 可选调整通道模块
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def block_part1(self,x):
        x = self.efficientnet._swish(self.efficientnet._bn0(self.efficientnet._conv_stem(x)))  # ._swish表示应用swish激活函数，._bn0表示使用BatchNorm，.conv_stem表示使用EfficientNet的第一层卷积。
        # x = self.efficientnet._blocks[0:10](x)
        for idx, block in enumerate(self.efficientnet._blocks[:10]):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate  # 来自 EfficientNet 的全局参数，表示初始的 Drop Connect 率。
            if drop_connect_rate:
                drop_connect_rate *= float(idx+0) / len(self.efficientnet._blocks)  # 这里动态调整 Drop Connect 率，使其随着 Block 的深度增加而线性增长（深层网络的正则化更强）。
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

    def block_part2(self,x):
        for idx, block in enumerate(self.efficientnet._blocks[10:22]):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx+10)  / len(self.efficientnet._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

    def block_part3(self,x):
        for idx, block in enumerate(self.efficientnet._blocks[22:]):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx+22)  / len(self.efficientnet._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self.efficientnet._swish(self.efficientnet._bn1(self.efficientnet._conv_head(x)))
        return x


    def features(self, x):
        # Extract features from the EfficientNet-B4 model
        x = self.efficientnet.extract_features(x)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x
    def end_points(self,x):
        return self.efficientnet.extract_endpoints(x)
    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Apply dropout if specified
        if self.dropout:
            x = self.dropout_layer(x)

        # Apply last_layer layer
        self.last_emb = x
        y = self.last_layer(x)
        return y

    def forward(self, x):
        # Extract features and apply classifier layer
        x = self.features(x)
        # if False:
        #     x = F.adaptive_avg_pool2d(x, (1, 1))
        #     x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
