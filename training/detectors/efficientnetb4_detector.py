'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the EfficientDetector

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
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
'''

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
from concurrent.futures import ThreadPoolExecutor
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random
import dlib
import cv2
logger = logging.getLogger(__name__)

# 使用注册机制注册名为'efficientnetb4'的检测器类
@DETECTOR.register_module(module_name='efficientnetb4')
class EfficientDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.landmark_data = LandmarkProcessor()
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        model_config['pretrained'] = self.config['pretrained']
        backbone = backbone_class(model_config)
        if config['pretrained'] != 'None':
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')
        return backbone

    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data) -> torch.tensor:
        x = self.backbone.features(data)
        return x

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    def get_npr(self,data_dict: dict) -> dict:
        data_dict['image'] = data_dict['image']-self.interpolate(data_dict['image'], 0.5)
        return data_dict
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get npr
        # data_dict = self.get_npr(data_dict)
        # get the features by backbone
        landmark_heatmaps = self.landmark_data.get_landmark(data_dict)
        if inference == False:
            noise = self.landmark_data.generate_region_noise_map(landmark_heatmaps)
            noise_data = 0.5 * noise + data_dict['image']
            features = self.features(noise_data)
        else:
            features = self.features(data_dict['image'])
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict

class LandmarkProcessor:
    def __init__(self,heatmap_shape=(256,256)): # 不用vit是256
        self.predictor_path = '/home/csy/disk1/project_1/DeepfakeBench-main/training/pretrained/shape_predictor_81_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.heatmap_shape = heatmap_shape
        # 初始化网格
        self._init_grid(heatmap_shape)

    def _init_grid(self, shape):
        h, w = shape
        self.x_grid = torch.arange(w, device='cuda').view(1, -1)
        self.y_grid = torch.arange(h, device='cuda').view(-1, 1)

    def generate_batch_gaussian_heatmaps(self, batch_keypoints, sigma=10.0):
        """
        生成批量高斯热力图，复用预先生成的网格
        :param batch_keypoints: list of keypoints for each image in batch
        :param sigma: 高斯核标准差
        :return: (B, 1, H, W)
        """
        h, w = self.heatmap_shape
        batch_size = len(batch_keypoints)
        heatmaps = torch.zeros((batch_size, h, w), device='cuda')

        for i, keypoints in enumerate(batch_keypoints):
            if not keypoints:
                continue

            kps_tensor = torch.tensor(keypoints, device='cuda', dtype=torch.float32)
            # 调整形状以便广播 (N, 1, 1)
            x_centers = kps_tensor[:, 0].view(-1, 1, 1)
            y_centers = kps_tensor[:, 1].view(-1, 1, 1)

            # 使用预先生成的网格
            dx = self.x_grid - x_centers  # (N, 1, W) - (N,1,1) => (N,1,W)
            dy = self.y_grid - y_centers  # (H, 1) - (N,1,1) => (N,H,1)
            # 计算平方距离 (N, H, W)
            sq_dist = dx**2 + dy**2
            gaussians = torch.exp(-sq_dist / (2 * sigma**2))

            # 聚合并归一化
            heatmap = gaussians.sum(dim=0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            heatmaps[i] = heatmap

        return heatmaps.unsqueeze(1)  # (B, 1, H, W)

    def _process_frame(self, frame_np):
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        dets = self.detector(frame_bgr, 0)
        keypoints = []
        for d in dets:
            shape = self.predictor(frame_bgr, d)
            keypoints.extend([(part.x, part.y) for part in shape.parts()])
        return keypoints

    def get_landmark(self, data_dict: dict) -> torch.tensor:
        batch_size = data_dict['image'].shape[0]

        # 转换数据到 CPU numpy
        frames = [
            (data_dict['image'][i] * 255).byte().permute(1, 2, 0).cpu().numpy()
            for i in range(batch_size)
        ]

        # 并行处理帧，提取关键点
        with ThreadPoolExecutor() as executor:
            all_keypoints = list(executor.map(self._process_frame, frames))

        # 批量计算热力图 (B, 256, 256)
        landmark_heatmaps = self.generate_batch_gaussian_heatmaps(all_keypoints, sigma=10.0)

        # (B, 1, 256, 256) 适配 PyTorch 格式
        return landmark_heatmaps

    def generate_region_noise_map(self, heatmaps, noise_strength=0.1, threshold_factor=0.1):
        """
        生成与热图激活区域对应的随机噪声图
        :param heatmaps: 输入热图张量 (B, 1, H, W)
        :param noise_strength: 噪声强度系数
        :param threshold_factor: 激活区域阈值比例
        :return: 噪声图 (B, 1, H, W)
        """
        device = heatmaps.device

        # 1. 生成基础噪声
        noise = torch.randn_like(heatmaps) * noise_strength

        # 2. 创建激活区域掩码
        with torch.no_grad():
            # 计算每个样本的自适应阈值
            batch_size = heatmaps.shape[0]
            max_vals = heatmaps.view(batch_size, -1).max(dim=1)[0]  # (B,)
            threshold = max_vals.view(-1, 1, 1, 1) * threshold_factor  # (B, 1, 1, 1)

            # 生成布尔掩码并转换为浮点型
            mask = (heatmaps > threshold).float().to(device)

        # 3. 应用区域限制
        region_noise = noise * mask

        # 4. 可选：标准化到[-1,1]范围
        # region_noise = torch.clamp(region_noise, -1.0, 1.0)

        return region_noise