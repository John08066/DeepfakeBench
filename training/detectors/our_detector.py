'''
# author: Shuaiyang chen
# date: 2025-0415
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
from torch.cuda.amp import autocast
from torchvision.transforms.functional import gaussian_blur
from torch.nn.functional import max_pool2d

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import dlib
import glob
from skimage import io
import numpy as np
import cv2

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random

from functools import partial
from sklearn.covariance import LedoitWolf
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


import kornia.morphology as morphology
import kornia.filters as filters

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='our')
class OurDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone_stream1 = self.build_backbone(config)
        self.backbone_stream2 = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.landmark_data = LandmarkProcessor()
        self.feature_enhance = FeatureEnhancer()
        # self.fusion = Fusion()
        self.L_attention = LocalAttention(    # 可以调卷积核的大小
            in_channels=2048,
            reduced_dim=256,  # 平衡效果与计算量  ,
            kernel_size=3,  # 扩大感受野 (5x5窗口)
            dilation=2  # 空洞注意力，扩大感受野 1:标准局部注意力,2扩大感受野至5x5(实际9x9), 3 等效13x13区域
        )

        self.crossfusion = SpatialCrossAttentionFusion()
        self.cs_loss = SpatialCosineSimilarityLoss()
        self.weights = nn.Parameter(torch.ones(3))

        # self.fc = nn.Sequential(
        #     nn.Linear(2 * 2048, 2048 // 16),
        #     nn.ReLU(),
        #     nn.Linear(2048 // 16, 2048),
        #     nn.Sigmoid()
        # )

    def build_backbone(self, config):   # xception 输入是多通道
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone1_name']]
        model_config = config['backbone1_config']
        backbone = backbone_class(model_config)

        # To get a good performance, use the ImageNet-pretrained Xception model
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

        # remove conv1 from state_dict
        conv1_data = state_dict.pop('conv1.weight')

        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model from {}'.format(config['pretrained']))

        # copy on conv1
        # let new conv1 use old param to balance the network
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)  # average across the RGB channels
        backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)  # repeat the averaged weights across the 4 new channels
        logger.info('Copy conv1 from pretrained model')
        return backbone



    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func


    # def channel_fusion(self, x1, x2):
    #     # 获取动态batch size
    #     batch_size = x1.size(0)
    #
    #     # 计算全局平均池化
    #     avg1 = x1.mean(dim=(2, 3))  # (B,2048)
    #     avg2 = x2.mean(dim=(2, 3))
    #     combined = torch.cat([avg1, avg2], dim=1)  # (B,4096)
    #
    #     # 动态调整view维度
    #     alpha = self.fc(combined).view(batch_size, -1, 1, 1)  # (B,2048,1,1)
    #
    #     return alpha * x1 + (1 - alpha) * x2

    # 双流网络的特征提取
    def features(self, data1,phase_fea,data2,high_freq,mask) -> torch.tensor:     # noise_data, phase_fea,data_dict['image'],high_freq
        features1 = torch.cat((data1, phase_fea), dim=1)
        x1 = self.backbone_stream1.features(features1)
        features2 = torch.cat((data2, high_freq), dim=1)
        x2 = self.extract_features(features2,mask)
        x2 = self.L_attention(x2)
        x1_det = x1.detach()
        x2_det = x2.detach()
        # x = self.fusion.gate_fusion(x1_det, x2_det)   # 可以加一个融合层
        x = self.crossfusion(x1_det, x2_det)
        return x1,x2,x

    def extract_features(self, inputs, mask):

        x = self.backbone_stream2.fea_part1(inputs)

        x = self.feature_enhance.feature_enhancement(x, mask)

        x1 = self.backbone_stream2.fea_part2(x)

        x2 = self.backbone_stream2.fea_part3(x1)

        x3 = self.backbone_stream2.fea_part4(x2)

        x = self.backbone_stream2.fea_part5(x3)

        return x


    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone_stream1.classifier(features)

    # 双流网络的损失
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred1 = pred_dict['cls1']
        pred2 = pred_dict['cls2']
        pred3 = pred_dict['cls']
        loss1 = self.loss_func(pred1, label)
        loss2 = self.loss_func(pred2, label)
        loss3 = self.loss_func(pred3, label)
        cs_loss = self.cs_loss(pred_dict['feat1'], pred_dict['feat2'])
        loss = loss1+loss2+loss3+cs_loss
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    # def interpolate(self, img, factor):  # npr
    #     return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
    #                          scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)
    #
    # def get_npr(self, weighted_data) -> dict: # npr
    #     npr_data = weighted_data - self.interpolate(weighted_data, 0.5)
    #     return npr_data

    def fuse_pred(self,pred1, pred2, pred3,mode='learnable'):

        if mode=='learnable':
            # 归一化权重，确保和为1
            normalized_weights = torch.softmax(self.weights, dim=0)

            # 加权融合logits
            final= (normalized_weights[0] * pred1 +
                    normalized_weights[1] * pred2 +
                    normalized_weights[2] * pred3)

        elif mode=='mean':
            final = (pred1 + pred2 + pred3) / 3

        return final


    def forward(self, data_dict: dict, inference=False) -> dict:

        # get npr
        # npr_data = self.get_npr(data_dict['image'])

        # get heatmaps
        # landmark_heatmaps = self.landmark_data.get_landmark(data_dict)

        # get phase
        phase_fea,high_freq = self.phase_without_amplitude(data_dict['image'])

        #get noise and mask
        landmark_heatmaps = self.landmark_data.get_landmark(data_dict)
        noise, mask = self.landmark_data.generate_region_noise_map(landmark_heatmaps)

        if inference==False:
            # get heatmaps
            # landmark_heatmaps = self.landmark_data.get_landmark(data_dict)
            # enhanced_data = data_dict['image']+landmark_heatmaps*data_dict['image']
            # get noise and mask
            # noise = self.landmark_data.generate_region_noise_map(landmark_heatmaps)
            # mask = self.landmark_data.get_masks(data_dict)
            # noise = self.generate_region_noise(mask)
            noise_data = 0.5*noise + data_dict['image']
            x1,x2,features = self.features(noise_data, phase_fea,data_dict['image'],high_freq,mask)
        else:
            x1,x2,features = self.features(data_dict['image'], phase_fea,data_dict['image'],high_freq,mask)


        # get the features by backbone
        # features = self.features(noise_data, phase_fea)
        pred1 = self.classifier(x1)
        pred2 = self.classifier(x2)
        # get the prediction by classifier
        pred = self.classifier(features) # 后面的2都可改为1

        pred1_d = pred1.detach()
        pred2_d = pred2.detach()

        fuse_pred = self.fuse_pred(pred1_d, pred2_d, pred)
        # get the probability of the pred
        prob = torch.softmax(fuse_pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': fuse_pred,'cls1': pred1,'cls2': pred2, 'prob': prob,'feat': features,'feat1': x1,'feat2': x2}  #,
        return pred_dict

    def phase_without_amplitude(self, img):
        # Convert to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True)  # (B, 1, H, W)

        # Compute DFT
        X = torch.fft.fftn(gray_img, dim=(-2, -1))  # (B, 1, H, W)

        # Extract phase and reconstruct (original logic)
        phase_spectrum = torch.angle(X)
        reconstructed_X = torch.exp(1j * phase_spectrum)
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X, dim=(-2, -1)))

        # --- 高通滤波修正部分 ---
        # 1. 将频谱中心化 (fftshift)
        X_shifted = torch.fft.fftshift(X, dim=(-2, -1))

        # 2. 获取中心坐标 (假设图像尺寸为 256x256)
        B, C, H, W = X_shifted.shape
        crow, ccol = H // 2, W // 2

        # 3. 创建掩码 (中心区域置零)
        mask = torch.ones_like(X_shifted)
        mask[..., crow - 15:crow + 15, ccol - 15:ccol + 15] = 0    # 这里可以调

        # 4. 应用高通滤波
        X_filtered_shifted = X_shifted * mask

        # 5. 逆中心化 (ifftshift)
        X_filtered = torch.fft.ifftshift(X_filtered_shifted, dim=(-2, -1))

        # 6. 逆变换回空域
        x_filtered = torch.real(torch.fft.ifftn(X_filtered, dim=(-2, -1)))

        return reconstructed_x, x_filtered  # 返回相位重建结果和高通滤波结果


class LandmarkProcessor:
    def __init__(self, heatmap_shape=(256, 256)):
        self.heatmap_shape = heatmap_shape
        self._init_grid(heatmap_shape)

    def _init_grid(self, shape):
        h, w = shape
        self.x_grid = torch.arange(w, device='cuda').view(1, -1)  # (1, W)
        self.y_grid = torch.arange(h, device='cuda').view(-1, 1)  # (H, 1)

    def generate_batch_gaussian_heatmaps(self, batch_keypoints, sigma=10.0):
        """
        优化后的全张量操作版本
        :param batch_keypoints: (B, N, 2) 位于CUDA的坐标张量
        :param sigma: 高斯核标准差
        :return: (B, 1, H, W)
        """
        h, w = self.heatmap_shape
        batch_size = batch_keypoints.size(0)

        # 调整坐标形状 (B, N, 2) -> (B, N, 1, 1)
        x_centers = batch_keypoints[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        y_centers = batch_keypoints[..., 1].unsqueeze(-1).unsqueeze(-1)

        # 广播计算距离矩阵
        dx = self.x_grid - x_centers  # (B, N, 1, W)
        dy = self.y_grid - y_centers  # (B, N, H, 1)
        sq_dist = dx.pow(2) + dy.pow(2)  # (B, N, H, W)

        # 批量计算高斯响应
        gaussians = torch.exp(-sq_dist / (2 * sigma ** 2))

        # 聚合响应并安全归一化
        heatmaps = gaussians.sum(dim=1)  # (B, H, W)
        max_vals = heatmaps.amax(dim=(1, 2), keepdim=True)
        heatmaps = heatmaps / torch.where(max_vals > 1e-6, max_vals, torch.ones_like(max_vals))

        return heatmaps.unsqueeze(1)  # (B, 1, H, W)

    def get_landmark(self, data_dict: dict) -> torch.tensor:
        """直接使用预存的CUDA张量"""
        return self.generate_batch_gaussian_heatmaps(data_dict['landmark'])


    def generate_region_noise_map(self, heatmaps, noise_strength=0.2, threshold_factor=0.1): # 这里也可以调
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

        # 形态学优化
        mask = max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = gaussian_blur(mask, kernel_size=5, sigma=1.0)

        # 3. 应用区域限制
        region_noise = noise * mask

        # 4. 可选：标准化到[-1,1]范围
        # region_noise = torch.clamp(region_noise, -1.0, 1.0)

        return region_noise,mask


class LocalAttention(nn.Module):
    def __init__(self, in_channels=2048, reduced_dim=256, kernel_size=3, dilation=1):
        """
        局部注意力模块
        Args:
            in_channels: 输入通道数 (默认2048)
            reduced_dim: 降维通道数 (默认256)
            kernel_size: 局部窗口大小 (默认3x3)
            dilation: 空洞率 (默认1)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # 通道降维 (减少计算量)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_dim, kernel_size=1)

        # 注意力参数生成
        self.query_conv = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=1)

        # 通道恢复
        self.conv_restore = nn.Conv2d(reduced_dim, in_channels, kernel_size=1)

        # 归一化
        self.norm = nn.BatchNorm2d(in_channels)
        self.softmax = nn.Softmax(dim=-1)

        # 位置编码
        self.pos_encoding = nn.Conv2d(2, reduced_dim, kernel_size=1)  # 坐标编码

    def forward(self, x):
        x = torch.relu(x)
        B, C, H, W = x.shape
        identity = x  # 保留原始特征

        # 1. 通道降维 (2048 -> 256)
        x_reduced = self.conv_reduce(x)

        # 2. 生成位置编码 (相对坐标)
        h_range = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        w_range = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        pos_grid = torch.cat([h_range, w_range], dim=1)
        pos_feat = self.pos_encoding(pos_grid)
        x_reduced = x_reduced + pos_feat  # 加入位置信息

        # 3. 计算Query/Key/Value
        Q = self.query_conv(x_reduced)  # (B, 256, 8, 8)
        K = self.key_conv(x_reduced)  # (B, 256, 8, 8)
        V = self.value_conv(x_reduced)  # (B, 256, 8, 8)

        # 4. 提取局部窗口 (使用unfold)
        pad = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)) // 2
        K_unfold = F.unfold(K, kernel_size=self.kernel_size, dilation=self.dilation, padding=pad)  # (B, 256*9, 64)
        V_unfold = F.unfold(V, kernel_size=self.kernel_size, dilation=self.dilation, padding=pad)  # (B, 256*9, 64)

        # 5. 重塑维度 (准备注意力计算)
        Q = Q.view(B, -1, H * W).permute(0, 2, 1)  # (B, 64, 256)
        K_unfold = K_unfold.view(B, -1, self.kernel_size * self.kernel_size, H * W)  # (B, 256, 9, 64)
        V_unfold = V_unfold.view(B, -1, self.kernel_size * self.kernel_size, H * W)  # (B, 256, 9, 64)

        # 6. 计算局部注意力
        attn_scores = torch.einsum('bic,bcjk->bij', Q, K_unfold)  # (B, 64, 9)
        attn_scores = self.softmax(attn_scores / (256 ** 0.5))  # 缩放点积

        # 7. 注意力加权聚合
        attended = torch.einsum('bij,bcjk->bic', attn_scores, V_unfold)  # (B, 64, 256)
        attended = attended.permute(0, 2, 1).view(B, -1, H, W)  # (B, 256, 8, 8)

        # 8. 恢复通道维度 (256 -> 2048)
        attended = self.conv_restore(attended)

        # 9. 残差连接 + 归一化
        return self.norm(identity + attended)

class SpatialCrossAttentionFusion(nn.Module):
    def __init__(self, dim=2048, heads=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape

        # reshape (B, C, H, W) -> (B, H*W, C)
        f1_seq = f1.view(B, C, -1).permute(0, 2, 1)
        f2_seq = f2.view(B, C, -1).permute(0, 2, 1)

        Q = self.q_proj(f1_seq)
        K = self.k_proj(f2_seq)
        V = self.v_proj(f2_seq)

        attn_out, _ = self.attn(Q, K, V)  # (B, 64, C)

        # reshape back to (B, C, H, W)
        fused = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return fused

class SpatialCosineSimilarityLoss(nn.Module):
    def __init__(self, lambda_weight=0.8):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.cos = nn.CosineSimilarity(dim=2)  # 修改dim为2

    def forward(self, x1, x2):
        # 输入形状: (B, 2048, 8, 8)
        B, C, H, W = x1.shape

        # 展开空间维度 -> (B, C, H*W)
        x1_flat = x1.view(B, C, H * W)  # (B, 2048, 64)
        x2_flat = x2.view(B, C, H * W)

        # 转置为 (B, H*W, C) 以便逐位置计算
        x1_flat = x1_flat.permute(0, 2, 1)  # (B, 64, 2048)
        x2_flat = x2_flat.permute(0, 2, 1)

        # 逐位置计算余弦相似度 -> (B, H*W)
        similarity = self.cos(x1_flat, x2_flat)  # 在dim=2计算

        # 计算平均损失（同时考虑B和H*W维度）
        loss = torch.mean(torch.abs(similarity))

        return self.lambda_weight * loss

class FeatureEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def feature_enhancement(self, features, mask):
        # 调整热图尺寸至与特征图一致
        adjusted_mask = F.interpolate(
            mask,
            size=features.shape[2:],  # 目标尺寸为中间层的H, W
            mode='bilinear',
            align_corners=False
        )

        adjusted_mask = torch.sigmoid(adjusted_mask)

        # 应用空间注意力：特征与权重逐元素相乘
        enhanced_features = features * adjusted_mask  # 广播机制自动扩展通道

        # 残差连接（可选）
        enhanced_features = features + self.gamma * enhanced_features

        return enhanced_features