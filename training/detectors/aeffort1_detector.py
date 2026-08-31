import os
import math
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

# --- 1. 引入 PEFT 库 ---
from peft import LoraConfig, get_peft_model

from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='aeffort1')
class Aeffort1Detector(nn.Module):
    def __init__(self, config=None):
        super(Aeffort1Detector, self).__init__()
        self.config = config

        # 构建带 LoRA 的骨干网络
        self.backbone = self.build_backbone(config)

        # 分类头 (全精度训练)
        # self.head = nn.Linear(1024, 2)

        self.loss_func = nn.CrossEntropyLoss()

        # vae 分类头
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 2)
        )

        # -------初始化 VAE 增强模块----------
        # clip_mean_list = config['mean']
        # clip_std_list = config['std']
        #
        # clip_mean_tensor = torch.tensor(clip_mean_list)
        # clip_std_tensor = torch.tensor(clip_std_list)
        #
        # self.vae_augmenter = VAEDataAugmentation(config['vae_path'], clip_mean_tensor, clip_std_tensor)

        #-------------------------------------

        # -------初始化 VAE 增强模块----------
        clip_mean_list = config['mean']
        clip_std_list = config['std']

        clip_mean_tensor = torch.tensor(clip_mean_list)
        clip_std_tensor = torch.tensor(clip_std_list)

        # 获取配置中的 patch_size，默认为 32
        # 你也可以在 config 文件里加一个字段 'patch_shuffle': True 来控制
        patch_size = config.get('patch_size', 32)
        do_shuffle = config.get('vae_shuffle', True)  # 默认开启打乱

        self.vae_augmenter = VAEDataAugmentation(
            config['vae_path'],
            clip_mean_tensor,
            clip_std_tensor,
            patch_size=patch_size,
            do_shuffle=do_shuffle
        )
        # -------------------------------------

    def build_backbone(self, config):
        # 1. 加载预训练的 CLIP 模型
        # 请根据你的实际路径修改
        clip_path = "/root/csy-7pw03c/disk/project1/DeepfakeBench-main/training/pretrained/clip14/"
        try:
            # 优先加载本地
            clip_model = CLIPModel.from_pretrained(clip_path)
            logger.info(f"Loaded CLIP from local path: {clip_path}")
        except OSError:
            # 本地失败则加载 HuggingFace
            logger.warning(f"Local path {clip_path} not found, trying huggingface hub...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        # 我们只需要 CLIP 的 Vision 部分
        vision_model = clip_model.vision_model

        # 2. 配置 LoRA (PEFT)
        # r: 秩，通常 8, 16, 32
        # lora_alpha: 缩放因子，通常与 r 相同或 r 的 2 倍
        # target_modules: 指定在哪些层应用 LoRA。
        # 对于 CLIP/ViT，["q_proj", "v_proj"] 是最经典且高效的选择。
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        # 3. 将 LoRA 注入模型
        # get_peft_model 会自动冻结原模型参数，并初始化 LoRA 参数
        vision_model = get_peft_model(vision_model, peft_config)

        # 4. 打印参数统计
        # 这会显示 Trainable params (LoRA) vs All params
        print("Scucessfully applied PEFT LoRA:")
        vision_model.print_trainable_parameters()

        return vision_model

    # def features(self, data_dict: dict) -> torch.tensor:
    #     # PEFT 模型的调用方式与原模型一致
    #     feat = self.backbone(data_dict['image'])['pooler_output']
    #     return feat

    def features(self, data) -> torch.tensor:
        # PEFT 模型的调用方式与原模型一致
        feat = self.backbone(data)['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']  # [batch_size]
        pred = pred_dict['cls']  # [batch_size, 2]

        # 1. 计算总的交叉熵损失
        loss = self.loss_func(pred, label)
        loss_dict = {
            'overall': loss,
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # 计算常规指标
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    # def forward(self, data_dict: dict, inference=False) -> dict:
    #     # 1. 提取特征 (LoRA backbone)
    #     features = self.features(data_dict)
    #
    #     # 2. 分类 (Linear head)
    #     pred = self.classifier(features)
    #
    #     # 3. 计算概率 (取类别1即Fake的概率)
    #     prob = torch.softmax(pred, dim=1)[:, 1]
    #
    #     pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
    #
    #     return pred_dict


    # 通过vae重建和原图之间的差值进行判断
    def forward(self, data_dict: dict, inference=False) -> dict:
        images = data_dict['image']
        labels = data_dict['label']

        # 1. 获取 VAE 重建图
        processed_images = self.vae_augmenter(images, labels)

        # 2. 分别提取特征
        # feat_orig: 原图特征 (包含伪影信息)
        feat_orig = self.features(images)

        # feat_vae: 重建特征 (伪影被抹除，只剩内容)
        # 注意：我们要detach，不让梯度传回VAE，只把VAE当工具人
        feat_vae = self.features(processed_images.detach())

        # 3. 核心创新点：计算特征差异
        # 逻辑：如果 feat_diff 很大，说明是假图；如果很小，说明是真图
        feat_diff = feat_orig - feat_vae


        # cos_sim = F.cosine_similarity(feat_orig, feat_vae, dim=1).unsqueeze(1)

        # 也可以尝试把它们拼起来： cat([feat_orig, feat_diff], dim=1)
        # 这里简单起见，直接用差异特征去分类，或者用 orig + diff
        # final_feat = feat_diff  # 类似于残差连接，强调差异

        final_feat = torch.cat([feat_orig, feat_diff], dim=1)

        # 4. 分类
        pred = self.classifier(final_feat)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': final_feat}


# --------------VAE 模块---------------------
class VAEDataAugmentation(nn.Module):
    """
    一个封装了 VAE 重建逻辑的模块。
    新增功能：支持 Patch Shuffle (分块打乱)。
    """

    def __init__(self, vae_path, clip_mean, clip_std, patch_size=32, do_shuffle=True):
        super(VAEDataAugmentation, self).__init__()

        # VAE 期望的输入尺寸
        self.vae_input_size = (256, 256)

        # 新增控制参数
        self.patch_size = patch_size
        self.do_shuffle = do_shuffle

        # 1. 注册参数
        self.register_buffer('clip_mean', clip_mean.view(1, -1, 1, 1))
        self.register_buffer('clip_std', clip_std.view(1, -1, 1, 1))

        vae_mean = torch.tensor([0.5, 0.5, 0.5])
        vae_std = torch.tensor([0.5, 0.5, 0.5])
        self.register_buffer('vae_mean', vae_mean.view(1, -1, 1, 1))
        self.register_buffer('vae_std', vae_std.view(1, -1, 1, 1))

        # 2. 加载 VAE
        logger.info(f"Loading VAE for augmentation from: {vae_path}")
        try:
            self.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        except Exception as e:
            logger.error(f"Failed to load VAE in float16: {e}. Attempting full precision.")
            self.vae = AutoencoderKL.from_pretrained(vae_path)

        # 3. 冻结 VAE
        self.vae.requires_grad_(False)
        self.vae.eval()

    # --- 保持原有的归一化函数不变 ---
    def clip_denormalize(self, x):
        return x * self.clip_std + self.clip_mean

    def clip_normalize(self, x):
        return (x - self.clip_mean) / self.clip_std

    def vae_denormalize(self, x):
        return x * self.vae_std + self.vae_mean

    def vae_normalize(self, x):
        return (x - self.vae_mean) / self.vae_std

    # --- 核心：分块打乱函数 ---
    def shuffle_patches(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # 如果尺寸不能整除，为了安全直接返回原图
        if H % p != 0 or W % p != 0:
            return x

        h_patches = H // p
        w_patches = W // p
        num_patches = h_patches * w_patches

        # 1. 拆分为 Patch
        # [B, C, H, W] -> [B, C, h_patches, p, w_patches, p]
        x = x.view(B, C, h_patches, p, w_patches, p)

        # 调整维度，将 patch 维度放在一起
        # [B, h_patches, w_patches, C, p, p]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()

        # 展平 patch 数量维度
        # [B, num_patches, C, p, p]
        x = x.view(B, num_patches, C, p, p)

        # 2. 生成随机索引
        # [B, num_patches]
        rand_score = torch.rand(B, num_patches, device=x.device)
        rand_idx = torch.argsort(rand_score, dim=1)

        # 3. 扩展索引以匹配数据维度用于 gather
        # [B, num_patches, C, p, p]
        expanded_idx = rand_idx.view(B, num_patches, 1, 1, 1).expand(-1, -1, C, p, p)

        # 4. 执行打乱
        x_shuffled = torch.gather(x, 1, expanded_idx)

        # 5. 恢复形状
        # [B, h_patches, w_patches, C, p, p]
        x_shuffled = x_shuffled.view(B, h_patches, w_patches, C, p, p)

        # [B, C, h_patches, p, w_patches, p]
        x_shuffled = x_shuffled.permute(0, 3, 1, 4, 2, 5).contiguous()

        # [B, C, H, W]
        x_shuffled = x_shuffled.view(B, C, H, W)

        return x_shuffled

    @torch.no_grad()
    def forward(self, images, labels=None):
        self.vae.eval()  # 双重保险

        original_size = (images.shape[2], images.shape[3])
        vae_dtype = self.vae.dtype
        img_dtype = images.dtype

        # 1. 预处理 (CLIP空间 -> VAE空间)
        x = self.clip_denormalize(images)
        x_vae_norm = self.vae_normalize(x)

        # Resize 到 256x256 (VAE 训练尺寸)
        x_resized = F.interpolate(x_vae_norm, size=self.vae_input_size, mode='bilinear', align_corners=False)

        # 2. 【核心修改】在此处进行分块打乱
        if self.do_shuffle:
            # 输入 256x256，默认 patch_size=32，会切成 8x8=64 块并打乱
            vae_input_tensor = self.shuffle_patches(x_resized)
        else:
            vae_input_tensor = x_resized

        # 3. VAE 重建
        vae_input = vae_input_tensor.to(vae_dtype)

        # Encode -> Mode -> Decode
        posterior = self.vae.encode(vae_input).latent_dist
        latents = posterior.mode()
        reconstructed = self.vae.decode(latents).sample

        reconstructed_f32 = reconstructed.to(img_dtype)

        # 4. 后处理 (VAE空间 -> CLIP空间)
        # Resize 回原图尺寸
        reconstructed_resized = F.interpolate(reconstructed_f32, size=original_size, mode='bilinear',
                                              align_corners=False)

        out_0_1 = self.vae_denormalize(reconstructed_resized)
        processed_images = self.clip_normalize(out_0_1)

        return processed_images