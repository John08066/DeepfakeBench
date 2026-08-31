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


@DETECTOR.register_module(module_name='aeffort')
class AeffortDetector(nn.Module):
    def __init__(self, config=None):
        super(AeffortDetector, self).__init__()
        self.config = config

        # 构建带 LoRA 的骨干网络
        self.backbone = self.build_backbone(config)

        # 分类头 (全精度训练)
        # self.head = nn.Linear(1024, 2)

        self.loss_func = nn.CrossEntropyLoss()

        # 一个简单的 MLP 生成权重 (类似 SE-Block)
        # self.gate = nn.Sequential(
        #     nn.Linear(1024, 1024 // 4),
        #     nn.ReLU(),
        #     nn.Linear(1024 // 4, 1024),
        #     nn.Sigmoid()
        # )

        # vae 分类头
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),  # 2560
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 2)
        )

        # -------初始化 VAE 增强模块----------
        clip_mean_list = config['mean']
        clip_std_list = config['std']

        clip_mean_tensor = torch.tensor(clip_mean_list)
        clip_std_tensor = torch.tensor(clip_std_list)

        self.vae_augmenter = VAEDataAugmentation(config['vae_path'], clip_mean_tensor, clip_std_tensor)
        #-------------------------------------


        #----------相似度计算模块初始化----------
        # self.relation_extractor = RelationDifferenceExtractor(
        #     input_dim=1024,  # 你的特征维度
        #     reduced_dim=128,  # 内部降维大小
        #     output_dim=512  # 你希望这个模块输出多少维特征
        # )
        #-------------------------------------


        #-----------初始化正交投影模块----------
        # self.ortho_projector = OrthogonalProjection()
        #-------------------------------------

        #-------------特征融合-------------
        # self.feature_fusion = FeatureFusion()
        #---------------------------------


    def build_backbone(self, config):
        # 1. 加载预训练的 CLIP 模型
        # 请根据你的实际路径修改
        clip_path = "/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pretrained/clip14/"
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

        # 2. 计算相似度损失
        cos_sim = F.cosine_similarity(pred_dict['feat_orig'],pred_dict['feat_vae'])

        loss_sim = torch.mean(
            torch.where(
                data_dict['label'] == 1,
                1.0-cos_sim,
                F.relu(cos_sim-0.5)
            )
        )

        loss = loss + loss_sim

        loss_dict = {
            'overall': loss
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

        #自相似度差异
        feat_relation = self.relation_extractor(feat_orig, feat_vae) #(B,256)

        #拼接 feat_diff 和 feat_relation
        # feat_fusion = self.feature_fusion(feat_diff, feat_relation)

        #提取正交分量
        feat_orth, _ = self.ortho_projector(feat_orig, feat_vae)

        # 权重学习，学习那些维度差异更重要
        # attention_weight = self.gate(torch.abs(feat_diff))

        # if inference == False:
        #     # lambda_val 控制替换的强度，比如 0.3
        #     lambda_val = 0.5
        #
        #     # 计算混合系数：权重越大，越倾向于被 feat_vae 替换（被修复）
        #     mix_ratio = lambda_val * attention_weight
        #
        #     # 混合操作
        #     augmented_feat = (1 - mix_ratio) * feat_orig + mix_ratio * feat_vae
        #
        #     # 拼接
        #     final_feat = torch.cat([augmented_feat, feat_diff], dim=1)
        # else:
        #     final_feat = torch.cat([feat_orig, feat_diff], dim=1)

        #通过噪声扰动
        # if inference == False:
        #     noise = torch.randn_like(feat_orig)
        #     # 权重越大，加入的噪声越多，破坏该特征的特定指纹
        #     noise_strength = 0.1
        #     augmented_feat = feat_orig + noise * attention_weight * noise_strength
        #     final_feat = torch.cat([augmented_feat, feat_diff], dim=1)
        # else:
        #     final_feat = torch.cat([feat_orig, feat_diff], dim=1)


        # 5. 拼接特征
        final_feat = torch.cat([feat_orig, feat_relation], dim=1)

        # 4. 分类
        pred = self.classifier(final_feat)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': final_feat, 'feat_orig': feat_orig, 'feat_vae': feat_vae}

# ----------------融合模块-----------------
class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 定义压缩层：把 feat_diff 从 1024 压到 512
        self.compress_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # 建议加BN，防止特征分布差异太大
            # nn.ReLU()  # 激活函数
        )

    def forward(self, feat_diff, feat_relation):
        # feat_diff: [batch, 1024]
        # feat_relation: [batch, 512]

        # 1. 压缩 diff
        # shape: [batch, 512]
        diff_reduced = self.compress_layer(feat_diff)

        # 2. 拼接
        # 512 (diff_reduced) + 512 (relation) = 1024
        # shape: [batch, 1024]
        out = torch.cat([diff_reduced, feat_relation], dim=1)

        return out
# ------------------------------------------


# --------------VAE 模块---------------------
class VAEDataAugmentation(nn.Module):
    """
    一个封装了 VAE 重建逻辑的模块。

    更新：现在对 batch 中的所有图片（无论真假）都进行 VAE 重建。
    """

    def __init__(self, vae_path, clip_mean, clip_std):
        super(VAEDataAugmentation, self).__init__()

        # VAE 期望的输入尺寸
        self.vae_input_size = (256, 256)

        # 1. 注册 CLIP 和 VAE 的归一化/反归一化参数
        self.register_buffer('clip_mean', clip_mean.view(1, -1, 1, 1))
        self.register_buffer('clip_std', clip_std.view(1, -1, 1, 1))

        vae_mean = torch.tensor([0.5, 0.5, 0.5])
        vae_std = torch.tensor([0.5, 0.5, 0.5])
        self.register_buffer('vae_mean', vae_mean.view(1, -1, 1, 1))
        self.register_buffer('vae_std', vae_std.view(1, -1, 1, 1))

        # 2. 加载 VAE
        logger.info(f"Loading VAE for augmentation from: {vae_path}")
        try:
            # 使用 float16 加载以节省显存
            self.vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=torch.float16
            )
        except Exception as e:
            logger.error(f"Failed to load VAE in float16: {e}. Attempting full precision.")
            self.vae = AutoencoderKL.from_pretrained(vae_path)

        # 3. 冻结 VAE
        self.vae.requires_grad_(False)
        self.vae.eval()

    # --- 归一化辅助函数 (保持不变) ---
    def clip_denormalize(self, x):
        return x * self.clip_std + self.clip_mean

    def clip_normalize(self, x):
        return (x - self.clip_mean) / self.clip_std

    def vae_denormalize(self, x):
        return x * self.vae_std + self.vae_mean

    def vae_normalize(self, x):
        return (x - self.vae_mean) / self.vae_std

    @torch.no_grad()  # VAE 冻结，不计算梯度
    def forward(self, images, labels=None):
        """
        输入:
            images: [B, 3, 224, 224] (CLIP Normalized)
            labels: [B] (为了接口兼容保留，实际不再使用)
        输出:
            processed_images: [B, 3, 224, 224] (全量经过 VAE 重建并恢复到 CLIP Normalized)
        """

        # 确保 VAE 在评估模式
        self.vae.eval()

        # 1. 准备元数据
        original_size = (images.shape[2], images.shape[3])  # (224, 224)
        vae_dtype = self.vae.dtype  # torch.float16
        img_dtype = images.dtype  # torch.float32

        # 2. 预处理：反归一化 CLIP -> 归一化 VAE -> Resize (全在 float32 下进行)
        # 此时 x 是 [0, 1] 范围
        x = self.clip_denormalize(images)

        # 此时 x 是 [-1, 1] 范围 (VAE 标准)
        x_vae_norm = self.vae_normalize(x)

        # Resize 到 (256, 256)
        x_vae_norm_resized = F.interpolate(
            x_vae_norm,
            size=self.vae_input_size,
            mode='bilinear',
            align_corners=False
        )

        # 3. VAE 重建 (关键步骤)
        # 转换精度: float32 -> float16
        vae_input = x_vae_norm_resized.to(vae_dtype)

        # VAE Inference
        reconstructed = self.vae(vae_input).sample

        # 转换精度: float16 -> float32
        reconstructed_f32 = reconstructed.to(img_dtype)

        # 4. 后处理：Resize 回原图 -> 反归一化 VAE -> 归一化 CLIP
        reconstructed_resized = F.interpolate(
            reconstructed_f32,
            size=original_size,  # (224, 224)
            mode='bilinear',
            align_corners=False
        )

        # 此时 x 回到 [0, 1]
        out_0_1 = self.vae_denormalize(reconstructed_resized)

        # 此时 x 回到 CLIP 分布
        processed_images = self.clip_normalize(out_0_1)

        return processed_images


# ----------------------------------------------------


# --------------------相似度计算模块------------------
class RelationDifferenceExtractor(nn.Module):
    def __init__(self, input_dim=1024, reduced_dim=64, output_dim=256):
        """
        Args:
            input_dim: 输入特征维度 (1024)
            reduced_dim: 降维大小 (64 -> 64x64 Gram矩阵)
            output_dim: 输出特征维度 (256)
        """
        super(RelationDifferenceExtractor, self).__init__()

        # [优化1] 去掉 ReLU。
        # 原因：我们需要保留特征的正负号，以便在 Gram 矩阵中计算“负相关性”。
        # 如果加上 ReLU，特征全变为正数，无法表达"此消彼长"的关系。
        self.reducer = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.BatchNorm1d(reduced_dim)
            # Remove ReLU here to preserve negative correlations
        )

        gram_feature_size = reduced_dim * reduced_dim

        # [优化2] 增加 LayerNorm。
        # 原因：Gram 矩阵是二阶统计量，数值范围变化大，
        # 在送入 MLP 前做一次归一化能显著提升训练稳定性。
        self.norm_layer = nn.LayerNorm(gram_feature_size)

        self.relation_mlp = nn.Sequential(
            nn.Linear(gram_feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # [优化3] 增加 Dropout 防止过拟合
            nn.Linear(512, output_dim)
        )

    def compute_gram_matrix(self, x):
        """
        计算特征向量的自相似矩阵
        x: [batch, reduced_dim]
        """
        # [b, dim, 1] * [b, 1, dim] -> [b, dim, dim]
        gram = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))

        # 归一化：除以 sqrt(dim) 是 Transformer 的做法，
        # 但在 Gram 矩阵中，通常除以 dim 更加稳定，或者不除（靠后面的 BN/LN 处理）
        # 这里保留你的写法，只要后面有 LayerNorm 就没问题。
        return gram / (x.shape[1] ** 0.5)

    def forward(self, feat_orig, feat_recon):
        batch_size = feat_orig.size(0)

        # [优化4] 合并计算，减少 GPU Kernel 调用次数
        # 将 orig 和 recon 拼起来一起过降维层
        feats_concat = torch.cat([feat_orig, feat_recon], dim=0)  # [2B, 1024]
        z_concat = self.reducer(feats_concat)  # [2B, 64]

        # 拆分
        z_orig, z_recon = torch.split(z_concat, batch_size, dim=0)

        # 计算 Gram 矩阵
        gram_orig = self.compute_gram_matrix(z_orig)  # [B, 64, 64]
        gram_recon = self.compute_gram_matrix(z_recon)  # [B, 64, 64]

        # 计算差异
        gram_diff = gram_orig - gram_recon  # [B, 64, 64]

        # 展平
        gram_diff_flat = gram_diff.view(batch_size, -1)  # [B, 4096]

        # [关键] 归一化后再送入 MLP
        gram_diff_norm = self.norm_layer(gram_diff_flat)

        # 提取最终特征
        diff_relation_feat = self.relation_mlp(gram_diff_norm)  # [B, 256]

        return diff_relation_feat

# ----------------------------------------------------

# --------------------特征正交分解------------------
class OrthogonalProjection(nn.Module):
    def __init__(self):
        super(OrthogonalProjection, self).__init__()

    def forward(self, feat_orig, feat_vae):
        """
        输入:
            feat_orig: 原图 CLIP 特征, 维度 (B, 1024)
            feat_vae:  VAE 重构图 CLIP 特征, 维度 (B, 1024)
        输出:
            feat_orth: 垂直分量 (伪造痕迹), 维度 (B, 1024)
            feat_para: 平行分量 (语义内容), 维度 (B, 1024)
        """

        # 1. 对基准向量 (feat_vae) 进行归一化，得到单位方向向量
        # eps 避免除零错误
        feat_vae_norm = F.normalize(feat_vae, p=2, dim=1, eps=1e-8)

        # 2. 计算投影量 (标量)
        # 公式: s = u · v_hat
        # (B, 1024) * (B, 1024) -> sum -> (B, 1)
        projection_scalar = torch.sum(feat_orig * feat_vae_norm, dim=1, keepdim=True)

        # 3. 计算平行分量 (Semantic Component)
        # 公式: F_para = s * v_hat
        # 这个分量代表原图中"可以被重构图解释"的部分 (如身份、姿态)
        feat_para = projection_scalar * feat_vae_norm

        # 4. 计算垂直分量 (Noise/Artifact Component)
        # 公式: F_perp = u - F_para
        # 这个分量代表原图中"无法被重构图解释"的部分 (即差异性、伪影)
        feat_orth = feat_orig - feat_para

        return feat_orth, feat_para
# ----------------------------------------------------