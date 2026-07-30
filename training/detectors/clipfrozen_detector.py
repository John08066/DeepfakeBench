import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from detectors import DETECTOR

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clipfrozen')
class ClipFrozenDetector(nn.Module):
    def __init__(self, config=None):
        super(ClipFrozenDetector, self).__init__()
        self.config = config

        # 1. 加载骨干网络并冻结
        self.backbone = self.build_backbone(config)

        # 2. 定义分类损失
        self.loss_func = nn.CrossEntropyLoss()

        # 3. 分类头 (这是唯一需要训练的部分)
        # CLIP-ViT-L/14 的 pooler_output 维度是 1024
        self.head = nn.Linear(1024, 2)

        logger.info("Detector 'clip_frozen' initialized. Backbone is locked.")

    def build_backbone(self, config):
        # 使用你原本代码中的路径
        clip_path = "/root/csy-7pw03c/disk/project1/DeepfakeBench-main/training/pretrained/clip14/"
        try:
            clip_model = CLIPModel.from_pretrained(clip_path)
            logger.info(f"Loaded CLIP from local path: {clip_path}")
        except Exception as e:
            logger.warning(f"Local path failed, loading from HF: {e}")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        # 提取 Vision 部分
        vision_model = clip_model.vision_model

        # --- 核心步骤：冻结所有参数 ---
        for param in vision_model.parameters():
            param.requires_grad = False

        # 确保 backbone 处于 eval 模式（关闭 Dropout 等）
        vision_model.eval()

        # 统计参数
        trainable_params = sum(p.numel() for p in vision_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in vision_model.parameters())
        print(f"Backbone frozen. Trainable params: {trainable_params} / {all_params}")

        return vision_model

    def features(self, data) -> torch.tensor:
        # 在冻结模式下，强制不计算梯度以节省显存和计算资源
        with torch.no_grad():
            outputs = self.backbone(data)
            feat = outputs['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        # 获取图像数据
        images = data_dict['image']

        # 1. 提取冻结的 CLIP 特征
        features = self.features(images)

        # 2. 通过可训练的分类头
        pred = self.classifier(features)

        # 3. 计算概率
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {'cls': pred, 'prob': prob, 'feat': features}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}