import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from detectors import DETECTOR
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='lora2')
class Lora2Detector(nn.Module):
    def __init__(self, config=None):
        super(Lora2Detector, self).__init__()
        self.config = config

        self.backbone = self.build_backbone(config)
        self.loss_func = nn.CrossEntropyLoss()

        clip_mean_tensor = torch.tensor(config['mean'])
        clip_std_tensor = torch.tensor(config['std'])

        perturb_config = self.resolve_perturbation_config(config)
        self.config['perturbation_name'] = perturb_config['name']
        self.config['perturbation_type'] = perturb_config['type']
        self.image_perturbation = ImagePerturbation(
            clip_mean_tensor,
            clip_std_tensor,
            mode=perturb_config['type'],
            kernel_size=perturb_config['kernel_size'],
            sigma=perturb_config['sigma'],
            noise_std=perturb_config['noise_std'],
            sharpen_amount=perturb_config['sharpen_amount'],
        )

        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )

    @staticmethod
    def resolve_perturbation_config(config):
        perturb_name = config.get('perturbation_name', config.get('perturbation_type', 'blur'))
        candidates = config.get('perturbation_candidates') or {}
        if not isinstance(candidates, dict):
            raise ValueError("perturbation_candidates must be a dict mapping names to settings.")
        if candidates and perturb_name not in candidates:
            available = ', '.join(candidates.keys())
            raise ValueError(
                f"Unknown perturbation_name '{perturb_name}'. "
                f"Available perturbation candidates: {available}"
            )

        candidate = candidates.get(perturb_name, {})
        if isinstance(candidate, str):
            candidate = {'type': candidate}
        elif candidate is None:
            candidate = {}
        elif not isinstance(candidate, dict):
            raise ValueError(
                f"perturbation_candidates['{perturb_name}'] must be a type string or a dict."
            )

        perturb_type = candidate.get('type', config.get('perturbation_type', perturb_name))
        if perturb_type not in {'blur', 'noise', 'sharpen', 'identity'}:
            raise ValueError(f"Unsupported perturbation_type: {perturb_type}")

        return {
            'name': perturb_name,
            'type': perturb_type,
            'kernel_size': candidate.get('kernel_size', config.get('perturbation_kernel_size', 5)),
            'sigma': candidate.get('sigma', config.get('perturbation_sigma', 1.0)),
            'noise_std': candidate.get('noise_std', config.get('perturbation_noise_std', 0.05)),
            'sharpen_amount': candidate.get(
                'sharpen_amount',
                config.get('perturbation_sharpen_amount', 1.0),
            ),
        }

    def build_backbone(self, config):
        clip_path = "/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pretrained/clip14/"
        try:
            clip_model = CLIPModel.from_pretrained(clip_path)
            logger.info(f"Loaded CLIP from local path: {clip_path}")
        except OSError:
            logger.warning(f"Local path {clip_path} not found, trying huggingface hub...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        vision_model = clip_model.vision_model
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        vision_model = get_peft_model(vision_model, peft_config)

        print("Scucessfully applied PEFT LoRA:")
        vision_model.print_trainable_parameters()
        return vision_model

    def features(self, data) -> torch.tensor:
        return self.backbone(data)['pooler_output']

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

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

    def forward(self, data_dict: dict, inference=False) -> dict:
        images = data_dict['image']
        processed_images = self.image_perturbation(images)

        feat_orig = self.features(images)
        feat_perturb = self.features(processed_images.detach())
        feat_diff = feat_orig - feat_perturb
        final_feat = torch.cat([feat_orig, feat_diff], dim=1)

        pred = self.classifier(final_feat)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {
            'cls': pred,
            'prob': prob,
            'feat': final_feat,
            'feat_orig': feat_orig,
            'feat_perturb': feat_perturb,
            'feat_vae': feat_perturb,
            'feat_diff': feat_diff,
        }


class ImagePerturbation(nn.Module):
    def __init__(
        self,
        clip_mean,
        clip_std,
        mode='blur',
        kernel_size=5,
        sigma=1.0,
        noise_std=0.05,
        sharpen_amount=1.0,
    ):
        super(ImagePerturbation, self).__init__()
        self.mode = mode
        self.kernel_size = self._parse_kernel_size(kernel_size)
        self.sigma = float(sigma)
        self.noise_std = float(noise_std)
        self.sharpen_amount = float(sharpen_amount)
        if self.sigma <= 0:
            raise ValueError("perturbation sigma must be positive.")

        self.register_buffer('clip_mean', clip_mean.view(1, -1, 1, 1))
        self.register_buffer('clip_std', clip_std.view(1, -1, 1, 1))

        logger.info(
            f"Using non-VAE image perturbation: mode={self.mode}, "
            f"kernel_size={self.kernel_size}, sigma={self.sigma}, "
            f"noise_std={self.noise_std}, sharpen_amount={self.sharpen_amount}"
        )

    @staticmethod
    def _parse_kernel_size(kernel_size):
        kernel_size = int(kernel_size)
        if kernel_size <= 0:
            raise ValueError("perturbation kernel_size must be positive.")
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def clip_denormalize(self, x):
        return x * self.clip_std + self.clip_mean

    def clip_normalize(self, x):
        return (x - self.clip_mean) / self.clip_std

    def _gaussian_kernel(self, channels, device, dtype):
        coords = torch.arange(self.kernel_size, device=device, dtype=dtype)
        coords = coords - (self.kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.view(1, 1, self.kernel_size, self.kernel_size).repeat(channels, 1, 1, 1)

    def _blur(self, x):
        channels = x.shape[1]
        kernel = self._gaussian_kernel(channels, x.device, x.dtype)
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.conv2d(x, kernel, groups=channels)

    @torch.no_grad()
    def forward(self, images, labels=None):
        x = self.clip_denormalize(images).clamp(0.0, 1.0)

        if self.mode == 'blur':
            processed = self._blur(x)
        elif self.mode == 'noise':
            processed = x + torch.randn_like(x) * self.noise_std
        elif self.mode == 'sharpen':
            blurred = self._blur(x)
            processed = x + self.sharpen_amount * (x - blurred)
        elif self.mode == 'identity':
            processed = x
        else:
            raise ValueError(f"Unsupported perturbation mode: {self.mode}")

        processed = processed.clamp(0.0, 1.0)
        return self.clip_normalize(processed)
