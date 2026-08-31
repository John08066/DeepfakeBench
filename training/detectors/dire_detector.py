import logging
import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from detectors import DETECTOR

logger = logging.getLogger(__name__)


def _valid_path(path):
    return isinstance(path, str) and len(path) > 0 and path.lower() not in {
        "none",
        "null",
        "no",
        "no need",
        "no need to provide this for the effort model",
    }


@DETECTOR.register_module(module_name="dire")
class DIREDetector(nn.Module):
    """DIRE detector adapted to the DeepfakeBench detector interface.

    DIRE uses a diffusion reconstruction error image as detector input. The
    diffusion reconstruction is loaded lazily from the official guided-diffusion
    code path configured in ``dire.yaml``.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        self.loss_func = nn.CrossEntropyLoss()
        self.backbone = self.build_backbone(self.config)

        mean = torch.tensor(self.config.get("mean", [0.5, 0.5, 0.5]), dtype=torch.float32)
        std = torch.tensor(self.config.get("std", [0.5, 0.5, 0.5]), dtype=torch.float32)
        self.register_buffer("data_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("data_std", std.view(1, 3, 1, 1))

        cls_mean = torch.tensor(
            self.config.get("dire_classifier_mean", [0.485, 0.456, 0.406]),
            dtype=torch.float32,
        )
        cls_std = torch.tensor(
            self.config.get("dire_classifier_std", [0.229, 0.224, 0.225]),
            dtype=torch.float32,
        )
        self.register_buffer("classifier_mean", cls_mean.view(1, 3, 1, 1))
        self.register_buffer("classifier_std", cls_std.view(1, 3, 1, 1))

        self.diffusion = None
        self.diffusion_model = None
        self._diffusion_loaded = False

        self._load_pretrained_classifier()
        if not self.config.get("dire_lazy_load", True):
            self._build_diffusion_reconstructor()

    def build_backbone(self, config):
        try:
            import torchvision.models as models
        except Exception as exc:
            raise ImportError("DIREDetector requires torchvision to build the ResNet classifier.") from exc

        arch = config.get("dire_classifier_arch", "resnet50")
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported DIRE classifier architecture: {arch}")

        use_imagenet = bool(config.get("dire_imagenet_pretrained", False))
        model_fn = getattr(models, arch)
        try:
            if use_imagenet:
                weights_enum = getattr(models, f"{arch.upper()}_Weights")
                backbone = model_fn(weights=weights_enum.DEFAULT)
            else:
                backbone = model_fn(weights=None)
        except Exception:
            backbone = model_fn(pretrained=use_imagenet)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        return backbone

    def build_loss(self, config):
        return nn.CrossEntropyLoss()

    def _load_pretrained_classifier(self):
        ckpt_path = self.config.get("dire_classifier_pretrained", self.config.get("pretrained"))
        if not _valid_path(ckpt_path):
            return
        if not os.path.exists(ckpt_path):
            logger.warning("DIRE classifier checkpoint not found: %s", ckpt_path)
            return

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        clean_state = {}
        own_state = self.backbone.state_dict()
        for key, value in state.items():
            key = key.replace("module.", "")
            key = key.replace("backbone.", "")
            if key in own_state and own_state[key].shape == value.shape:
                clean_state[key] = value

        missing, unexpected = self.backbone.load_state_dict(clean_state, strict=False)
        logger.info(
            "Loaded DIRE classifier from %s with %d tensors, missing=%d, unexpected=%d",
            ckpt_path,
            len(clean_state),
            len(missing),
            len(unexpected),
        )

    def _diffusion_config(self) -> Dict:
        return self.config.get("dire_diffusion", {})

    def _build_diffusion_reconstructor(self):
        if self._diffusion_loaded:
            return

        diff_cfg = self._diffusion_config()
        guided_diffusion_path = diff_cfg.get(
            "guided_diffusion_path",
            self.config.get("guided_diffusion_path", ""),
        )
        if _valid_path(guided_diffusion_path):
            guided_diffusion_path = os.path.abspath(guided_diffusion_path)
            if guided_diffusion_path not in sys.path:
                sys.path.insert(0, guided_diffusion_path)

        try:
            from guided_diffusion.script_util import (
                create_model_and_diffusion,
                model_and_diffusion_defaults,
            )
        except Exception as exc:
            raise ImportError(
                "DIRE online reconstruction requires the official guided-diffusion "
                "package. Set dire_diffusion.guided_diffusion_path in dire.yaml "
                "to the DIRE/guided-diffusion directory."
            ) from exc

        model_path = diff_cfg.get("model_path", self.config.get("diffusion_model_path", ""))
        if not _valid_path(model_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "DIRE diffusion checkpoint is required for online reconstruction. "
                "Set dire_diffusion.model_path in dire.yaml."
            )

        defaults = model_and_diffusion_defaults()
        defaults.update(diff_cfg.get("model_kwargs", {}))
        model, diffusion = create_model_and_diffusion(**defaults)

        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        if diff_cfg.get("use_fp16", defaults.get("use_fp16", False)) and hasattr(model, "convert_to_fp16"):
            model.convert_to_fp16()
        model.requires_grad_(False)
        model.eval()

        self.diffusion_model = model
        self.diffusion = diffusion
        self.diffusion_image_size = int(defaults.get("image_size", 256))
        self.diffusion_class_cond = bool(defaults.get("class_cond", False))
        self._diffusion_loaded = True

    def _denormalize_data(self, images):
        return (images * self.data_std + self.data_mean).clamp(0.0, 1.0)

    def _normalize_classifier_input(self, images):
        return (images - self.classifier_mean) / self.classifier_std

    @staticmethod
    def _resize_for_diffusion(images, image_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        original_size = (images.shape[-2], images.shape[-1])
        if images.shape[-2] != images.shape[-1]:
            side = min(images.shape[-2], images.shape[-1])
            top = (images.shape[-2] - side) // 2
            left = (images.shape[-1] - side) // 2
            images = images[:, :, top : top + side, left : left + side]
        if images.shape[-1] != image_size:
            images = F.interpolate(images, size=(image_size, image_size), mode="bicubic", align_corners=False)
        return images, original_size

    @torch.no_grad()
    def _reconstruct_with_guided_diffusion(self, images_0_1):
        self._build_diffusion_reconstructor()

        model = self.diffusion_model
        diffusion = self.diffusion
        model.to(images_0_1.device)
        model.eval()

        images_m1_1 = images_0_1 * 2.0 - 1.0
        diffusion_input, original_size = self._resize_for_diffusion(images_m1_1, self.diffusion_image_size)

        batch_size = diffusion_input.shape[0]
        diff_cfg = self._diffusion_config()
        model_kwargs = {}
        if self.diffusion_class_cond:
            model_kwargs["y"] = torch.zeros(batch_size, dtype=torch.long, device=diffusion_input.device)

        reverse_fn = diffusion.ddim_reverse_sample_loop
        latent = reverse_fn(
            model,
            (batch_size, 3, self.diffusion_image_size, self.diffusion_image_size),
            noise=diffusion_input,
            clip_denoised=diff_cfg.get("clip_denoised", True),
            model_kwargs=model_kwargs,
            real_step=diff_cfg.get("real_step", 0),
        )

        sample_fn = diffusion.ddim_sample_loop if diff_cfg.get("use_ddim", True) else diffusion.p_sample_loop
        recon_m1_1 = sample_fn(
            model,
            (batch_size, 3, self.diffusion_image_size, self.diffusion_image_size),
            noise=latent,
            clip_denoised=diff_cfg.get("clip_denoised", True),
            model_kwargs=model_kwargs,
            real_step=diff_cfg.get("real_step", 0),
        )

        if recon_m1_1.shape[-2:] != original_size:
            recon_m1_1 = F.interpolate(recon_m1_1, size=original_size, mode="bicubic", align_corners=False)
        return ((recon_m1_1 + 1.0) / 2.0).clamp(0.0, 1.0)

    def _compute_dire_map(self, images):
        images_0_1 = self._denormalize_data(images)
        recon_0_1 = self._reconstruct_with_guided_diffusion(images_0_1)
        return torch.abs(images_0_1 - recon_0_1).clamp(0.0, 1.0)

    def features(self, data_dict: dict) -> torch.Tensor:
        return self._compute_dire_map(data_dict["image"])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        score = self.backbone(self._normalize_classifier_input(features)).view(-1, 1)
        zeros = torch.zeros_like(score)
        return torch.cat([zeros, score], dim=1)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        loss = self.loss_func(pred, label)
        return {"overall": loss, "cls": loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        dire_map = self.features(data_dict)
        pred = self.classifier(dire_map)
        score = pred[:, 1]
        prob = torch.sigmoid(score)
        feat_diff = dire_map.mean(dim=(2, 3))
        return {
            "cls": pred,
            "prob": prob,
            "feat": score.unsqueeze(1),
            "feat_diff": feat_diff,
            "dire": dire_map,
        }

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state.items() if not k.startswith("diffusion_model.")}

    def load_state_dict(self, state_dict, strict=True):
        excluded_prefixes = ("diffusion_model.",)
        filtered = {k: v for k, v in state_dict.items() if not k.startswith(excluded_prefixes)}
        result = super().load_state_dict(filtered, strict=False)
        if strict:
            missing = [k for k in result.missing_keys if not k.startswith(excluded_prefixes)]
            unexpected = [k for k in result.unexpected_keys if not k.startswith(excluded_prefixes)]
            if missing or unexpected:
                raise RuntimeError(
                    "Error(s) in loading state_dict for DIREDetector:\n"
                    f"\tMissing key(s): {missing}\n"
                    f"\tUnexpected key(s): {unexpected}"
                )
        return result
