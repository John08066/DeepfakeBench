"""Feature-encoder adapter interfaces for PRD experiments."""

import torch


class FeatureEncoder:
    """Expose an encoder as ``forward_features(x) -> [B, D]``."""

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return a two-dimensional per-image feature matrix."""
        raise NotImplementedError


class CLIPFeatureEncoder(FeatureEncoder):
    """Adapter around the existing LoRA-wrapped CLIP vision backbone."""

    def __init__(self, backbone):
        """Keep a non-owning reference so checkpoint keys remain ``backbone.*``."""
        self.backbone = backbone

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return CLIP pooler output with shape ``[B, D]``."""
        return self.backbone(images)["pooler_output"]


def build_feature_encoder(encoder_config, clip_backbone):
    """Return a configured encoder without downloading unavailable weights."""
    encoder_type = (encoder_config or {}).get("type", "clip")
    if encoder_type == "clip":
        return CLIPFeatureEncoder(clip_backbone)
    if encoder_type in {"dinov2", "xception"}:
        raise RuntimeError(
            f"encoder.type={encoder_type} requires a local compatible pretrained checkpoint; none is configured."
        )
    raise ValueError(f"Unsupported encoder.type: {encoder_type}")
