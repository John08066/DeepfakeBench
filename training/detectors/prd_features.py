"""Feature-space components for PRD mechanism-validation experiments."""

import torch
import torch.nn.functional as F


VECTOR_RESIDUALS = {"signed_diff", "abs_diff"}
SCALAR_RESIDUALS = {"cosine", "l2"}


def compute_residual(features_orig: torch.Tensor, features_rec: torch.Tensor, residual_type: str) -> torch.Tensor:
    """Return a PRD residual with shape ``[B, D]`` or scalar discrepancy ``[B, 1]``."""
    if features_orig.shape != features_rec.shape or features_orig.ndim != 2:
        raise ValueError("Residual inputs must be same-shaped feature matrices [B, D].")
    if residual_type == "signed_diff":
        return features_orig - features_rec
    if residual_type == "abs_diff":
        return torch.abs(features_orig - features_rec)
    if residual_type == "cosine":
        return (1.0 - F.cosine_similarity(features_orig, features_rec, dim=1)).unsqueeze(1)
    if residual_type == "l2":
        return torch.linalg.vector_norm(features_orig - features_rec, ord=2, dim=1, keepdim=True)
    raise ValueError(f"Unsupported residual.type: {residual_type}")


def compose_features(features_orig: torch.Tensor, residual: torch.Tensor, feature_mode: str) -> torch.Tensor:
    """Select original, residual, or their concatenation for the classifier."""
    if feature_mode == "orig":
        return features_orig
    if feature_mode == "residual":
        return residual
    if feature_mode == "concat":
        return torch.cat([features_orig, residual], dim=1)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def classifier_input_dim(feature_dim: int, residual_type: str, feature_mode: str) -> int:
    """Compute the classifier input width before constructing the head."""
    residual_dim = feature_dim if residual_type in VECTOR_RESIDUALS else 1
    if residual_type not in VECTOR_RESIDUALS | SCALAR_RESIDUALS:
        raise ValueError(f"Unsupported residual.type: {residual_type}")
    if feature_mode == "orig":
        return feature_dim
    if feature_mode == "residual":
        return residual_dim
    if feature_mode == "concat":
        return feature_dim + residual_dim
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")
