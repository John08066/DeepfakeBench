"""Non-generative probe operators used by PRD mechanism experiments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseProbeOperator(nn.Module):
    """Map a CLIP-normalized image batch ``[B, 3, H, W]`` to a same-shaped reconstruction."""

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed batch in the input's normalization and dtype."""
        raise NotImplementedError

    def forward(self, images: torch.Tensor, labels=None) -> torch.Tensor:
        """Keep the legacy PRD call signature; labels are not probe inputs."""
        del labels
        return self.reconstruct(images)


class IdentityProbeOperator(BaseProbeOperator):
    """Strict negative control with ``T(x) = x``."""

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Return the original tensor without changing its values or shape."""
        return images


class GaussianBlurProbeOperator(BaseProbeOperator):
    """Apply a per-channel Gaussian blur directly on the input GPU tensor."""

    def __init__(self, sigma: float = 1.0):
        """Create a normalized, odd-sized Gaussian kernel for the requested sigma."""
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"Gaussian blur sigma must be positive, got {sigma}.")
        self.sigma = float(sigma)
        radius = max(1, math.ceil(3.0 * self.sigma))
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * self.sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("kernel", kernel_2d.view(1, 1, *kernel_2d.shape))

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Blur ``[B, C, H, W]`` on its existing device while preserving its shape."""
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got shape {tuple(images.shape)}.")
        channels = images.shape[1]
        padding = self.kernel.shape[-1] // 2
        kernel = self.kernel.to(dtype=images.dtype).expand(channels, 1, -1, -1)
        padded_images = F.pad(images, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(padded_images, kernel, groups=channels)
