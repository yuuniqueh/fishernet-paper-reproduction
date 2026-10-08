"""Differentiable Fisher Layer following Eqs. (8)--(10) of the paper."""

import math
import torch
from torch import nn


class FisherLayer(nn.Module):
    """Aggregate patch descriptors into a 2*K*D Fisher representation."""

    def __init__(self, descriptor_dim: int = 256, num_components: int = 32):
        super().__init__()
        if descriptor_dim <= 0 or num_components <= 0:
            raise ValueError("descriptor_dim and num_components must be positive")
        self.descriptor_dim = descriptor_dim
        self.num_components = num_components
        self.w = nn.Parameter(torch.ones(num_components, descriptor_dim))
        self.b = nn.Parameter(torch.zeros(num_components, descriptor_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.descriptor_dim:
            raise ValueError(f"expected [B, N, {self.descriptor_dim}], got {tuple(x.shape)}")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Fisher aggregation requires a non-empty batch and patch set")
        z = self.w[None, None, :, :] * (x[:, :, None, :] + self.b[None, None, :, :])
        logits = -0.5 * (z * z).sum(dim=-1)
        gamma = torch.softmax(logits, dim=-1)
        g_mu = gamma[..., None] * z
        g_sigma = gamma[..., None] * (z.square() - 1.0) / math.sqrt(2.0)
        pooled_mu = g_mu.mean(dim=1)
        pooled_sigma = g_sigma.mean(dim=1)
        observer = getattr(self, "diagnostic_observer", None)
        if observer is not None:
            observer(x, z, gamma, pooled_mu, pooled_sigma)
        return torch.cat((pooled_mu, pooled_sigma), dim=1).flatten(start_dim=1)
