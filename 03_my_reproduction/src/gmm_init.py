"""GMM fitting helpers for Fisher Layer initialization."""

import numpy as np
import torch

from src.fisher_layer import FisherLayer


def initialize_fisher_from_gmm(
    fisher: FisherLayer, means: np.ndarray, diagonal_variances: np.ndarray
) -> None:
    """Set b=-mu and w=1/sigma from a diagonal-covariance GMM."""
    if means.shape != (fisher.num_components, fisher.descriptor_dim):
        raise ValueError(f"unexpected GMM means shape: {means.shape}")
    if diagonal_variances.shape != means.shape:
        raise ValueError(f"unexpected GMM covariance shape: {diagonal_variances.shape}")
    if not np.isfinite(means).all() or not np.isfinite(diagonal_variances).all():
        raise ValueError("GMM means and variances must be finite")
    if np.any(diagonal_variances <= 0):
        raise ValueError("GMM diagonal variances must be positive")
    standard_deviations = np.sqrt(np.maximum(diagonal_variances, 1e-12))
    with torch.no_grad():
        fisher.b.copy_(torch.as_tensor(-means, dtype=fisher.b.dtype, device=fisher.b.device))
        fisher.w.copy_(torch.as_tensor(1.0 / standard_deviations, dtype=fisher.w.dtype, device=fisher.w.device))
