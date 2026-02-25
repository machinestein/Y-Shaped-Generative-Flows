"""
Utility functions for computing Wasserstein distances and MMD distances.
"""

import torch
import numpy as np
import math
from typing import Tuple, Optional, Dict
from functools import partial
import ot as pot


def wasserstein_distance(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 1,
    **kwargs,
) -> float:
    """
    Compute Wasserstein distance between two distributions.
    """
    assert power == 1 or power == 2
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2

    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))

    if power == 2:
        ret = math.sqrt(ret)

    return ret


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel between two sets of points.
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))

    return torch.exp(-dist / (2 * sigma**2))


def mmd_distance(
    x0: torch.Tensor,
    x1: torch.Tensor,
    sigma: float = 1.0,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) with RBF kernel between two distributions.
    """

    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    K_xx = rbf_kernel(x0, x0, sigma)
    K_yy = rbf_kernel(x1, x1, sigma)
    K_xy = rbf_kernel(x0, x1, sigma)

    mmd_squared = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    mmd = torch.sqrt(torch.clamp(mmd_squared, min=0))

    return mmd.item()
