"""Hole metrics: scores computed only over the filled pixels.

Known pixels are copied into the output exactly, so including them would
inflate every score. Inputs are in [-1, 1]; errors are measured in [0, 1] units.
"""

from __future__ import annotations

import torch


def _hole_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hole = (1 - mask).expand_as(values)
    return (values * hole).flatten(1).sum(1) / hole.flatten(1).sum(1).clamp(min=1)


def hole_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB over hole pixels; shape ``B``."""
    mse = _hole_mean(((pred - target) / 2) ** 2, mask)
    return -10 * torch.log10(mse.clamp(min=1e-10))


def hole_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image mean absolute error over hole pixels; shape ``B``."""
    return _hole_mean((pred - target).abs() / 2, mask)
