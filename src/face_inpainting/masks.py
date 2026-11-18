"""Hole masks for training and evaluation (1 = known pixel, 0 = hole)."""

from __future__ import annotations

import random

import numpy as np
import torch

IMAGE_SIZE = 256
EVAL_MASK_KINDS = ("box", "irregular", "half")


def random_training_masks(batch: int, device: torch.device | str, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Return ``batch x 1 x size x size`` training masks.

    45 % of masks are a single rectangle; the rest are 2-5 overlapping
    rectangles. Every rectangle is 24-110 px per side and placed anywhere in
    the image. Draws from Python's ``random`` module, so seed that for
    reproducible masks.
    """
    mask = torch.ones((batch, 1, size, size), device=device)
    for item in range(batch):
        pieces = 1 if random.random() < 0.45 else random.randint(2, 5)
        for _ in range(pieces):
            width = random.randint(24, 110)
            height = random.randint(24, 110)
            left = random.randint(0, size - width)
            top = random.randint(0, size - height)
            mask[item, :, top : top + height, left : left + width] = 0
    return mask


def eval_mask(kind: str, index: int, size: int = IMAGE_SIZE) -> np.ndarray:
    """Deterministic ``size x size`` uint8 mask for evaluation image ``index``.

    - ``box``: centred 96x96 hole (14.1 % of a 256x256 image).
    - ``half``: left half missing.
    - ``irregular``: 2-5 overlapping rectangles, 24-110 px per side, seeded
      by ``1000 + index`` so every image gets its own fixed mask.
    """
    mask = np.ones((size, size), dtype=np.uint8)
    if kind == "box":
        lo, hi = size * 80 // 256, size * 176 // 256
        mask[lo:hi, lo:hi] = 0
    elif kind == "half":
        mask[:, : size // 2] = 0
    elif kind == "irregular":
        rng = np.random.default_rng(1000 + index)
        for _ in range(rng.integers(2, 6)):
            w, h = rng.integers(24, 111, size=2)
            left, top = rng.integers(0, size - w + 1), rng.integers(0, size - h + 1)
            mask[top : top + h, left : left + w] = 0
    else:
        raise ValueError(f"unknown mask kind {kind!r}; expected one of {EVAL_MASK_KINDS}")
    return mask
