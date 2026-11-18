"""Image loading: a folder dataset for training and helpers for single photos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def to_tensor(picture: Image.Image) -> torch.Tensor:
    """PIL image -> ``3 x H x W`` float tensor in [-1, 1]."""
    pixels = np.asarray(picture.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1)


def fit_square(picture: Image.Image, size: int = 256, resample=Image.Resampling.LANCZOS) -> Image.Image:
    """Centre-crop to a square and resize to ``size x size``."""
    return ImageOps.fit(picture, (size, size), method=resample, centering=(0.5, 0.5))


class FaceFolder(Dataset):
    """Every image below ``root``, centre-cropped to 256x256, in [-1, 1]."""

    def __init__(self, root: str | Path, size: int = 256) -> None:
        root = Path(root)
        self.size = size
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(self.paths) < 32:
            raise ValueError(f"need at least 32 images below {root}; found {len(self.paths)}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as source:
            return to_tensor(fit_square(source.convert("RGB"), self.size))


def load_photo_and_mask(
    image_path: str | Path, mask_path: str | Path, size: int = 256, invert_mask: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one photo and its mask as ``1 x 3 x size x size`` and ``1 x 1 x size x size``.

    Both are centre-cropped and resized the same way. The mask is white = keep,
    black = fill (``invert_mask`` for the opposite), thresholded at 128.
    """
    picture = fit_square(Image.open(image_path).convert("RGB"), size)
    mask_image = fit_square(Image.open(mask_path).convert("L"), size, Image.Resampling.NEAREST)
    known = np.asarray(mask_image) >= 128
    if invert_mask:
        known = ~known
    if known.all() or not known.any():
        raise ValueError("mask must contain both white (keep) and black (fill) pixels")
    mask = torch.from_numpy(known.astype(np.float32))[None, None]
    return to_tensor(picture)[None], mask
