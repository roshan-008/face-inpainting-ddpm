"""Building and loading the mask-aware UNet.

The pretrained ``UNet2DModel`` takes 3 input channels (the noisy image). The
inpainting model takes 7::

    [ x_t (3) | clean_image * mask (3) | mask (1) ]

The pretrained RGB filters are kept and the 4 new channels start at zero, so
before fine-tuning the widened model computes exactly what the pretrained
model computes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import UNet2DModel

IN_CHANNELS = 7


def widen_input(base: UNet2DModel) -> tuple[UNet2DModel, dict]:
    """Return a 7-channel copy of ``base`` and its config.

    All weights except ``conv_in`` are loaded strictly; ``conv_in`` keeps the
    pretrained filters for the first 3 channels and zeros for the rest.
    """
    config = dict(base.config)
    config["in_channels"] = IN_CHANNELS
    model = UNet2DModel(**config)
    state = base.state_dict()
    conv_weight = state.pop("conv_in.weight")
    conv_bias = state.pop("conv_in.bias")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"conv_in.weight", "conv_in.bias"} or unexpected:
        raise RuntimeError(f"unexpected conversion mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        model.conv_in.weight.zero_()
        model.conv_in.weight[:, : conv_weight.shape[1]].copy_(conv_weight)
        model.conv_in.bias.copy_(conv_bias)
    return model, config


def load_checkpoint(
    path: str | Path,
    weights: str = "model",
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[UNet2DModel, int]:
    """Load a fine-tuned checkpoint written by ``modal_app.finetune``.

    ``weights`` is ``"model"`` (the trained weights, used for all results) or
    ``"ema"``. Returns the UNet in eval mode and the training step.
    """
    if weights not in ("model", "ema"):
        raise ValueError("weights must be 'model' or 'ema'")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unet = UNet2DModel(**ckpt["unet_config"])
    unet.load_state_dict(ckpt[weights])
    return unet.to(device=device, dtype=dtype).eval(), int(ckpt["step"])
