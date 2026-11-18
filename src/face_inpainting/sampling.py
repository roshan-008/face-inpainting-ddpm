"""Inpainting sampler: one DDPM reverse pass with per-step known-pixel replacement."""

from __future__ import annotations

import torch
from diffusers import DDPMScheduler, UNet2DModel


@torch.no_grad()
def inpaint(
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    image: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Fill the holes of ``image`` (B x 3 x H x W, [-1, 1]) where ``mask`` is 0.

    Runs every noise level of ``scheduler`` once, from pure noise to a clean
    image. At each step the UNet sees ``[x_t, image * mask, mask]`` and
    predicts the noise. After the step, the known region is overwritten with
    ``image`` noised to the next level, so the fill has to agree with its
    border throughout sampling. Known pixels in the output are exact.
    """
    device = image.device
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    timesteps = scheduler.timesteps
    cond = torch.cat((image * mask, mask), dim=1)
    y = torch.randn(image.shape, generator=generator, device=device)
    for i, t in enumerate(timesteps):
        eps = unet(torch.cat((y, cond), dim=1).to(dtype), t).sample.float()
        y = scheduler.step(eps, t, y, generator=generator).prev_sample
        if i + 1 < len(timesteps):
            noise = torch.randn(image.shape, generator=generator, device=device)
            known = scheduler.add_noise(image, noise, timesteps[i + 1 : i + 2])
        else:
            known = image
        y = mask * known + (1 - mask) * y
    y = y.float().clamp(-1, 1)
    return mask * image + (1 - mask) * y
