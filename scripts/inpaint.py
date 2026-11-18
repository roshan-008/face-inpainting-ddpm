"""Inpaint a local photo with the released weights (no Modal needed).

    python scripts/inpaint.py --image face.jpg --mask mask.png --out result.png

Mask: white = keep, black = fill (``--invert-mask`` for the opposite). Output
is one strip: masked input | independent fills | original.

Runs all 1,000 denoising steps. On a CUDA GPU this takes about a minute per
batch; on CPU expect tens of minutes. ``--weights`` accepts a Hugging Face
repo id or a local folder from ``scripts/export_weights.py``.
"""

from __future__ import annotations

import argparse

import torch
from diffusers import DDPMScheduler
from torchvision.utils import save_image

from face_inpainting.data import load_photo_and_mask
from face_inpainting.model import HUB_REPO, load_pretrained
from face_inpainting.sampling import inpaint

BASE_MODEL = "google/ddpm-ema-celebahq-256"  # only its noise-schedule config is used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="result.png")
    parser.add_argument("--samples", type=int, default=4, help="independent fills to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default=HUB_REPO)
    parser.add_argument("--invert-mask", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    unet = load_pretrained(args.weights, args.device, dtype)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL)
    x, m = load_photo_and_mask(args.image, args.mask, invert_mask=args.invert_mask)
    generator = torch.Generator(args.device).manual_seed(args.seed)
    y = inpaint(unet, scheduler, x.repeat(args.samples, 1, 1, 1).to(args.device),
                m.repeat(args.samples, 1, 1, 1).to(args.device), generator, dtype=dtype)
    strip = torch.cat([x * m + (1 - m), y.cpu(), x]) * 0.5 + 0.5
    save_image(strip.clamp(0, 1), args.out, nrow=args.samples + 2, padding=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
