"""Create an inpainting mask for a local photo.

The output has the photo's size and is white where pixels are kept and black
where the model should fill in new content. Needs only Pillow.

Examples:
    python scripts/make_mask.py --image photo.jpg --out mask.png --kind box --size 0.35
    python scripts/make_mask.py --image photo.jpg --out mask.png --kind half
    python scripts/make_mask.py --image photo.jpg --out mask.png --kind random --hole-fraction 0.25 --seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", choices=["box", "half", "random"], default="box")
    parser.add_argument("--size", type=float, default=0.35,
                        help="box side as a fraction of the shorter image side")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed used for the reproducible irregular mask")
    parser.add_argument("--hole-fraction", type=float, default=0.25,
                        help="target fraction of pixels to fill for random masks")
    args = parser.parse_args()

    if not 0.05 <= args.size <= 0.95:
        raise SystemExit("--size must be between 0.05 and 0.95")
    if not 0.10 <= args.hole_fraction <= 0.60:
        raise SystemExit("--hole-fraction must be between 0.10 and 0.60")

    with Image.open(args.image) as source:
        width, height = source.size
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    if args.kind == "box":
        side = int(min(width, height) * args.size)
        left = (width - side) // 2
        top = (height - side) // 2
        draw.rectangle((left, top, left + side - 1, top + side - 1), fill=0)
    else:
        if args.kind == "half":
            draw.rectangle((width // 2, 0, width - 1, height - 1), fill=0)
        else:
            rng = random.Random(args.seed)
            # Several small overlapping ellipses/rectangles approximate
            # free-form brush damage. Build one candidate at a time and stop
            # near the requested hole fraction so a "random" mask does not
            # accidentally remove almost the entire image.
            short_side = min(width, height)
            hole_fraction = 0.0
            for _ in range(40):
                layer = Image.new("L", (width, height), 255)
                cx = rng.randint(width // 5, max(width // 5, 4 * width // 5))
                cy = rng.randint(height // 5, max(height // 5, 4 * height // 5))
                rw = rng.randint(max(8, short_side // 20), max(12, short_side // 8))
                rh = rng.randint(max(8, short_side // 20), max(12, short_side // 8))
                layer_draw = ImageDraw.Draw(layer)
                if rng.random() < 0.65:
                    layer_draw.ellipse((cx - rw, cy - rh, cx + rw, cy + rh), fill=0)
                else:
                    layer_draw.rounded_rectangle(
                        (cx - rw, cy - rh, cx + rw, cy + rh),
                        radius=max(2, min(rw, rh) // 3),
                        fill=0,
                    )
                candidate = ImageChops.darker(mask, layer)
                candidate_hole_fraction = sum(
                    pixel < 128 for pixel in candidate.getdata()
                ) / (width * height)
                if candidate_hole_fraction <= min(0.60, args.hole_fraction * 1.25):
                    mask = candidate
                    hole_fraction = candidate_hole_fraction
                if hole_fraction >= args.hole_fraction * 0.85:
                    break

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    print(f"wrote {output} ({width}x{height}; black=fill, white=preserve; kind={args.kind})")


if __name__ == "__main__":
    main()
