"""Export a training checkpoint as a small, release-ready weights folder.

Writes ``model.safetensors`` (the trained weights in fp16, about 230 MB; no
optimiser state and no EMA copy) and ``unet_config.json``. Needs only torch
and safetensors.

    python scripts/export_weights.py step_006000.pt release/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = {name: t.contiguous().half() for name, t in ckpt["model"].items()}
    save_file(weights, out / "model.safetensors", metadata={"format": "pt", "finetune_steps": str(ckpt["step"])})
    (out / "unet_config.json").write_text(json.dumps(ckpt["unet_config"], indent=2, default=list) + "\n")

    reloaded = load_file(out / "model.safetensors")
    assert reloaded.keys() == weights.keys()
    for name, t in weights.items():
        assert torch.equal(reloaded[name], t), name
    print(f"wrote {out}: {len(weights)} tensors, step {ckpt['step']}")


if __name__ == "__main__":
    main()
