"""Modal app: clone the pretrained DDPM, prepare CelebA-HQ, fine-tune, evaluate, inpaint.

Everything runs on Modal GPUs and stores its files in one Modal Volume.
Run from the repository root::

    modal run modal_app.py::clone_pretrained                     # copy the pretrained model
    modal run modal_app.py::prepare_data                         # CelebA-HQ 28k train / 2k val
    modal run --detach modal_app.py::train --spawn               # fine-tune (resumes automatically)
    modal run modal_app.py::evaluate                             # score held-out faces
    modal run modal_app.py::inpaint_image --image /input/face.jpg --mask /input/mask.png

Mask convention everywhere: 1/white = known pixel, 0/black = hole to fill.
"""

from __future__ import annotations

import copy
import os
import random
import signal
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
VOLUME_NAME = os.environ.get("IMAGE_INPAINTING_VOLUME", "image-inpainting-training")
VOLUME_MOUNT = "/mnt/image-inpainting"
MODEL_ID = "google/ddpm-ema-celebahq-256"
# The cloned copy of the pretrained repository inside the Volume. Every
# function loads from here, so all runs use byte-identical weights.
MODEL_DIR = Path(VOLUME_MOUNT) / "models" / "ddpm-ema-celebahq-256"
HF_DATASET = "korexyz/celeba-hq-256x256"
TRAIN_DIR = Path(VOLUME_MOUNT) / "data" / "portraits"
VAL_DIR = Path(VOLUME_MOUNT) / "data" / "portraits_val"
RUNS_DIR = Path(VOLUME_MOUNT) / "runs"
DEFAULT_RUN = "pretrained_conditional"
EVAL_DIR = Path(VOLUME_MOUNT) / "results" / "eval"
TRAIN_GPU = os.environ.get("TRAIN_GPU", "L4")
EVAL_GPU = os.environ.get("EVAL_GPU", "L40S")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.1",
        "diffusers==0.41.0",
        "accelerate==1.15.0",   # lets diffusers load weights without a random init first
        "huggingface_hub==1.33.0",
        "pyarrow==26.0.0",
        "pillow==12.3.0",
        "numpy==2.4.6",
    )
    .add_local_dir(PROJECT_ROOT / "src" / "face_inpainting", remote_path="/root/face_inpainting")
)
app = modal.App("face-inpainting", image=image)
ENV = {"HF_HOME": f"{VOLUME_MOUNT}/hf-cache", "PYTHONUNBUFFERED": "1"}


@app.function(volumes={VOLUME_MOUNT: volume}, env=ENV, timeout=30 * 60)
def clone_pretrained() -> str:
    """Copy the pretrained DDPM repository (weights + configs) into the Volume."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR))
    volume.commit()
    files = sorted(p.relative_to(MODEL_DIR).as_posix() for p in MODEL_DIR.rglob("*") if p.is_file())
    print(f"[clone] {MODEL_ID} -> {path}: {files}")
    return path


@app.function(volumes={VOLUME_MOUNT: volume}, env=ENV, cpu=4, memory=8192, timeout=2 * 60 * 60)
def prepare_data(splits: str = "train,val") -> None:
    """Write CelebA-HQ 256x256 faces: 28,000 for training, 2,000 held out for evaluation."""
    import io

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from PIL import Image

    local = snapshot_download(
        HF_DATASET, repo_type="dataset", local_dir="/tmp/celebahq", allow_patterns=["data/*.parquet"]
    )
    for split in splits.split(","):
        prefix, root, expected = {
            "train": ("train-", TRAIN_DIR, 28000),
            "val": ("validation-", VAL_DIR, 2000),
        }[split]
        root.mkdir(parents=True, exist_ok=True)
        count = 0
        for f in sorted(Path(local, "data").glob(f"{prefix}*.parquet")):
            for cell in pq.read_table(f, columns=["image"]).column("image").to_pylist():
                data = cell["bytes"]
                fmt = Image.open(io.BytesIO(data)).format.lower()
                ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}[fmt]
                (root / f"{count:05d}{ext}").write_bytes(data)
                count += 1
        if count != expected:
            raise RuntimeError(f"{split}: expected {expected} images, wrote {count}")
        volume.commit()
        print(f"[prepare] {split}: {count} images in {root}")


@app.function(
    gpu=TRAIN_GPU,
    timeout=6 * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=1),
    volumes={VOLUME_MOUNT: volume},
    env=ENV,
)
def finetune(
    total_steps: int = 6_000,
    batch_size: int = 4,
    gradient_accumulation: int = 8,
    learning_rate: float = 1e-5,
    checkpoint_every: int = 500,
    seed: int = 0,
    run_name: str = DEFAULT_RUN,
) -> str:
    """Fine-tune the mask-aware UNet up to ``total_steps``, resuming ``latest.pt`` if present."""
    import json
    import math
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, UNet2DModel
    from torch.utils.data import DataLoader

    from face_inpainting.data import FaceFolder
    from face_inpainting.masks import random_training_masks
    from face_inpainting.model import widen_input

    if total_steps < 1 or total_steps > 20_000:
        raise ValueError("total_steps must be in 1..20000")
    if batch_size < 1 or gradient_accumulation < 1 or checkpoint_every < 1:
        raise ValueError("batch_size, gradient_accumulation, and checkpoint_every must be positive")
    if not 1e-6 <= learning_rate <= 5e-5:
        raise ValueError("learning_rate must be between 1e-6 and 5e-5")
    run_dir = RUNS_DIR / run_name

    def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}

    def atomic_save(payload: dict[str, object], step: int) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        numbered = run_dir / f"step_{step:06d}.pt"
        for target in (numbered, run_dir / "latest.pt"):
            temporary = run_dir / f".{target.name}.tmp"
            torch.save(payload, temporary)
            os.replace(temporary, target)
        volume.commit()
        return numbered

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    loader = DataLoader(FaceFolder(TRAIN_DIR), batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    iterator = iter(loader)

    source = str(MODEL_DIR) if MODEL_DIR.exists() else MODEL_ID
    model, config = widen_input(UNet2DModel.from_pretrained(source))
    model.to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    scheduler = DDPMScheduler.from_pretrained(source)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_step = 0
    latest = run_dir / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        del state
        # Re-seed from the resume step so a continued run draws new batches,
        # masks and noise instead of replaying those of steps 1..start_step.
        random.seed(seed + start_step)
        np.random.seed(seed + start_step)
        torch.manual_seed(seed + start_step)
        torch.cuda.manual_seed_all(seed + start_step)
        iterator = iter(loader)
        print(f"[train] resuming {run_name} at step {start_step}")
    else:
        print(f"[train] starting {run_name} from {source}")

    stop_requested = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[train] received signal {signum}; checkpointing after the current optimizer step")

    # Modal signals cancellation with SIGINT, preemption with SIGTERM.
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    last_loss = math.nan
    saved_step = start_step

    def checkpoint(reason: str, step: int) -> Path:
        nonlocal saved_step
        payload: dict[str, object] = {
            "format_version": 1,
            "model_id": MODEL_ID,
            "unet_config": config,
            "step": step,
            "reason": reason,
            "model": cpu_state(model),
            "ema": cpu_state(ema_model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": {
                "batch_size": batch_size,
                "gradient_accumulation": gradient_accumulation,
                "learning_rate": learning_rate,
                "seed": seed,
            },
            "last_loss": last_loss,
            "saved_at_unix": time.time(),
        }
        saved = atomic_save(payload, step)
        saved_step = step
        (run_dir / "latest.json").write_text(json.dumps({"step": step, "reason": reason, "last_loss": last_loss}) + "\n")
        volume.commit()
        print(f"[train] checkpointed {saved} ({reason})")
        return saved

    step = start_step
    try:
        for step in range(start_step + 1, total_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(gradient_accumulation):
                try:
                    clean = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    clean = next(iterator)
                clean = clean.to(device, non_blocking=True)
                known = random_training_masks(clean.shape[0], device)
                noise = torch.randn_like(clean)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=device).long()
                noisy = scheduler.add_noise(clean, noise, timesteps)
                condition = torch.cat((noisy, clean * known, known), dim=1)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(condition, timesteps).sample
                    loss = F.mse_loss(prediction, noise) / gradient_accumulation
                scaler.scale(loss).backward()
                accumulated_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                    ema_parameter.mul_(0.9999).add_(parameter, alpha=0.0001)
            last_loss = accumulated_loss
            if step % 25 == 0:
                print(f"[train] step={step} loss={last_loss:.5f}")
            if step % checkpoint_every == 0 or stop_requested:
                checkpoint("signal" if stop_requested else "interval", step)
            if stop_requested:
                break
    except BaseException:
        if step > saved_step:
            checkpoint("exception", step)
        raise
    else:
        if step > saved_step:
            checkpoint("complete", step)

    return str(run_dir / "latest.pt")


@app.function(gpu=EVAL_GPU, volumes={VOLUME_MOUNT: volume}, env=ENV, timeout=2 * 60 * 60)
def evaluate(
    checkpoint: str = "latest.pt",
    run_name: str = DEFAULT_RUN,
    n_faces: int = 8,
    seed: int = 0,
    weights: str = "model",
) -> dict:
    """Inpaint the first ``n_faces`` held-out faces with each evaluation mask.

    Writes one grid per mask (rows: input with white holes / result / original)
    and ``metrics.json`` to ``/results/eval/<checkpoint>``. Scores are measured
    only over the filled pixels.
    """
    import json

    import torch
    from diffusers import DDPMScheduler
    from PIL import Image
    from torchvision.utils import save_image

    from face_inpainting.data import to_tensor
    from face_inpainting.masks import EVAL_MASK_KINDS, eval_mask
    from face_inpainting.metrics import hole_l1, hole_psnr
    from face_inpainting.model import load_checkpoint
    from face_inpainting.sampling import inpaint

    volume.reload()
    device = "cuda"
    unet, step = load_checkpoint(RUNS_DIR / run_name / checkpoint, weights, device)
    scheduler = DDPMScheduler.from_pretrained(str(MODEL_DIR))
    name = Path(checkpoint).stem if checkpoint != "latest.pt" else f"step_{step:06d}"
    out_dir = EVAL_DIR / (name if weights == "model" else f"{name}_{weights}")
    out_dir.mkdir(parents=True, exist_ok=True)

    faces = sorted(VAL_DIR.iterdir())[:n_faces]
    if len(faces) < n_faces:
        raise FileNotFoundError(f"need {n_faces} validation faces in {VAL_DIR}; run prepare_data")
    x = torch.stack([to_tensor(Image.open(p)) for p in faces]).to(device)

    metrics = {"checkpoint": checkpoint, "finetune_steps": step, "weights": weights,
               "seed": seed, "n_faces": n_faces, "sets": {}}
    for kind in EVAL_MASK_KINDS:
        m = torch.stack([torch.from_numpy(eval_mask(kind, i)).float()[None] for i in range(n_faces)]).to(device)
        g = torch.Generator(device=device).manual_seed(seed)
        y = inpaint(unet, scheduler, x, m, g)
        metrics["sets"][f"val_{kind}"] = {
            "images": n_faces,
            "hole_fraction": float((1 - m).mean()),
            "hole_psnr_db": float(hole_psnr(y, x, m).mean()),
            "hole_l1": float(hole_l1(y, x, m).mean()),
        }
        observed = x * m + (1 - m)                            # holes shown in white
        grid = torch.cat([observed, y, x]) * 0.5 + 0.5        # rows: input / result / original
        save_image(grid.clamp(0, 1), out_dir / f"val_{kind}.png", nrow=n_faces, padding=2)
        print(f"[eval] {name} val_{kind}: {metrics['sets'][f'val_{kind}']}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    volume.commit()
    return metrics


@app.function(gpu=EVAL_GPU, volumes={VOLUME_MOUNT: volume}, env=ENV, timeout=60 * 60)
def inpaint_image(
    image: str,
    mask: str,
    out: str = "",
    samples: int = 4,
    seed: int = 0,
    checkpoint: str = "latest.pt",
    run_name: str = DEFAULT_RUN,
    invert_mask: bool = False,
) -> str:
    """Inpaint one photo uploaded to the Volume, e.g. ``--image /input/face.jpg``.

    Mask: white = keep, black = fill (``--invert-mask`` for the opposite).
    Both are centre-cropped to a square and resized to 256x256. Writes one
    strip, input | ``samples`` independent fills | original, to
    ``/results/inpaint_<name>.png`` unless ``--out`` is given.
    """
    import torch
    from diffusers import DDPMScheduler
    from torchvision.utils import save_image

    from face_inpainting.data import load_photo_and_mask
    from face_inpainting.model import load_checkpoint
    from face_inpainting.sampling import inpaint

    volume.reload()
    device = "cuda"
    src = Path(VOLUME_MOUNT) / image.lstrip("/")
    x, m = load_photo_and_mask(src, Path(VOLUME_MOUNT) / mask.lstrip("/"), invert_mask=invert_mask)
    unet, _ = load_checkpoint(RUNS_DIR / run_name / checkpoint, "model", device)
    scheduler = DDPMScheduler.from_pretrained(str(MODEL_DIR))
    g = torch.Generator(device=device).manual_seed(seed)
    y = inpaint(unet, scheduler, x.repeat(samples, 1, 1, 1).to(device), m.repeat(samples, 1, 1, 1).to(device), g)
    strip = torch.cat([x * m + (1 - m), y.cpu(), x]) * 0.5 + 0.5
    dest = Path(VOLUME_MOUNT) / (out.lstrip("/") or f"results/inpaint_{src.stem}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_image(strip.clamp(0, 1), dest, nrow=samples + 2, padding=2)
    volume.commit()
    print(f"[inpaint] wrote {dest}")
    return str(dest)


@app.local_entrypoint()
def train(
    total_steps: int = 6_000,
    batch_size: int = 4,
    gradient_accumulation: int = 8,
    learning_rate: float = 1e-5,
    checkpoint_every: int = 500,
    seed: int = 0,
    run_name: str = DEFAULT_RUN,
    spawn: bool = False,
) -> None:
    """Start fine-tuning; resumes from the run's ``latest.pt`` if present.

    Use ``modal run --detach modal_app.py::train --spawn`` for long runs: the
    job starts and the command returns at once, so the run no longer depends
    on this machine's connection to Modal. Follow it with
    ``modal app logs <app id>`` or ``runs/<run_name>/latest.json``.
    """
    kwargs = dict(
        total_steps=total_steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        checkpoint_every=checkpoint_every,
        seed=seed,
        run_name=run_name,
    )
    if spawn:
        call = finetune.spawn(**kwargs)
        print(f"[train] spawned {call.object_id}")
        return
    print(f"[train] finished: {finetune.remote(**kwargs)}")
