import pytest
import torch
from diffusers import DDPMScheduler, UNet2DModel


@pytest.fixture
def tiny_unet() -> UNet2DModel:
    """A 3-channel UNet small enough to run on CPU in milliseconds."""
    torch.manual_seed(0)
    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    ).eval()


@pytest.fixture
def tiny_scheduler() -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=10)
