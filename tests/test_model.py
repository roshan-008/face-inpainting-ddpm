import torch

from face_inpainting.model import IN_CHANNELS, load_checkpoint, widen_input


def test_widened_model_matches_pretrained_before_training(tiny_unet):
    """With zero-initialised condition channels the 7-channel model must ignore them."""
    wide, config = widen_input(tiny_unet)
    assert config["in_channels"] == IN_CHANNELS
    assert wide.conv_in.in_channels == IN_CHANNELS
    x = torch.randn(2, 3, 32, 32)
    cond = torch.randn(2, 4, 32, 32)
    t = torch.tensor([3, 7])
    with torch.no_grad():
        expected = tiny_unet(x, t).sample
        actual = wide.eval()(torch.cat((x, cond), dim=1), t).sample
    torch.testing.assert_close(actual, expected)


def test_widen_keeps_rgb_filters_and_zeroes_new_channels(tiny_unet):
    wide, _ = widen_input(tiny_unet)
    torch.testing.assert_close(wide.conv_in.weight[:, :3], tiny_unet.conv_in.weight)
    assert torch.count_nonzero(wide.conv_in.weight[:, 3:]) == 0


def test_checkpoint_round_trip(tiny_unet, tmp_path):
    wide, config = widen_input(tiny_unet)
    path = tmp_path / "ckpt.pt"
    torch.save({"unet_config": config, "model": wide.state_dict(), "ema": wide.state_dict(), "step": 42}, path)
    loaded, step = load_checkpoint(path, "model", device="cpu", dtype=torch.float32)
    assert step == 42
    for name, tensor in wide.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor)
