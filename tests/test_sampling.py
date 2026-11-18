import torch

from face_inpainting.model import widen_input
from face_inpainting.sampling import inpaint


def _inputs():
    torch.manual_seed(1)
    image = torch.rand(2, 3, 32, 32) * 2 - 1
    mask = torch.ones(2, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 0
    return image, mask


def test_known_pixels_are_exact_and_output_in_range(tiny_unet, tiny_scheduler):
    unet, _ = widen_input(tiny_unet)
    image, mask = _inputs()
    out = inpaint(unet.eval(), tiny_scheduler, image, mask, torch.Generator().manual_seed(0), dtype=torch.float32)
    assert out.shape == image.shape
    known = mask.expand_as(image).bool()
    torch.testing.assert_close(out[known], image[known], rtol=0, atol=0)
    assert out.min() >= -1 and out.max() <= 1


def test_sampling_is_deterministic_for_a_seed(tiny_unet, tiny_scheduler):
    unet, _ = widen_input(tiny_unet)
    image, mask = _inputs()
    a = inpaint(unet.eval(), tiny_scheduler, image, mask, torch.Generator().manual_seed(0), dtype=torch.float32)
    b = inpaint(unet.eval(), tiny_scheduler, image, mask, torch.Generator().manual_seed(0), dtype=torch.float32)
    c = inpaint(unet.eval(), tiny_scheduler, image, mask, torch.Generator().manual_seed(1), dtype=torch.float32)
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)
