import math

import torch

from face_inpainting.metrics import hole_l1, hole_psnr


def test_metrics_ignore_known_pixels():
    target = torch.zeros(1, 3, 4, 4)
    pred = torch.zeros(1, 3, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    mask[..., :2, :] = 0                 # top half is the hole
    pred[..., 2:, :] = 1.0               # errors only in the known half
    assert hole_l1(pred, target, mask).item() == 0
    assert math.isclose(hole_psnr(pred, target, mask).item(), 100.0, rel_tol=1e-5)  # mse clamped at 1e-10


def test_psnr_value_in_unit_range():
    target = torch.zeros(1, 3, 4, 4)
    pred = torch.full((1, 3, 4, 4), 0.2)  # 0.1 error in [0, 1] units
    mask = torch.zeros(1, 1, 4, 4)
    assert math.isclose(hole_psnr(pred, target, mask).item(), 20.0, rel_tol=1e-5)
    assert math.isclose(hole_l1(pred, target, mask).item(), 0.1, rel_tol=1e-5)
