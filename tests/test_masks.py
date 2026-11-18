import random

import numpy as np
import pytest
import torch

from face_inpainting.masks import EVAL_MASK_KINDS, eval_mask, random_training_masks


def test_box_mask_is_centred_96px_square():
    mask = eval_mask("box", 0)
    assert mask.shape == (256, 256)
    assert (mask == 0).sum() == 96 * 96
    assert mask[128, 128] == 0 and mask[0, 0] == 1


def test_half_mask_removes_left_half():
    mask = eval_mask("half", 0)
    assert (mask[:, :128] == 0).all() and (mask[:, 128:] == 1).all()


def test_irregular_mask_is_deterministic_per_index():
    np.testing.assert_array_equal(eval_mask("irregular", 3), eval_mask("irregular", 3))
    assert not np.array_equal(eval_mask("irregular", 3), eval_mask("irregular", 4))


@pytest.mark.parametrize("kind", EVAL_MASK_KINDS)
def test_eval_masks_have_known_and_hole_pixels(kind):
    for index in range(8):
        mask = eval_mask(kind, index)
        assert set(np.unique(mask)) == {0, 1}


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        eval_mask("circle", 0)


def test_training_masks_shape_values_and_seeding():
    random.seed(0)
    first = random_training_masks(6, "cpu")
    random.seed(0)
    second = random_training_masks(6, "cpu")
    assert first.shape == (6, 1, 256, 256)
    assert torch.equal(first, second)
    assert set(first.unique().tolist()) == {0.0, 1.0}
    hole = 1 - first.flatten(1).mean(1)
    assert (hole > 0).all() and (hole < 1).all()
