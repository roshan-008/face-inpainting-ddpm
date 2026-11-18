import numpy as np
import pytest
from PIL import Image

from face_inpainting.data import FaceFolder, load_photo_and_mask


def _save_photo_and_mask(tmp_path, width=320, height=240):
    photo = Image.fromarray(np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8))
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[60:180, 100:220] = 0                     # black = fill
    photo.save(tmp_path / "face.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    return tmp_path / "face.png", tmp_path / "mask.png"


def test_photo_and_mask_are_cropped_resized_and_scaled(tmp_path):
    photo_path, mask_path = _save_photo_and_mask(tmp_path)
    photo, mask = load_photo_and_mask(photo_path, mask_path)
    assert photo.shape == (1, 3, 256, 256) and mask.shape == (1, 1, 256, 256)
    assert photo.min() >= -1 and photo.max() <= 1
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    assert mask[0, 0, 128, 128] == 0 and mask[0, 0, 0, 0] == 1


def test_invert_mask_and_rejects_uniform_masks(tmp_path):
    photo_path, mask_path = _save_photo_and_mask(tmp_path)
    _, mask = load_photo_and_mask(photo_path, mask_path)
    _, inverted = load_photo_and_mask(photo_path, mask_path, invert_mask=True)
    assert (mask + inverted == 1).all()
    Image.new("L", (320, 240), 255).save(tmp_path / "all_white.png")
    with pytest.raises(ValueError):
        load_photo_and_mask(photo_path, tmp_path / "all_white.png")


def test_face_folder_needs_32_images(tmp_path):
    for i in range(31):
        Image.new("RGB", (300, 300)).save(tmp_path / f"{i:05d}.jpg")
    with pytest.raises(ValueError):
        FaceFolder(tmp_path)
    Image.new("RGB", (300, 300)).save(tmp_path / "00031.jpg")
    faces = FaceFolder(tmp_path)
    assert len(faces) == 32 and faces[0].shape == (3, 256, 256)
