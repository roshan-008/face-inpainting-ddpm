"""Face inpainting with a mask-aware fine-tune of Google's CelebA-HQ DDPM.

Mask convention everywhere: 1 = known pixel, 0 = hole to fill.
Images are float tensors in [-1, 1], shaped ``B x 3 x 256 x 256``.
"""

__version__ = "0.1.0"
