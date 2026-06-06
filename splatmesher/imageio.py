"""Tiny image I/O helpers built on Pillow."""

from __future__ import annotations

import numpy as np
from PIL import Image


def save_image(image: np.ndarray, path: str) -> None:
    """Save a float RGB image to disk as an 8-bit PNG/JPEG.

    Args:
        image: Array (H, W, 3) of RGB values in [0, 1] (float) or [0, 255]
            (uint8). Float images are scaled to 8-bit.
        path: Output file path; the extension determines the format.

    Returns:
        None. The image is written to ``path``.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
