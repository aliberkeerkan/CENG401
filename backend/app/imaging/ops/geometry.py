"""Geometric tools (project plan, Sections 5.2.1-5.2.2).

These change the image size, so they work on any number of channels and
are applied to the alpha channel too. Inputs and outputs are float arrays
of shape (H, W, C). Index arithmetic is written step by step so that it
matches frontend/src/imageOps.js exactly.
"""

import numpy as np

MAX_SIDE = 4000


class OperationError(ValueError):
    """Raised when parameters do not fit the current image (HTTP 422)."""


def crop(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Eq. (4): g(x, y) = f(x + x_c, y + y_c). No interpolation."""
    h, w = img.shape[:2]
    if x + width > w or y + height > h:
        raise OperationError(
            f"Crop area {x},{y} {width}x{height} is outside the {w}x{h} image."
        )
    return img[y:y + height, x:x + width].copy()


def _source_coords(n_out: int, n_in: int) -> np.ndarray:
    """Pixel-centre alignment: x = (x' + 1/2) * M / M' - 1/2."""
    return (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5


def resize(img: np.ndarray, width: int, height: int, method: str = "bilinear") -> np.ndarray:
    h, w = img.shape[:2]
    if width > MAX_SIDE or height > MAX_SIDE:
        raise OperationError(f"Output size is limited to {MAX_SIDE} px per side.")
    xs = _source_coords(width, w)
    ys = _source_coords(height, h)

    if method == "nearest":
        xi = np.clip(np.floor(xs + 0.5), 0, w - 1).astype(int)
        yi = np.clip(np.floor(ys + 0.5), 0, h - 1).astype(int)
        return img[yi][:, xi].copy()

    # Eq. (5), bilinear. Source coordinates are clamped to the image so the
    # border pixels are repeated instead of reading outside the image.
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    a = (xs - x0)[None, :, None]
    b = (ys - y0)[:, None, None]
    f00 = img[y0][:, x0]
    f10 = img[y0][:, x1]
    f01 = img[y1][:, x0]
    f11 = img[y1][:, x1]
    out = (1 - a) * (1 - b) * f00 + a * (1 - b) * f10 + (1 - a) * b * f01 + a * b * f11
    return np.clip(np.floor(out + 0.5), 0, 255)
