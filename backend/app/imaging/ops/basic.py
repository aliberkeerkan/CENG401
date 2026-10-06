"""Basic tools (project plan, Section 5.2).

Every operation takes an RGB image as a float64 array of shape (H, W, 3)
with values in [0, 255] and returns an array of the same shape. Rounding
and clipping follow the plan's convention: compute in floating point,
then round to the nearest integer and clip to [0, L-1].

Rounding uses floor(x + 0.5) so the results match JavaScript's
Math.round, which the frontend uses for its instant slider previews.
"""

import numpy as np

L = 256


def round_half_up(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


def clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, L - 1)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Eq. (6): Y = 0.299 R + 0.587 G + 0.114 B."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def grayscale(rgb: np.ndarray) -> np.ndarray:
    """Eq. (6), stored in all three channels as (Y, Y, Y)."""
    y = round_half_up(luma(rgb))
    return np.repeat(y[..., None], 3, axis=2)


def brightness(rgb: np.ndarray, beta: float) -> np.ndarray:
    """Eq. (8): g = clip(f + beta)."""
    return clip(rgb + beta)


def contrast(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Eq. (9): g = clip(alpha (f - c) + c), with pivot c = mean luma of the input."""
    c = round_half_up(luma(rgb)).mean()
    return clip(round_half_up(alpha * (rgb - c) + c))
