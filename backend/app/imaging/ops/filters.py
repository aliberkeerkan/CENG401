"""Smoothing filters (project plan, Sections 5.2.5 and 5.4.2).

Border handling: reflection without repeating the edge pixel
(d c b | a b c d | c b a), the same rule as frontend/src/imageOps.js.
"""

import math

import numpy as np
from scipy.ndimage import median_filter as _scipy_median


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Eq. (10), sampled on [-r, r] with r = ceil(3 sigma), normalized to sum 1.

    The 2-D Gaussian is separable, so one 1-D kernel is used twice.
    """
    r = max(1, math.ceil(3 * sigma))
    # Plain loops (not vectorized) so the weights are computed exactly like
    # the JavaScript version and previews match the server result.
    k = [math.exp(-(t * t) / (2 * sigma * sigma)) for t in range(-r, r + 1)]
    total = 0.0
    for v in k:
        total += v
    return np.array([v / total for v in k], dtype=np.float64)


def _reflect_index(n: int, r: int) -> np.ndarray:
    idx = np.arange(-r, n + r)
    if n == 1:
        return np.zeros_like(idx)
    period = 2 * (n - 1)
    idx = np.abs(idx) % period
    return np.where(idx >= n, period - idx, idx)


def gaussian_blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel(sigma)
    r = len(k) // 2
    h, w = rgb.shape[:2]

    # horizontal pass: taps accumulated in the same order as the JS version
    src = rgb[:, _reflect_index(w, r)]
    tmp = np.zeros_like(rgb, dtype=np.float64)
    for i, wt in enumerate(k):
        tmp = tmp + wt * src[:, i:i + w]

    # vertical pass
    src = tmp[_reflect_index(h, r)]
    out = np.zeros_like(rgb, dtype=np.float64)
    for i, wt in enumerate(k):
        out = out + wt * src[i:i + h]

    return np.clip(np.floor(out + 0.5), 0, 255)


def median(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (33): median over a size x size window, each channel separately."""
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        out[..., c] = _scipy_median(rgb[..., c], size=size, mode="mirror")
    return out
