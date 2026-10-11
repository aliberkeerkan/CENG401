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


# Added to g before ln(g) and 1/g so that zero pixels (for example pepper
# impulses) stay finite (Section 5.4.1).
EPS = 1e-6


def _window_mean(x: np.ndarray, size: int) -> np.ndarray:
    """(1/mn) * sum of x over the size x size window S_xy around each pixel.

    The box sum is separable: a horizontal pass, then a vertical pass.
    """
    r = size // 2
    h, w = x.shape[:2]

    src = x[:, _reflect_index(w, r)]
    tmp = np.zeros_like(x, dtype=np.float64)
    for i in range(size):
        tmp = tmp + src[:, i:i + w]

    src = tmp[_reflect_index(h, r)]
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(size):
        out = out + src[i:i + h]

    return out / (size * size)


def arithmetic_mean(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (27): f^(x, y) = (1/mn) sum g(s, t) over S_xy."""
    return np.clip(np.floor(_window_mean(rgb, size) + 0.5), 0, 255)


def geometric_mean(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (28), in the log domain: exp((1/mn) sum ln(g(s, t) + eps)).

    The direct product of mn values overflows for large windows.
    """
    out = np.exp(_window_mean(np.log(rgb + EPS), size))
    return np.clip(np.floor(out + 0.5), 0, 255)


def harmonic_mean(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (29): f^(x, y) = mn / sum 1/(g(s, t) + eps).

    Works well for salt noise; a single pepper (0) pixel pulls the window
    to ~0, as expected for the harmonic mean.
    """
    out = 1.0 / _window_mean(1.0 / (rgb + EPS), size)
    return np.clip(np.floor(out + 0.5), 0, 255)


def median(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (33): median over a size x size window, each channel separately."""
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        out[..., c] = _scipy_median(rgb[..., c], size=size, mode="mirror")
    return out
