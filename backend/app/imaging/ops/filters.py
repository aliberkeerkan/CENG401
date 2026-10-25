"""Smoothing filters (project plan, Sections 5.2.5 and 5.4).

Border handling: reflection without repeating the edge pixel
(d c b | a b c d | c b a), the same rule as frontend/src/imageOps.js.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter as _scipy_max
from scipy.ndimage import median_filter as _scipy_median
from scipy.ndimage import minimum_filter as _scipy_min


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


def minimum(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (31): f^_min(x, y) = min of g over S_xy, each channel separately.

    Suppresses isolated bright (salt) impulses.
    """
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        out[..., c] = _scipy_min(rgb[..., c], size=size, mode="mirror")
    return out


def maximum(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (32): f^_max(x, y) = max of g over S_xy, each channel separately.

    Suppresses isolated dark (pepper) impulses.
    """
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        out[..., c] = _scipy_max(rgb[..., c], size=size, mode="mirror")
    return out


def midpoint(rgb: np.ndarray, size: int) -> np.ndarray:
    """Eq. (34): f^_mid(x, y) = (max + min) / 2 over S_xy.

    Most naturally associated with bounded random noise such as uniform
    noise; not a generally optimal salt-and-pepper filter.
    """
    lo = minimum(rgb, size)
    hi = maximum(rgb, size)
    return np.clip(np.floor((lo + hi) / 2 + 0.5), 0, 255)


# Order statistics are the slow part of these filters. The image is cut
# into horizontal strips that are processed in parallel threads (NumPy and
# SciPy release the GIL here); each strip carries the extra rows its windows
# need, so the result is identical to processing the whole image at once.
_WORKERS = min(8, os.cpu_count() or 1)


def _strips(h: int, n: int) -> list[slice]:
    bounds = np.linspace(0, h, min(h, n) + 1).astype(int)
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _windows(channel: np.ndarray, size: int, rows: slice) -> np.ndarray:
    """The size x size windows S_xy of the given rows, as (rows, W, size*size).

    Same border rule as the other filters (reflection without repeating
    the edge pixel).
    """
    r = size // 2
    h, w = channel.shape
    ys = _reflect_index(h, r)[rows.start: rows.stop + 2 * r]
    padded = channel[ys][:, _reflect_index(w, r)]
    win = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return win.reshape(win.shape[0], win.shape[1], size * size)


def alpha_trimmed_mean(rgb: np.ndarray, size: int, d: int) -> np.ndarray:
    """Eq. (35): sort the mn samples of S_xy, drop d/2 from each end and
    average the remaining mn - d:

        f^(x, y) = 1 / (mn - d) * sum_{r = d/2 + 1}^{mn - d/2} g_(r)

    d is even with 0 <= d < mn. d = 0 gives the arithmetic mean (27),
    d = mn - 1 the median (33).
    """
    mn = size * size
    k = d // 2
    h, w = rgb.shape[:2]
    out = np.empty(rgb.shape, dtype=np.float64)
    # small strips: all windows of a large image at once would need
    # gigabytes of memory for a 15 x 15 window
    rows_per_strip = max(1, 1_000_000 // (w * mn))

    def run(task):
        c, rows = task
        win = _windows(rgb[..., c], size, rows)
        if k:
            # only the d/2 smallest and d/2 largest have to be found,
            # a full sort is not needed
            win = np.partition(win, (k - 1, mn - k), axis=-1)[..., k:mn - k]
        out[rows, :, c] = win.sum(axis=-1) / (mn - d)

    tasks = [(c, rows) for c in range(rgb.shape[2])
             for rows in _strips(h, -(-h // rows_per_strip))]
    with ThreadPoolExecutor(_WORKERS) as ex:
        list(ex.map(run, tasks))
    return np.clip(np.floor(out + 0.5), 0, 255)


def _min_max_median(z: np.ndarray, size: int, ex: ThreadPoolExecutor):
    """z_min, z_max and z_med of every size x size window, strip by strip."""
    r = size // 2
    h = z.shape[0]
    z_min, z_max, z_med = np.empty_like(z), np.empty_like(z), np.empty_like(z)

    def run(rows):
        lo, hi = max(0, rows.start - r), min(h, rows.stop + r)
        part = z[lo:hi]
        keep = slice(rows.start - lo, rows.start - lo + (rows.stop - rows.start))
        z_min[rows] = _scipy_min(part, size=size, mode="mirror")[keep]
        z_max[rows] = _scipy_max(part, size=size, mode="mirror")[keep]
        z_med[rows] = _scipy_median(part, size=size, mode="mirror")[keep]

    list(ex.map(run, _strips(h, 2 * _WORKERS)))
    return z_min, z_max, z_med


def adaptive_median(rgb: np.ndarray, s_max: int) -> np.ndarray:
    """Adaptive median filter (Section 5.4.3), starting from a 3 x 3 window.

    Stage A: if z_min < z_med < z_max, go to stage B. Otherwise enlarge the
             window by 2; if it would exceed s_max, output z_med.
    Stage B: if z_min < z_xy < z_max, output z_xy, otherwise z_med.

    z_min, z_max and z_med are the minimum, maximum and median of the
    window around the pixel and z_xy is the pixel itself. Each window size
    is computed for the whole image at once; every pixel takes its result
    from the first size at which stage A passes.
    """
    out = np.empty_like(rgb, dtype=np.float64)
    with ThreadPoolExecutor(_WORKERS) as ex:
        for c in range(rgb.shape[2]):
            z = np.ascontiguousarray(rgb[..., c], dtype=np.float64)
            result = np.empty_like(z)
            pending = np.ones(z.shape, dtype=bool)
            for size in range(3, s_max + 1, 2):
                z_min, z_max, z_med = _min_max_median(z, size, ex)
                stage_a = pending & (z_min < z_med) & (z_med < z_max)
                keep = (z_min < z) & (z < z_max)  # stage B: not an impulse
                result[stage_a] = np.where(keep, z, z_med)[stage_a]
                pending &= ~stage_a
                if size == s_max:
                    result[pending] = z_med[pending]
                if not pending.any():
                    break
            out[..., c] = result
    return out
