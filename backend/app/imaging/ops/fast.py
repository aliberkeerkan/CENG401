"""Fast versions of the window filters for the live camera.

The camera sends many frames a second, and the exact filters in filters.py
(written to match the frontend and the formulas step by step) take 50-450 ms
per 640 px frame. These use OpenCV for the same formulas:

  gaussian_blur, sharpen   the same sampled kernel (Eq. 10), separable,
                           same reflection border; float32 arithmetic
  arithmetic_mean (27)     box filter
  geometric_mean (28)      box filter of ln(g + eps), then exp
  harmonic_mean (29)       box filter of 1 / (g + eps), then 1 / x
  median (33)              OpenCV median; the border repeats the edge pixel
  minimum, maximum (31-32) erosion / dilation with a square window
  midpoint (34)            (min + max) / 2

Results equal the exact filters up to rounding (at most 1 gray level),
except the median, which may differ in the outermost size // 2 rows and
columns because of its border rule. Inputs and outputs are float64 RGB
arrays of shape (H, W, 3) in [0, 255], like every other operation.
"""

import cv2
import numpy as np

from .filters import EPS, gaussian_kernel

_REFLECT = cv2.BORDER_REFLECT_101  # d c b | a b c d, the project's border rule


def _round(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(x.astype(np.float64) + 0.5), 0, 255)


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8))


def _gaussian(rgb: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel(sigma).astype(np.float32)
    return cv2.sepFilter2D(rgb.astype(np.float32), -1, k, k, borderType=_REFLECT)


def _box(x: np.ndarray, size: int) -> np.ndarray:
    return cv2.blur(x.astype(np.float32), (size, size), borderType=_REFLECT)


def gaussian_blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return _round(_gaussian(rgb, sigma))


def sharpen(rgb: np.ndarray, amount: float, sigma: float = 1.5) -> np.ndarray:
    return _round(rgb + amount * (rgb - _gaussian(rgb, sigma)))


def arithmetic_mean(rgb: np.ndarray, size: int) -> np.ndarray:
    return _round(_box(rgb, size))


def geometric_mean(rgb: np.ndarray, size: int) -> np.ndarray:
    return _round(np.exp(_box(np.log(rgb + EPS), size)))


def harmonic_mean(rgb: np.ndarray, size: int) -> np.ndarray:
    return _round(1.0 / _box(1.0 / (rgb + EPS), size))


def median(rgb: np.ndarray, size: int) -> np.ndarray:
    return cv2.medianBlur(_u8(rgb), size).astype(np.float64)


def minimum(rgb: np.ndarray, size: int) -> np.ndarray:
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(_u8(rgb), kernel, borderType=_REFLECT).astype(np.float64)


def maximum(rgb: np.ndarray, size: int) -> np.ndarray:
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(_u8(rgb), kernel, borderType=_REFLECT).astype(np.float64)


def midpoint(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.clip(np.floor((minimum(rgb, size) + maximum(rgb, size)) / 2 + 0.5), 0, 255)


# operation type -> fast function, used by run_pipeline(fast=True)
FAST = {
    "gaussian_blur": gaussian_blur,
    "sharpen": sharpen,
    "arithmetic_mean": arithmetic_mean,
    "geometric_mean": geometric_mean,
    "harmonic_mean": harmonic_mean,
    "median": median,
    "minimum": minimum,
    "maximum": maximum,
    "midpoint": midpoint,
}
