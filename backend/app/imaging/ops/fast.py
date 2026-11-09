"""Fast versions of the spatial filters and denoisers for the live camera.

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
  denoise_poisson          Anscombe transform, Gaussian, algebraic inverse
  denoise_gaussian         local Wiener from box-filter mean and variance
  denoise_rayleigh,        the same local Wiener after the documented
  denoise_exponential      positive-mean correction
  denoise_gamma            the geometric-mean core (Eq. 28)

Results equal the exact filters up to rounding (at most 1 gray level),
except the median and salt-and-pepper denoiser, which may differ in the
outermost size // 2 rows and columns because of OpenCV's median border rule.
Inputs and outputs are float64 RGB arrays of shape (H, W, 3) in [0, 255],
like every other operation. Frequency operations already use FFTs and run
through the same registry without a separate fast implementation.
"""

import math

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


def denoise_salt_pepper(rgb: np.ndarray, size: int) -> np.ndarray:
    return median(rgb, size)


def denoise_uniform(rgb: np.ndarray, size: int) -> np.ndarray:
    return midpoint(rgb, size)


def denoise_quantization(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_blur(rgb, sigma)


def denoise_poisson(rgb: np.ndarray, sigma: float) -> np.ndarray:
    stabilized = 2.0 * np.sqrt(np.clip(rgb, 0, None) + 0.375)
    blurred = _gaussian(stabilized, sigma)
    restored = (blurred / 2.0) ** 2 - 0.375
    return _round(restored)


def denoise_gaussian(rgb: np.ndarray, noise_std: float, size: int) -> np.ndarray:
    if noise_std <= 0:
        return _round(rgb)
    mean = _box(rgb, size)
    variance = np.maximum(_box(rgb * rgb, size) - mean * mean, 0.0)
    gain = np.maximum(variance - noise_std ** 2, 0.0) / np.maximum(variance, EPS)
    return _round(mean + gain * (rgb - mean))


def denoise_rayleigh(rgb: np.ndarray, noise_std: float, size: int) -> np.ndarray:
    if noise_std <= 0:
        return _round(rgb)
    shifted = rgb - noise_std * math.sqrt(math.pi / (4.0 - math.pi))
    return denoise_gaussian(shifted, noise_std, size)


def denoise_exponential(rgb: np.ndarray, noise_std: float, size: int) -> np.ndarray:
    if noise_std <= 0:
        return _round(rgb)
    return denoise_gaussian(rgb - noise_std, noise_std, size)


def denoise_gamma(rgb: np.ndarray, size: int) -> np.ndarray:
    return geometric_mean(rgb, size)


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
    "denoise_salt_pepper": denoise_salt_pepper,
    "denoise_uniform": denoise_uniform,
    "denoise_quantization": denoise_quantization,
    "denoise_poisson": denoise_poisson,
    "denoise_gaussian": denoise_gaussian,
    "denoise_rayleigh": denoise_rayleigh,
    "denoise_exponential": denoise_exponential,
    "denoise_gamma": denoise_gamma,
}
