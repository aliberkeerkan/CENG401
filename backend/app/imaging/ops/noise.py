"""Noise models.

Salt-and-pepper follows Gonzalez and Woods: a pixel is replaced by pepper
(0) with probability P_a or salt (255) with probability P_b.

Quantization follows the uniform mid-tread quantizer. The error
e = g - f is modeled as uniform on [-Δ/2, Δ/2], so σ² = Δ²/12.
"""

import numpy as np

from .basic import clip, round_half_up


def _uniforms(n: int, seed: int = 1) -> np.ndarray:
    """Same LCG as the frontend (Numerical Recipes): s = 1664525 s + 1013904223 mod 2^32."""
    out = np.empty(n, dtype=np.float64)
    s = seed & 0xFFFFFFFF
    for i in range(n):
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        out[i] = s / 4294967296
    return out


def salt_pepper(rgb: np.ndarray, pepper: float, salt: float) -> np.ndarray:
    """Impulse noise. One draw per pixel, applied to all colour channels."""
    out = rgb.copy()
    if pepper <= 0 and salt <= 0:
        return out
    h, w = rgb.shape[:2]
    u = _uniforms(h * w).reshape(h, w)
    pepper_mask = u < pepper
    salt_mask = (u >= pepper) & (u < pepper + salt)
    out[pepper_mask] = 0
    out[salt_mask] = 255
    return out


def quantization(rgb: np.ndarray, bits: int) -> np.ndarray:
    """Requantize each channel to 2^bits levels on [0, 255]. bits = 8 is identity."""
    bits = int(bits)
    if bits >= 8:
        return rgb.copy()
    step = 255.0 / (2**bits - 1)
    return clip(round_half_up(rgb / step) * step)
