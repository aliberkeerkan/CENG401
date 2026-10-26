"""Noise simulation (project plan, Section 5.3).

Inputs and outputs are float64 RGB arrays of shape (H, W, 3) with values
in [0, 255], like the tools in basic.py. Following Section 5.1, the result
is computed in floating point, then rounded to the nearest integer and
clipped to [0, L-1].

Additive noise (Rayleigh, uniform) is drawn independently for each pixel
and channel. The random models take an optional seed: with the same seed
the same noise is produced, so /process and /export return identical
images; with seed=None every call gives a new noise pattern.
"""

import math

import numpy as np

from .basic import L, clip, round_half_up


def salt_pepper(rgb: np.ndarray, p_pepper: float, p_salt: float,
                seed: int | None = None) -> np.ndarray:
    """Eq. (18): g = 0 with probability P_p, L-1 with probability P_s,
    f otherwise. Requires P_p, P_s >= 0 and P_p + P_s <= 1.

    One draw per pixel, shared by the three channels, so corrupted pixels
    are pure black/white rather than colored.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(rgb.shape[:2])
    out = rgb.copy()
    out[u < p_pepper] = 0
    out[(u >= p_pepper) & (u < p_pepper + p_salt)] = L - 1
    return out


def quantization(rgb: np.ndarray, bits: int) -> np.ndarray:
    """Eq. (22): q(f) = Delta * round(f / Delta), 2^B levels on [0, L-1]
    with step Delta = (L-1) / (2^B - 1).

    With this step q(f) never exceeds L-1. Deterministic, no seed needed.
    """
    delta = (L - 1) / (2 ** bits - 1)
    return clip(round_half_up(delta * round_half_up(rgb / delta)))


def rayleigh(rgb: np.ndarray, a: float, b: float, zero_mean: bool = False,
             seed: int | None = None) -> np.ndarray:
    """Eq. (24): p(z) = (2/b)(z - a) exp(-(z - a)^2 / b), z >= a; g = f + z.

    b is not NumPy's Rayleigh scale: b = 2 sigma_R^2, so the samples are
    drawn with sigma_R = sqrt(b / 2). Mean a + sqrt(pi b / 4), variance
    b (4 - pi) / 4. The positive mean also brightens the image; with
    zero_mean=True it is subtracted so only the fluctuation is added.
    """
    rng = np.random.default_rng(seed)
    z = a + rng.rayleigh(scale=math.sqrt(b / 2), size=rgb.shape)
    if zero_mean:
        z = z - (a + math.sqrt(math.pi * b / 4))
    return clip(round_half_up(rgb + z))


def uniform(rgb: np.ndarray, a: float, b: float,
            seed: int | None = None) -> np.ndarray:
    """Eq. (25): p(z) = 1 / (b - a), a <= z <= b; g = f + z.

    Mean (a + b) / 2, variance (b - a)^2 / 12.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(a, b, rgb.shape)
    return clip(round_half_up(rgb + z))


def exponential(rgb: np.ndarray, a: float, zero_mean: bool = False,
                seed: int | None = None) -> np.ndarray:
    """Eq. (26): p(z) = a_E exp(-a_E z) for z >= 0; g = f + z.

    NumPy's exponential takes the scale 1 / a_E. Mean 1 / a_E, variance
    1 / a_E^2. The positive mean also brightens the image; with
    zero_mean=True it is subtracted so only the fluctuation is added.
    """
    rng = np.random.default_rng(seed)
    z = rng.exponential(scale=1.0 / a, size=rgb.shape)
    if zero_mean:
        z = z - 1.0 / a
    return clip(round_half_up(rgb + z))


def gamma_speckle(rgb: np.ndarray, nu: float, seed: int | None = None) -> np.ndarray:
    """Eqs. (20)-(21): multiplicative speckle g = f S with a gamma gain

        p_S(s) = nu^nu / Gamma(nu) s^(nu - 1) exp(-nu s),  s >= 0,

    i.e. shape nu and scale 1 / nu, so E[S] = 1 and Var(S) = 1 / nu.
    Smaller nu gives stronger fluctuations. One gain per pixel, shared by
    the three channels: it scales the intensity and keeps the colour.
    """
    rng = np.random.default_rng(seed)
    s = rng.gamma(shape=nu, scale=1.0 / nu, size=rgb.shape[:2])
    return clip(round_half_up(rgb * s[..., None]))
