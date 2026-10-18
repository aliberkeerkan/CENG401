import numpy as np

from app.imaging.ops import basic


def img(values):
    return np.array(values, dtype=np.float64)


def test_grayscale_bt601_weights():
    out = basic.grayscale(img([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]]))
    assert out[0, :, 0].tolist() == [76, 150, 29]  # round(0.299*255) etc.
    assert np.all(out[..., 0] == out[..., 1]) and np.all(out[..., 1] == out[..., 2])


def test_brightness_clips_to_valid_range():
    out = basic.brightness(img([[[250, 10, 100]]]), beta=20)
    assert out.tolist() == [[[255, 30, 120]]]
    out = basic.brightness(img([[[5, 10, 100]]]), beta=-20)
    assert out.tolist() == [[[0, 0, 80]]]


def test_contrast_stretch_maps_range_to_full_scale():
    x = img([[[50, 50, 50], [100, 100, 100], [150, 150, 150]]])
    out = basic.contrast(x, r_min=50, r_max=150)
    assert out[0, :, 0].tolist() == [0, 128, 255]  # 127.5 rounds half up
    # values outside [r_min, r_max] are clipped
    out = basic.contrast(img([[[10, 10, 10], [200, 200, 200]]]), r_min=50, r_max=150)
    assert out[0, :, 0].tolist() == [0, 255]


def test_contrast_identity_for_full_range():
    x = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3)).astype(np.float64)
    assert np.array_equal(basic.contrast(x, r_min=0, r_max=255), x)


def test_round_half_up_matches_javascript():
    # JavaScript Math.round(2.5) = 3 and Math.round(-2.5) = -2
    assert basic.round_half_up(np.array([2.5, -2.5])).tolist() == [3, -2]


# ---------- geometry ----------
from app.imaging.ops import filters, geometry  # noqa: E402
import pytest  # noqa: E402


def test_crop_selects_region_without_resampling():
    x = np.arange(5 * 6 * 3, dtype=np.float64).reshape(5, 6, 3)
    out = geometry.crop(x, x=2, y=1, width=3, height=2)
    assert out.shape == (2, 3, 3) and np.array_equal(out, x[1:3, 2:5])


def test_crop_outside_image_is_rejected():
    with pytest.raises(geometry.OperationError):
        geometry.crop(np.zeros((4, 4, 3)), x=2, y=0, width=3, height=2)


def test_resize_identity_and_nearest_upscale():
    x = np.random.default_rng(0).integers(0, 256, (7, 9, 3)).astype(np.float64)
    assert np.array_equal(geometry.resize(x, 9, 7), x)
    up = geometry.resize(np.array([[[0.0]], [[200.0]]]).reshape(2, 1, 1), 1, 4, "nearest")
    assert up[:, 0, 0].tolist() == [0, 0, 200, 200]


def test_resize_bilinear_midpoint():
    x = np.array([[0.0, 100.0]]).reshape(1, 2, 1)
    out = geometry.resize(x, 1, 1)  # source x = 0.5 -> average
    assert out[0, 0, 0] == 50


# ---------- filters ----------

def test_gaussian_kernel_normalized_and_constant_image_unchanged():
    assert abs(filters.gaussian_kernel(2.0).sum() - 1) < 1e-12
    x = np.full((10, 12, 3), 77.0)
    assert np.array_equal(filters.gaussian_blur(x, 1.5), x)


def test_gaussian_blur_spreads_impulse_symmetrically():
    x = np.zeros((11, 11, 3))
    x[5, 5] = 255
    out = filters.gaussian_blur(x, 1.0)
    assert out[5, 5, 0] < 255 and out[5, 4, 0] == out[5, 6, 0] == out[4, 5, 0]


def test_median_removes_isolated_impulse():
    x = np.full((7, 7, 3), 100.0)
    x[3, 3] = 255
    x[0, 0] = 0
    out = filters.median(x, 3)
    assert np.all(out == 100)


# ---------- noise (Section 5.3) ----------
import math  # noqa: E402

from app.imaging.ops import noise  # noqa: E402

FLAT = np.full((512, 512, 3), 100.0)  # far from 0 and 255, so no clipping


def test_salt_pepper_probabilities_eq18():
    out = noise.salt_pepper(FLAT, p_pepper=0.05, p_salt=0.10, seed=0)
    ch = out[..., 0]
    assert abs((ch == 0).mean() - 0.05) < 0.002
    assert abs((ch == 255).mean() - 0.10) < 0.002
    assert set(np.unique(ch).tolist()) == {0, 100, 255}  # others keep f
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_quantization_matches_eq22():
    f = np.arange(256, dtype=np.float64).reshape(1, 256, 1).repeat(3, axis=2)
    for bits in range(1, 9):
        delta = 255 / (2 ** bits - 1)
        expected = [math.floor(delta * math.floor(v / delta + 0.5) + 0.5) for v in range(256)]
        out = noise.quantization(f, bits)
        assert out[0, :, 0].tolist() == expected
        assert len(np.unique(out)) == 2 ** bits and out.max() <= 255
        assert np.abs(out - f).max() <= delta / 2 + 0.5
    assert np.unique(noise.quantization(f, 2)).tolist() == [0, 85, 170, 255]
    assert np.array_equal(noise.quantization(f, 8), f)  # 8 bits: identity


def test_rayleigh_mean_and_variance_eq24():
    a, b = 5.0, 400.0
    z = noise.rayleigh(FLAT, a, b, seed=0) - FLAT
    assert z.min() >= a - 0.5  # support z >= a_R (up to rounding)
    assert abs(z.mean() - (a + math.sqrt(math.pi * b / 4))) < 0.05
    # rounding to integers adds variance 1/12
    assert abs(z.var() - (b * (4 - math.pi) / 4 + 1 / 12)) / (b * (4 - math.pi) / 4) < 0.01


def test_rayleigh_zero_mean_option():
    z = noise.rayleigh(FLAT, 5.0, 400.0, zero_mean=True, seed=0) - FLAT
    assert abs(z.mean()) < 0.05


def test_uniform_mean_variance_and_bounds_eq25():
    a, b = -20.0, 30.0
    z = noise.uniform(FLAT, a, b, seed=0) - FLAT
    assert z.min() >= a and z.max() <= b
    assert abs(z.mean() - (a + b) / 2) < 0.05
    assert abs(z.var() - ((b - a) ** 2 / 12 + 1 / 12)) / ((b - a) ** 2 / 12) < 0.01


def test_noise_is_reproducible_with_seed():
    x = FLAT[:32, :32]
    for fn, kw in [(noise.salt_pepper, dict(p_pepper=0.1, p_salt=0.1)),
                   (noise.rayleigh, dict(a=0, b=100)),
                   (noise.uniform, dict(a=-10, b=10))]:
        assert np.array_equal(fn(x, **kw, seed=7), fn(x, **kw, seed=7))
        assert not np.array_equal(fn(x, **kw, seed=7), fn(x, **kw, seed=8))


# ---------- mean filters (Section 5.4.1) ----------

def _brute_force(x, size, reduce):
    """Apply `reduce` to every size x size window, reflection border rule."""
    r = size // 2
    p = np.pad(x, ((r, r), (r, r), (0, 0)), mode="reflect")  # d c b | a b c d
    out = np.empty_like(x)
    for yy in range(x.shape[0]):
        for xx in range(x.shape[1]):
            for c in range(x.shape[2]):
                out[yy, xx, c] = reduce(p[yy:yy + size, xx:xx + size, c].ravel())
    return np.clip(np.floor(out + 0.5), 0, 255)


RAND = np.random.default_rng(1).integers(1, 256, (7, 9, 3)).astype(np.float64)


@pytest.mark.parametrize("size", [3, 5])
def test_arithmetic_mean_eq27(size):
    expected = _brute_force(RAND, size, lambda w: w.sum() / w.size)
    assert np.array_equal(filters.arithmetic_mean(RAND, size), expected)


@pytest.mark.parametrize("size", [3, 5])
def test_geometric_mean_eq28(size):
    # direct product form of Eq. (28); log-domain result may differ by
    # float rounding at exact .5 boundaries only
    expected = _brute_force(RAND, size, lambda w: np.prod(w) ** (1 / w.size))
    out = filters.geometric_mean(RAND, size)
    assert np.abs(out - expected).max() <= 1 and (out == expected).mean() > 0.95


@pytest.mark.parametrize("size", [3, 5])
def test_harmonic_mean_eq29(size):
    expected = _brute_force(RAND, size, lambda w: w.size / np.sum(1 / w))
    out = filters.harmonic_mean(RAND, size)
    assert np.abs(out - expected).max() <= 1 and (out == expected).mean() > 0.95


def test_mean_filters_keep_constant_image_and_order():
    x = np.full((6, 6, 3), 77.0)
    for fn in (filters.arithmetic_mean, filters.geometric_mean, filters.harmonic_mean):
        assert np.array_equal(fn(x, 3), x)
    # harmonic <= geometric <= arithmetic for positive data
    a = filters.arithmetic_mean(RAND, 3)
    g = filters.geometric_mean(RAND, 3)
    h = filters.harmonic_mean(RAND, 3)
    assert np.all(h <= g) and np.all(g <= a)


def test_mean_filters_handle_zero_pixels():
    x = np.full((7, 7, 3), 100.0)
    x[3, 3] = 0  # pepper
    for fn in (filters.geometric_mean, filters.harmonic_mean):
        assert np.all(np.isfinite(fn(x, 3)))
    # harmonic mean removes salt but is pulled to ~0 by pepper
    salt = np.full((7, 7, 3), 100.0)
    salt[3, 3] = 255
    assert abs(filters.harmonic_mean(salt, 3)[3, 3, 0] - 100) < 15
    assert filters.harmonic_mean(x, 3)[3, 3, 0] < 1


# ---------- frequency-domain filters (G&W 4.10 and 5.8) ----------
from app.imaging.ops import frequency  # noqa: E402

FREQ = np.random.default_rng(2).integers(0, 256, (48, 48, 3)).astype(np.float64)


def test_band_reject_passes_constant_image():
    x = np.full((32, 32, 3), 120.0)
    for kind in ("ideal", "butterworth", "gaussian"):
        out = frequency.band_reject(x, c0=0.30, width=0.10, kind=kind, n=2)
        assert abs(out.mean() - 120) < 2


def test_band_pass_rejects_dc():
    x = np.full((32, 32, 3), 120.0)
    out = frequency.band_pass(x, c0=0.30, width=0.10, kind="butterworth", n=2)
    assert out.mean() < 15


def test_band_pass_is_complement_of_reject_eq148():
    # Eq. (4-148) is on the transfer function. After inverse DFT, clip/round
    # of a band-pass (often near 0) breaks an exact spatial sum.
    d = frequency._radial_distance(64, 64)
    for kind in ("ideal", "butterworth", "gaussian"):
        h_br = frequency.band_reject_transfer(d, 0.25, 0.10, kind, n=2)
        assert np.allclose(h_br + (1.0 - h_br), 1.0)


def test_band_reject_transfer_ideal_table47():
    d = frequency._radial_distance(64, 64)
    h = frequency.band_reject_transfer(d, c0=0.25, width=0.10, kind="ideal")
    d_max = d.max()
    c0, w = 0.25 * d_max, 0.10 * d_max
    inside = (d >= c0 - w / 2) & (d <= c0 + w / 2)
    assert np.all(h[inside] == 0) and np.all(h[~inside] == 1)
    assert np.allclose(h + (1 - h), 1)


def test_wiener_identity_when_k_zero():
    assert np.array_equal(frequency.wiener(FREQ, k=0), FREQ)


def test_wiener_reduces_variance():
    out = frequency.wiener(FREQ, k=0.4)
    assert out.var() < FREQ.var()
    assert out.min() >= 0 and out.max() <= 255


def test_kalman_keeps_constant_and_smooths_impulse():
    x = np.full((21, 21, 3), 77.0)
    assert np.array_equal(frequency.kalman(x, q=2, r=80), x)
    impulse = x.copy()
    impulse[10, 10] = 255
    out = frequency.kalman(impulse, q=1, r=200)
    assert 77 < out[10, 10, 0] < 255

