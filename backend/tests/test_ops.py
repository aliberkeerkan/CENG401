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


# ---------- order-statistic filters (Section 5.4.2) ----------


@pytest.mark.parametrize("size", [3, 5])
def test_minimum_eq31(size):
    expected = _brute_force(RAND, size, lambda w: w.min())
    assert np.array_equal(filters.minimum(RAND, size), expected)


@pytest.mark.parametrize("size", [3, 5])
def test_maximum_eq32(size):
    expected = _brute_force(RAND, size, lambda w: w.max())
    assert np.array_equal(filters.maximum(RAND, size), expected)


@pytest.mark.parametrize("size", [3, 5])
def test_midpoint_eq34(size):
    expected = _brute_force(RAND, size, lambda w: (w.max() + w.min()) / 2)
    assert np.array_equal(filters.midpoint(RAND, size), expected)


def test_order_statistic_filters_suppress_isolated_impulses():
    # minimum removes an isolated bright (salt) impulse
    salt = np.full((7, 7, 3), 100.0)
    salt[3, 3] = 255
    assert filters.minimum(salt, 3)[3, 3, 0] == 100
    # maximum removes an isolated dark (pepper) impulse
    pepper = np.full((7, 7, 3), 100.0)
    pepper[3, 3] = 0
    assert filters.maximum(pepper, 3)[3, 3, 0] == 100
    # midpoint of an isolated impulse is the average of impulse and background
    assert filters.midpoint(salt, 3)[3, 3, 0] == 178   # (255 + 100) / 2 = 177.5
    assert filters.midpoint(pepper, 3)[3, 3, 0] == 50  # (0 + 100) / 2


def test_order_statistic_filters_keep_constant_image_and_order():
    x = np.full((6, 6, 3), 77.0)
    for fn in (filters.minimum, filters.maximum, filters.midpoint):
        assert np.array_equal(fn(x, 3), x)
    # min <= midpoint <= max for any data
    lo = filters.minimum(RAND, 3)
    mid = filters.midpoint(RAND, 3)
    hi = filters.maximum(RAND, 3)
    assert np.all(lo <= mid) and np.all(mid <= hi)


# ---------- frequency-domain filters (G&W 4.10 and 5.8) ----------
from app.imaging.ops import frequency  # noqa: E402

FREQ = np.random.default_rng(2).integers(0, 256, (48, 48, 3)).astype(np.float64)


def test_band_reject_passes_constant_image():
    x = np.full((32, 32, 3), 120.0)
    for kind in ("ideal", "butterworth", "gaussian"):
        out = frequency.band_reject(x, c0=0.30, width=0.10, kind=kind, n=2)
        assert abs(out.mean() - 120) < 2


def test_band_pass_rejects_dc_and_is_shown_around_mid_gray():
    # the mean (DC) is removed; the result is offset by 128 so negative
    # values stay visible instead of being clipped to black
    x = np.full((32, 32, 3), 120.0)
    out = frequency.band_pass(x, c0=0.30, width=0.10, kind="butterworth", n=2)
    assert abs(out.mean() - 128) < 4
    noisy = np.random.default_rng(3).integers(0, 256, (40, 40, 3)).astype(np.float64)
    assert (frequency.band_pass(noisy, c0=0.25, width=0.12) == 0).mean() < 0.01


def _band_reference(rgb, c0, width, kind, n, bandpass):
    """The original full-spectrum implementation (fft2 + fftshift), Eq. (39)."""
    h, w = rgb.shape[:2]
    p, q = 2 * h, 2 * w
    hbr = frequency.band_reject_transfer(frequency._radial_distance(p, q), c0, width, kind, n)
    tf = 1 - hbr if bandpass else hbr
    padded = np.zeros((p, q, 3))
    padded[:h, :w] = rgb
    out = np.empty_like(padded)
    for c in range(3):
        g = np.fft.fftshift(np.fft.fft2(padded[..., c]))
        out[..., c] = np.fft.ifft2(np.fft.ifftshift(tf * g)).real
    out = out[:h, :w] + (128 if bandpass else 0)
    return np.clip(np.floor(out + 0.5), 0, 255)


@pytest.mark.parametrize("kind", ["ideal", "butterworth", "gaussian"])
def test_fast_band_filters_match_fft2_reference(kind):
    x = np.random.default_rng(4).integers(0, 256, (30, 46, 3)).astype(np.float64)
    for bandpass, fn in ((False, frequency.band_reject), (True, frequency.band_pass)):
        out = fn(x, c0=0.3, width=0.1, kind=kind, n=2)
        ref = _band_reference(x, 0.3, 0.1, kind, 2, bandpass)
        assert np.abs(out - ref).max() <= 1  # only .5 rounding ties may differ


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


def test_wiener_identity_without_noise():
    assert np.array_equal(frequency.wiener(FREQ, noise_std=0), FREQ)


def test_wiener_matches_eq44():
    x = np.random.default_rng(5).integers(0, 256, (12, 10, 1)).astype(np.float64)
    sigma = 30.0
    g = np.fft.fft2(x[..., 0])
    s_eta = x.shape[0] * x.shape[1] * sigma ** 2  # S_eta = MN sigma^2
    s_f = np.maximum(np.abs(g) ** 2 - s_eta, 0)
    expected = np.fft.ifft2(s_f / (s_f + s_eta) * g).real
    expected = np.clip(np.floor(expected + 0.5), 0, 255)
    assert np.abs(frequency.wiener(x, noise_std=sigma)[..., 0] - expected).max() <= 1


def test_wiener_removes_additive_noise():
    y, x = np.mgrid[0:96, 0:96]
    clean = np.dstack([110 + 50 * np.sin(x / 11) * np.cos(y / 13)] * 3)
    noisy = clean + np.random.default_rng(6).normal(0, 20, clean.shape)
    out = frequency.wiener(noisy, noise_std=20)
    mse = lambda a: ((a - clean) ** 2).mean()
    assert mse(out) < 0.5 * mse(noisy)


def test_kalman_keeps_constant_and_smooths_impulse():
    x = np.full((21, 21, 3), 77.0)
    assert np.array_equal(frequency.kalman(x, q=2, r=80), x)
    impulse = x.copy()
    impulse[10, 10] = 255
    out = frequency.kalman(impulse, q=1, r=200)
    assert 77 < out[10, 10, 0] < 255


def _kalman_1d_reference(z, q, r):
    """Forward pass and RTS backward pass, written exactly as specified."""
    n = len(z)
    x, p = [0.0] * n, [0.0] * n
    x[0], p[0] = z[0], r
    for i in range(1, n):
        p_pred = p[i - 1] + q
        k = p_pred / (p_pred + r)
        x[i] = x[i - 1] + k * (z[i] - x[i - 1])
        p[i] = (1 - k) * p_pred
    xs = x[:]
    for i in range(n - 2, -1, -1):
        c = p[i] / (p[i] + q)
        xs[i] = x[i] + c * (xs[i + 1] - x[i])
    return xs


def test_kalman_matches_specification_rows_then_columns():
    x = np.random.default_rng(8).integers(0, 256, (9, 11, 3)).astype(np.float64)
    q, r = 3.0, 90.0
    expected = np.empty_like(x)
    for c in range(3):  # each channel on its own
        rows = np.array([_kalman_1d_reference(row, q, r) for row in x[..., c]])
        cols = np.array([_kalman_1d_reference(col, q, r) for col in rows.T]).T
        expected[..., c] = cols
    expected = np.clip(np.floor(expected + 0.5), 0, 255)
    assert np.abs(frequency.kalman(x, q=q, r=r) - expected).max() <= 1


def test_kalman_smooths_more_as_r_over_q_grows():
    noisy = np.clip(np.random.default_rng(9).normal(128, 30, (40, 40, 3)), 0, 255)
    weak = frequency.kalman(noisy, q=10, r=20)
    strong = frequency.kalman(noisy, q=1, r=400)
    assert strong.std() < weak.std() < noisy.std()


# ---------- alpha-trimmed mean (35) and adaptive median (5.4.3) ----------

def _pad(x, r):
    return np.pad(x, ((r, r), (r, r), (0, 0)), mode="reflect")  # d c b | a b c d


def test_alpha_trimmed_mean_matches_eq35():
    x = np.random.default_rng(10).integers(0, 256, (11, 13, 3)).astype(np.float64)
    for size, d in ((3, 2), (3, 4), (5, 8)):
        mn, r = size * size, size // 2
        p = _pad(x, r)
        expected = np.empty_like(x)
        for yy in range(x.shape[0]):
            for xx in range(x.shape[1]):
                for c in range(3):
                    g = np.sort(p[yy:yy + size, xx:xx + size, c].ravel())
                    expected[yy, xx, c] = g[d // 2: mn - d // 2].sum() / (mn - d)
        expected = np.clip(np.floor(expected + 0.5), 0, 255)
        assert np.abs(filters.alpha_trimmed_mean(x, size, d) - expected).max() <= 1


def test_alpha_trimmed_mean_between_mean_and_median():
    x = np.random.default_rng(11).integers(0, 256, (16, 18, 3)).astype(np.float64)
    assert np.array_equal(filters.alpha_trimmed_mean(x, 3, 0), filters.arithmetic_mean(x, 3))
    assert np.array_equal(filters.alpha_trimmed_mean(x, 5, 24), filters.median(x, 5))
    # one salt pixel is trimmed away with d = 2
    flat = np.full((7, 7, 3), 100.0)
    flat[3, 3] = 255
    assert np.all(filters.alpha_trimmed_mean(flat, 3, 2) == 100)


def _adaptive_reference(x, s_max):
    """Stage A / stage B for every pixel, written out as in Section 5.4.3."""
    out = np.empty_like(x)
    for c in range(x.shape[2]):
        for yy in range(x.shape[0]):
            for xx in range(x.shape[1]):
                z_xy, size = x[yy, xx, c], 3
                while True:
                    r = size // 2
                    win = _pad(x, r)[yy:yy + size, xx:xx + size, c]
                    z_min, z_max, z_med = win.min(), win.max(), np.median(win)
                    if z_min < z_med < z_max:  # stage A passed: stage B
                        out[yy, xx, c] = z_xy if z_min < z_xy < z_max else z_med
                        break
                    size += 2
                    if size > s_max:
                        out[yy, xx, c] = z_med
                        break
    return out


def _salt_pepper(x, density, seed):
    u = np.random.default_rng(seed).random(x.shape[:2])
    y = x.copy()
    y[u < density / 2] = 0
    y[(u >= density / 2) & (u < density)] = 255
    return y


def test_adaptive_median_matches_stages_a_and_b():
    clean = np.random.default_rng(12).integers(40, 200, (14, 15, 3)).astype(np.float64)
    noisy = _salt_pepper(clean, 0.4, 13)
    for s_max in (3, 7):
        assert np.array_equal(filters.adaptive_median(noisy, s_max), _adaptive_reference(noisy, s_max))


def test_adaptive_median_beats_median_on_dense_impulse_noise():
    y, x = np.mgrid[0:64, 0:64]
    clean = np.dstack([128 + 60 * np.sin(x / 7) * np.cos(y / 9)] * 3).round()
    noisy = _salt_pepper(clean, 0.5, 14)
    mse = lambda a: ((a - clean) ** 2).mean()
    assert mse(filters.adaptive_median(noisy, 9)) < 0.5 * mse(filters.median(noisy, 3))
    # pixels that are not impulses are kept as they are
    out = filters.adaptive_median(clean, 7)
    assert (out == clean).mean() > 0.6


# ---------- exponential (26) and gamma / speckle (20)-(21) noise ----------

def test_exponential_mean_and_variance_eq26():
    a = 0.05  # mean 1 / a = 20, std 20
    z = noise.exponential(FLAT, a, seed=0) - FLAT
    assert z.min() >= 0
    assert abs(z.mean() - 1 / a) < 0.1
    # rounding to integers adds variance 1/12
    assert abs(z.var() - (1 / a ** 2 + 1 / 12)) / (1 / a ** 2) < 0.01


def test_exponential_zero_mean_option():
    z = noise.exponential(FLAT, 0.05, zero_mean=True, seed=0) - FLAT
    assert abs(z.mean()) < 0.1


def test_gamma_speckle_unit_mean_gain_eq20_21():
    nu = 10.0
    g = noise.gamma_speckle(FLAT, nu, seed=0)
    s = g[..., 0] / 100  # FLAT is 100, so g / f is the gain S
    assert abs(s.mean() - 1) < 0.002
    assert abs(s.var() - 1 / nu) / (1 / nu) < 0.02
    # multiplicative: black stays black, and the gain is shared by R, G, B
    assert np.all(noise.gamma_speckle(np.zeros((8, 8, 3)), nu, seed=1) == 0)
    assert np.array_equal(g[..., 0], g[..., 1]) and np.array_equal(g[..., 1], g[..., 2])


def test_gamma_speckle_smaller_nu_is_stronger():
    weak = noise.gamma_speckle(FLAT, 50, seed=2)
    strong = noise.gamma_speckle(FLAT, 2, seed=2)
    assert strong.std() > weak.std()


def test_exponential_and_gamma_are_reproducible_with_seed():
    x = FLAT[:32, :32]
    for fn, kw in [(noise.exponential, dict(a=0.1)), (noise.gamma_speckle, dict(nu=5))]:
        assert np.array_equal(fn(x, **kw, seed=7), fn(x, **kw, seed=7))
        assert not np.array_equal(fn(x, **kw, seed=7), fn(x, **kw, seed=8))


# ---------- Poisson noise (19) ----------

def test_poisson_mean_and_signal_dependent_variance_eq19():
    kappa = 50.0
    g = noise.poisson(FLAT, kappa, seed=0)  # FLAT is 100
    assert abs(g.mean() - 100) < 0.2  # E[g] = f
    expected_var = 255 * 100 / kappa  # Var(g) = (L-1) f / kappa
    assert abs(g.var() - (expected_var + 1 / 12)) / expected_var < 0.02


def test_poisson_noise_depends_on_the_signal():
    dark = noise.poisson(np.full((256, 256, 3), 20.0), 50, seed=1)
    bright = noise.poisson(np.full((256, 256, 3), 200.0), 50, seed=1)
    assert bright.std() > 2.5 * dark.std()  # std grows with sqrt(f)
    assert np.all(noise.poisson(np.zeros((8, 8, 3)), 50, seed=2) == 0)


def test_poisson_larger_kappa_is_weaker():
    strong = noise.poisson(FLAT, 5, seed=3)
    weak = noise.poisson(FLAT, 500, seed=3)
    assert weak.std() < strong.std()
    assert np.array_equal(noise.poisson(FLAT[:16, :16], 20, seed=4),
                          noise.poisson(FLAT[:16, :16], 20, seed=4))


def test_sharpen_identity_and_soft_edge():
    flat = np.full((16, 16, 3), 90.0)
    assert np.array_equal(filters.sharpen(flat, amount=0), flat)
    step = np.full((31, 31, 3), 40.0)
    step[:, 16:] = 200
    out = filters.sharpen(step, amount=1.5)
    assert out[15, 15, 0] < 40 and out[15, 16, 0] > 200
    soft = filters._gaussian_float(step, 1.5)
    restored = filters.sharpen(soft, amount=1.0, sigma=1.5)
    before = soft[15, 16, 0] - soft[15, 15, 0]
    after = restored[15, 16, 0] - restored[15, 15, 0]
    assert after > before


def test_denoise_extras_clean_their_noise():
    flat = np.full((11, 11, 3), 100.0)
    speckled = flat.copy()
    speckled[3, 3] = 0
    speckled[7, 8] = 255
    assert np.all(filters.denoise_salt_pepper(speckled, 3) == 100)
    noisy = noise.uniform(np.full((18, 18, 3), 120.0), -25, 25, seed=2)
    assert filters.denoise_uniform(noisy, 3).std() < noisy.std()
    assert np.array_equal(filters.denoise_poisson(flat, 1.0), flat)
    grain = noise.poisson(flat, kappa=20, seed=3)
    assert (filters.denoise_poisson(grain, 1.2) - flat).var() < (grain - flat).var()


def test_denoise_additive_filters_clean_their_noise():
    flat = np.full((20, 20, 3), 120.0)
    rng = np.random.default_rng(4)
    # Gaussian: local Wiener reduces the variance of additive noise
    noisy = flat + rng.normal(0, 20, flat.shape)
    out = filters.denoise_gaussian(noisy, noise_std=20, size=3)
    assert (out - flat).var() < (noisy - flat).var()
    # constant image is unchanged, and noise_std = 0 is the identity
    assert np.array_equal(filters.denoise_gaussian(flat, 20, 3), flat)
    assert np.array_equal(filters.denoise_gaussian(noisy, 0, 3),
                          np.clip(np.floor(noisy + 0.5), 0, 255))
    # Rayleigh / exponential: positive mean is subtracted before smoothing
    noisy_r = flat + rng.rayleigh(scale=15, size=flat.shape)
    out_r = filters.denoise_rayleigh(noisy_r, noise_std=15 * math.sqrt((4 - math.pi) / 2), size=3)
    assert abs(out_r.mean() - 120) < abs(noisy_r.mean() - 120)
    noisy_e = flat + rng.exponential(scale=18, size=flat.shape)
    out_e = filters.denoise_exponential(noisy_e, noise_std=18, size=3)
    assert abs(out_e.mean() - 120) < abs(noisy_e.mean() - 120)
    # Gamma speckle: log-domain mean restores the flat level
    speckle = flat * rng.gamma(shape=10, scale=0.1, size=flat.shape)
    out_g = filters.denoise_gamma(speckle, 3)
    assert (out_g - flat).var() < (speckle - flat).var()


# ---------- line detection (masks, SHT, PHT, LSD, EDLines, Hessian) ----------
from app.imaging.ops import detection  # noqa: E402

# two bright 3 px lines on a dark background: rho = x cos(t) + y sin(t)
LINES = [(150.0, np.deg2rad(30)), (-40.0, np.deg2rad(-60))]


def _line_image(h=300, w=300, width=1.5, noise=0.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 90.0)
    for rho, t in LINES:
        img[np.abs(xx * np.cos(t) + yy * np.sin(t) - rho) < width] = 220.0
    img = img + np.random.default_rng(0).normal(0, noise, img.shape)
    return np.repeat(np.clip(img, 0, 255)[..., None], 3, axis=2)


def _red(out):
    return (out[..., 0] > 200) & (out[..., 1] < 90) & (out[..., 2] < 90)


def _distance_to_lines(ys, xs):
    return np.min([np.abs(xs * np.cos(t) + ys * np.sin(t) - rho) for rho, t in LINES], axis=0)


def test_gw_line_masks_respond_6_delta_on_a_one_pixel_line():
    # one-pixel line of height d above a flat background: R = 2*3*d - 3*d - 3*d... on
    # its own direction the centre row gives 2*3*(b+d) - 3*b - 3*b = 6d
    img = np.full((9, 9), 50.0)
    img[4, :] = 80.0
    r = detection.line_mask_responses(img)
    assert r["horizontal"][4, 4] == 6 * 30
    assert abs(r["vertical"][4, 4]) < abs(r["horizontal"][4, 4])
    diag = np.full((9, 9), 50.0)
    diag[np.arange(9), 8 - np.arange(9)] = 80.0  # rising line, /
    rd = detection.line_mask_responses(diag)
    assert rd["plus45"][4, 4] == 6 * 30 and rd["minus45"][4, 4] < rd["plus45"][4, 4]


def test_masks_mark_the_direction_of_each_line():
    img = np.full((40, 40, 3), 60.0)
    img[10, :] = 200.0   # horizontal
    img[:, 30] = 200.0   # vertical
    out = detection.line_detection(img, method="masks", direction="all", threshold=300)
    assert tuple(out[10, 5]) == detection.MASK_COLORS["horizontal"]
    assert tuple(out[25, 30]) == detection.MASK_COLORS["vertical"]
    # dark polarity does not mark the bright lines themselves; it does mark
    # their flanks, where the mask sees a dark row between bright ones
    # (the double-line effect of the G&W masks)
    dark = detection.line_detection(img, method="masks", polarity="dark", threshold=300)
    colours = [np.array(c) for c in detection.MASK_COLORS.values()]
    marked = lambda y, x: any((dark[y, x] == c).all() for c in colours)  # noqa: E731
    assert not marked(10, 5) and not marked(25, 30)
    assert marked(9, 5) and marked(11, 5)


def test_hough_accumulator_counts_one_vote_per_edge_pixel():
    edges = np.zeros((60, 80), dtype=bool)
    edges[30, :] = True  # horizontal line: theta = 90 deg is not in [-90, 90), -90 is
    acc, thetas, diag = detection.hough_accumulator(edges)
    assert acc.sum() == edges.sum() * thetas.size
    j = int(np.argmin(np.abs(thetas - np.deg2rad(-90))))
    assert acc[-30 + diag, j] == 80  # rho = y sin(-90) = -30, all 80 pixels


def test_sht_finds_known_lines_in_noise():
    for noise in (0, 15):
        edges = detection.canny_edges(_line_image(noise=noise)[..., 0], 1.4, 50, 150)
        acc, thetas, diag = detection.hough_accumulator(edges)
        peaks = detection.hough_peaks(acc, thetas, diag, 75, 4)
        for rho, t in LINES:
            # each thick line gives two parallel edges, about 2 px either side
            assert any(abs(r - rho) <= 4 and abs(th - t) <= np.deg2rad(2) for r, th, _ in peaks), noise


def test_sht_hough_view_and_drawn_lines():
    img = _line_image()
    hough = detection.line_detection(img, method="sht", show="hough", vote_threshold=0.25)
    assert hough.shape == img.shape
    assert ((hough[..., 1] > 200) & (hough[..., 0] < 100)).any()  # peaks circled in green
    out = detection.line_detection(img, method="sht", vote_threshold=0.25, max_lines=4)
    ys, xs = np.nonzero(_red(out))
    assert ys.size and np.mean(_distance_to_lines(ys, xs) < 6) > 0.95
    with pytest.raises(geometry.OperationError):
        detection.line_detection(img, method="pht", show="hough")


@pytest.mark.parametrize("method", ["pht", "lsd", "edlines"])
def test_segment_methods_draw_on_the_known_lines(method):
    img = _line_image(noise=8)
    out = detection.line_detection(img, method=method, votes=60, min_length=40, max_gap=10)
    ys, xs = np.nonzero(_red(out))
    assert ys.size > 200
    assert np.mean(_distance_to_lines(ys, xs) < 6) > 0.95
    # the same call gives the same image (/process and /export must agree)
    assert np.array_equal(out, detection.line_detection(img, method=method, votes=60, min_length=40, max_gap=10))


def test_hessian_line_points_lie_on_the_line_centre():
    img = _line_image(width=2.0)[..., 0]
    points, _ = detection.ridge_points(img, 2.0, "bright", 4.0)
    ys, xs = np.nonzero(points)
    assert ys.size > 300
    assert np.mean(_distance_to_lines(ys, xs) <= 1.0) > 0.9  # sub-pixel test keeps the centre
    # a dark line on a bright background gives the same points
    dark, _ = detection.ridge_points(255 - img, 2.0, "dark", 4.0)
    assert (points & dark).sum() / (points | dark).sum() > 0.98
    # looking for dark lines finds almost nothing, and nothing on the bright
    # lines' centres (only a few points at the crossing and the border)
    none, _ = detection.ridge_points(img, 2.0, "dark", 4.0)
    assert none.sum() < 0.1 * points.sum()
    ny, nx = np.nonzero(none)
    assert not (_distance_to_lines(ny, nx) <= 0.5).any()


def test_line_detection_keeps_the_image_dimmed_where_nothing_is_found():
    flat = np.full((40, 50, 3), 100.0)
    for m in ("masks", "sht", "pht", "lsd", "edlines", "hessian"):
        out = detection.line_detection(flat, method=m)
        assert out.shape == flat.shape and np.allclose(out, 60.0), m


def _on_square_border(ys, xs, y0, y1, x0, x1, tol=2):
    """True where a pixel is within tol of the boundary of [y0:y1, x0:x1]."""
    return (
        (np.abs(ys - y0) <= tol) | (np.abs(ys - (y1 - 1)) <= tol)
        | (np.abs(xs - x0) <= tol) | (np.abs(xs - (x1 - 1)) <= tol)
    )


def test_canny_edges_lie_on_the_square_and_thin_out_as_the_high_threshold_rises():
    y0, y1, x0, x1 = 20, 60, 25, 75
    img = np.full((80, 100, 3), 40.0)
    img[y0:y1, x0:x1] = 220.0
    out = detection.edge_detection(img, show="edges", sigma=1.0, canny_low=40, canny_high=120)
    white = out[..., 0] > 200
    ys, xs = np.nonzero(white)
    assert ys.size > 50
    assert np.mean(_on_square_border(ys, xs, y0, y1, x0, x1)) > 0.9
    assert np.array_equal(out, detection.edge_detection(img, show="edges", sigma=1.0, canny_low=40, canny_high=120))

    overlay = detection.edge_detection(img, show="image", sigma=1.0, canny_low=40, canny_high=120)
    assert _red(overlay).any()
    # away from the square the colour image is only dimmed
    assert np.allclose(overlay[5, 5], 24.0)

    flat = np.full((40, 50, 3), 100.0)
    assert not (detection.edge_detection(flat, show="edges") > 0).any()
    assert np.allclose(detection.edge_detection(flat, show="image"), 60.0)

    # a faint step survives a low T_H and drops out as T_H rises
    faint = np.full((60, 80, 3), 100.0)
    faint[15:45, 20:60] = 130.0
    loose = detection.edge_detection(faint, sigma=1.0, canny_low=10, canny_high=30)
    tight = detection.edge_detection(faint, sigma=1.0, canny_low=10, canny_high=200)
    assert (loose[..., 0] > 200).sum() > (tight[..., 0] > 200).sum()

    with pytest.raises(geometry.OperationError):
        detection.edge_detection(img, show="hough")


def _square(h=80, w=100, y0=20, y1=60, x0=25, x1=75, lo=40.0, hi=220.0):
    img = np.full((h, w, 3), lo)
    img[y0:y1, x0:x1] = hi
    return img


@pytest.mark.parametrize("method", ["sobel", "prewitt", "roberts", "log"])
def test_gradient_and_log_edges_follow_the_square(method):
    y0, y1, x0, x1 = 20, 60, 25, 75
    img = _square()
    out = detection.edge_detection(img, method=method, show="edges", sigma=1.0, threshold=20)
    white = out[..., 0] > 200
    ys, xs = np.nonzero(white)
    assert ys.size > 40, method
    assert np.mean(_on_square_border(ys, xs, y0, y1, x0, x1)) > 0.85, method
    flat = np.full((40, 50, 3), 100.0)
    assert not (detection.edge_detection(flat, method=method, show="edges", sigma=1.0, threshold=20) > 0).any()
    again = detection.edge_detection(img, method=method, show="edges", sigma=1.0, threshold=20)
    assert np.array_equal(out, again)
    # the response view is the strength, not a binary map: more than two levels
    response = detection.edge_detection(img, method=method, show="response", sigma=1.0, threshold=20)
    assert len(np.unique(response[..., 0])) > 2


def test_roberts_reports_more_false_edges_than_sobel_in_noise():
    rng = np.random.default_rng(0)
    img = _square() + rng.normal(0, 18, _square().shape)
    y0, y1, x0, x1 = 20, 60, 25, 75

    def off_border(method, threshold):
        out = detection.edge_detection(img, method=method, sigma=0, threshold=threshold)
        white = out[..., 0] > 200
        ys, xs = np.nonzero(white)
        return int((~_on_square_border(ys, xs, y0, y1, x0, x1, tol=3)).sum())

    # thresholds chosen so both still see the square; Roberts, looking at two
    # pixels, keeps far more of the noise
    assert off_border("roberts", 40) > 2 * off_border("sobel", 120)


def test_unknown_edge_method_is_rejected():
    img = _square()
    with pytest.raises(geometry.OperationError):
        detection.edge_detection(img, method="kirsch")


# ---------- Harris-Stephens corner detection ----------

def _brute_harris_response(gray, sigma, k):
    """Direct loops for Sobel + Gaussian moments + R, independent of production code."""
    h, w = gray.shape
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    padded = np.pad(gray, 1, mode="reflect")
    gx, gy = np.empty_like(gray), np.empty_like(gray)
    for y in range(h):
        for x in range(w):
            patch = padded[y:y + 3, x:x + 3]
            # scipy.ndimage.convolve reverses the kernel; these Sobels are antisymmetric.
            gx[y, x] = np.sum(patch * sx[::-1, ::-1])
            gy[y, x] = np.sum(patch * sy[::-1, ::-1])

    radius = int(4 * sigma + 0.5)  # scipy.ndimage.gaussian_filter's default truncate=4
    t = np.arange(-radius, radius + 1)
    g = np.exp(-(t * t) / (2 * sigma * sigma))
    g /= g.sum()
    kernel = np.outer(g, g)

    def smooth(z):
        p = np.pad(z, radius, mode="reflect")
        out = np.empty_like(z)
        for y in range(h):
            for x in range(w):
                out[y, x] = np.sum(p[y:y + 2 * radius + 1, x:x + 2 * radius + 1] * kernel)
        return out

    a, b, c = smooth(gx * gx), smooth(gy * gy), smooth(gx * gy)
    return a * b - c * c - k * (a + b) ** 2


def test_harris_response_matches_direct_formula_and_reflection_border():
    gray = np.random.default_rng(81).integers(0, 256, (7, 9)).astype(np.float64)
    expected = _brute_harris_response(gray, sigma=0.8, k=0.04)
    actual = detection.harris_response(gray, sigma=0.8, k=0.04)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)


def test_harris_finds_four_square_corners_but_no_flat_or_edge_corners():
    square = np.zeros((80, 80), dtype=np.float64)
    square[20:60, 20:60] = 255
    response = detection.harris_response(square, sigma=1.0, k=0.04)
    points = detection.harris_points(response, quality=0.01, nms_size=5, max_corners=200)
    ys, xs = np.nonzero(points)
    assert len(xs) == 4
    for x, y in ((20, 20), (59, 20), (20, 59), (59, 59)):
        assert np.min(np.hypot(xs - x, ys - y)) <= 1

    flat_response = detection.harris_response(np.full((30, 40), 100.0))
    assert np.all(flat_response == 0)
    assert not detection.harris_points(flat_response).any()

    edge = np.zeros((60, 60), dtype=np.float64)
    edge[:, 30:] = 255
    edge_response = detection.harris_response(edge)
    assert edge_response.max() <= 1e-8  # one large and one zero eigenvalue: det(M) = 0
    assert not detection.harris_points(edge_response).any()


def test_harris_threshold_nms_and_max_corners():
    response = np.zeros((15, 15), dtype=np.float64)
    response[3, 3] = 10
    response[3, 4] = 9       # suppressed by the stronger adjacent peak
    response[12, 12] = 8
    response[7, 7] = 0.05    # below quality=0.01 times the maximum (0.1)
    points = detection.harris_points(response, quality=0.01, nms_size=3, max_corners=10)
    assert np.array_equal(np.argwhere(points), np.array([[3, 3], [12, 12]]))
    strongest_only = detection.harris_points(response, quality=0.01, nms_size=3, max_corners=1)
    assert np.array_equal(np.argwhere(strongest_only), np.array([[3, 3]]))


def test_corner_detection_draws_green_circles_and_has_response_view():
    square = np.zeros((80, 80, 3), dtype=np.float64)
    square[20:60, 20:60] = 255
    for show in ("image", "response"):
        out = detection.corner_detection(square, show=show)
        assert out.shape == square.shape and out.dtype == np.float64
        green = (out[..., 1] > 180) & (out[..., 0] < 160) & (out[..., 2] < 160)
        assert green.any()
        assert np.array_equal(out, detection.corner_detection(square, show=show))
    flat = np.full((40, 50, 3), 100.0)
    assert np.all(detection.corner_detection(flat) == 60)


# ---------- fast (OpenCV) filters for the live camera ----------
from app.imaging.ops import fast as fast_ops  # noqa: E402

CAM = np.random.default_rng(20).integers(0, 256, (48, 64, 3)).astype(np.float64)


@pytest.mark.parametrize("name,kw", [
    ("gaussian_blur", {"sigma": 1.5}), ("gaussian_blur", {"sigma": 3.0}),
    ("sharpen", {"amount": 1.0, "sigma": 1.5}),
    ("arithmetic_mean", {"size": 5}), ("geometric_mean", {"size": 3}), ("harmonic_mean", {"size": 3}),
    ("minimum", {"size": 5}), ("maximum", {"size": 5}), ("midpoint", {"size": 3}),
])
def test_fast_filters_equal_the_exact_ones_up_to_rounding(name, kw):
    exact = getattr(filters, name)(CAM, **kw)
    quick = fast_ops.FAST[name](CAM, **kw)
    assert quick.shape == exact.shape
    assert np.abs(quick - exact).max() <= 1


def test_fast_median_equals_the_exact_one_away_from_the_border():
    for size in (3, 5, 7):
        r = size // 2
        exact = filters.median(CAM, size)
        quick = fast_ops.median(CAM, size)
        assert np.array_equal(quick[r:-r, r:-r], exact[r:-r, r:-r])
