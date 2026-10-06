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


def test_contrast_identity_when_alpha_is_one():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(8, 8, 3)).astype(np.float64)
    assert np.array_equal(basic.contrast(x, alpha=1.0), x)


def test_contrast_pivot_is_mean_luma():
    x = img([[[100, 100, 100], [200, 200, 200]]])  # mean luma = 150
    out = basic.contrast(x, alpha=2.0)
    assert out[0, :, 0].tolist() == [50, 250]


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
