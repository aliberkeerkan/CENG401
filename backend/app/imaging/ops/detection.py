"""Advanced tools: classic feature detectors, no ML or deep learning.

Only line detection is implemented here (Gonzalez & Woods Ch. 10: edge
detection followed by a Hough transform). Edge and corner detection are
left for teammates to add in this module, in the same hand-written style.

Inputs and outputs are float64 RGB arrays of shape (H, W, 3) in [0, 255].
"""

import numpy as np
from scipy.ndimage import binary_dilation as _dilate
from scipy.ndimage import convolve as _convolve
from scipy.ndimage import gaussian_filter as _gaussian
from scipy.ndimage import maximum_filter as _max_filter

from .basic import clip, luma, round_half_up

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

# Smoothing before differentiation, so texture and sensor noise do not
# turn into edge points (the Gaussian step of the Canny detector).
_PRESMOOTH_SIGMA = 1.2
# Each edge point votes only for line angles this close to its own
# gradient direction, instead of all 180.
_ANGLE_WINDOW_DEG = 2


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical Sobel responses (reflection border)."""
    gx = _convolve(gray, _SOBEL_X, mode="mirror")
    gy = _convolve(gray, _SOBEL_Y, mode="mirror")
    return gx, gy


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """|gradient| from the two 3x3 Sobel kernels."""
    gx, gy = sobel_gradients(gray)
    return np.hypot(gx, gy)


def _thin_edges(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Non-maximum suppression: keep a pixel only if its magnitude is the
    largest along its gradient direction, so every edge is one pixel wide
    and casts one set of votes instead of a thick band of them."""
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    sector = np.round(angle / 45.0).astype(int) % 4  # 0, 45, 90, 135 degrees
    padded = np.pad(magnitude, 1, mode="edge")
    h, w = magnitude.shape

    def shifted(dy: int, dx: int) -> np.ndarray:
        return padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    # neighbours along the gradient for each of the four sectors
    steps = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    keep = np.zeros(magnitude.shape, dtype=bool)
    for s, (dy, dx) in steps.items():
        m = sector == s
        keep |= m & (magnitude >= shifted(dy, dx)) & (magnitude >= shifted(-dy, -dx))
    return keep


def _hough_lines(edges: np.ndarray, vote_threshold: float, max_lines: int,
                 angles: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Classic Hough transform for straight lines rho = x cos(theta) + y sin(theta).

    theta is stepped every degree over [0, 180). Without `angles` every edge
    pixel votes for all 180 thetas. With `angles` (the gradient direction of
    each edge pixel, in degrees) it votes only within a few degrees of it:
    the gradient is normal to the line, so other angles are noise.

    Bins with at least vote_threshold * max(H, W) votes that are also a local
    maximum of the accumulator are candidates; the max_lines strongest are
    kept, a few pixels and degrees apart.
    """
    h, w = edges.shape
    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return []

    thetas = np.deg2rad(np.arange(0, 180, 1.0))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    diag = int(np.ceil(np.hypot(h, w)))
    n_rho = 2 * diag + 1
    acc = np.zeros((n_rho, thetas.size), dtype=np.int64)

    def vote(theta_idx: np.ndarray | int, px: np.ndarray, py: np.ndarray) -> None:
        rho = px * cos_t[theta_idx] + py * sin_t[theta_idx]
        bins = np.clip(np.round(rho).astype(np.int64) + diag, 0, n_rho - 1)
        np.add.at(acc, (bins, theta_idx), 1)

    if angles is None:
        for j in range(thetas.size):
            vote(j, xs, ys)
    else:
        base = np.round(angles[ys, xs]).astype(np.int64)
        for d in range(-_ANGLE_WINDOW_DEG, _ANGLE_WINDOW_DEG + 1):
            vote((base + d) % 180, xs, ys)

    min_rho_gap = max(2.0, 0.02 * diag)
    min_theta_gap = np.deg2rad(5)
    peaks = acc == _max_filter(acc, size=(2 * int(min_rho_gap) + 1, 5), mode="constant")
    min_votes = max(1, int(vote_threshold * max(h, w)))
    rows, cols = np.nonzero(peaks & (acc >= min_votes))
    if rows.size == 0:
        return []
    order = np.argsort(-acc[rows, cols], kind="stable")

    chosen: list[tuple[float, float]] = []
    for idx in order:
        rho, theta = float(rows[idx] - diag), float(thetas[cols[idx]])
        if any(abs(rho - cr) < min_rho_gap and abs(theta - ct) < min_theta_gap
               for cr, ct in chosen):
            continue
        chosen.append((rho, theta))
        if len(chosen) == max_lines:
            break
    return chosen


def _line_segments(support: np.ndarray, rho: float, theta: float,
                   min_length: int, max_gap: int) -> list[np.ndarray]:
    """Walk along a Hough line and keep only the stretches that edge pixels
    actually cover (short gaps allowed), instead of the infinite line."""
    h, w = support.shape
    diag = int(np.ceil(np.hypot(h, w)))
    a, b = np.cos(theta), np.sin(theta)
    t = np.arange(-diag, diag + 1)
    xs = np.round(rho * a - t * b).astype(np.int64)
    ys = np.round(rho * b + t * a).astype(np.int64)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[inside], ys[inside]
    if xs.size == 0:
        return []
    on = support[ys, xs]

    segments, start, last = [], None, None
    for i, hit in enumerate(on):
        if not hit:
            continue
        if start is None:
            start = last = i
        elif i - last > max_gap + 1:
            if last - start + 1 >= min_length:
                segments.append(np.stack([ys[start:last + 1], xs[start:last + 1]], axis=1))
            start = i
        last = i
    if start is not None and last - start + 1 >= min_length:
        segments.append(np.stack([ys[start:last + 1], xs[start:last + 1]], axis=1))
    return segments


def _paint(canvas: np.ndarray, points: np.ndarray, color: np.ndarray, radius: int) -> None:
    """Paints a disc of the given radius around every point."""
    h, w = canvas.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            y, x = points[:, 0] + dy, points[:, 1] + dx
            ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            canvas[y[ok], x[ok]] = color


def line_detection(rgb: np.ndarray, edge_threshold: float, vote_threshold: float,
                   max_lines: int) -> np.ndarray:
    """Straight-line detection with a Hough transform; no learned model.

    1. Luma is smoothed with a Gaussian, then Sobel gives the gradient.
    2. Non-maximum suppression thins edges to one pixel; pixels whose
       magnitude is at least edge_threshold are edge points.
    3. Each edge point votes in the Hough accumulator only near its own
       gradient direction.
    4. The max_lines strongest accumulator peaks above vote_threshold are
       kept, and only the stretches of each line that edge pixels cover are
       drawn, in red, over the image dimmed to 60%.
    """
    gray = _gaussian(luma(rgb), _PRESMOOTH_SIGMA, mode="mirror")
    gx, gy = sobel_gradients(gray)
    magnitude = np.hypot(gx, gy)
    edges = _thin_edges(magnitude, gx, gy) & (magnitude >= edge_threshold)
    angles = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

    lines = _hough_lines(edges, vote_threshold, int(max_lines), angles)

    h, w = edges.shape
    support = _dilate(edges, iterations=1)
    min_length = max(15, int(0.05 * max(h, w)))
    max_gap = max(4, int(0.01 * max(h, w)))
    # about 5 px wide on a 1000 px image, never thinner than 3 px
    radius = max(1, round(max(h, w) / 450))

    out = rgb * 0.6
    red = np.array([255.0, 32.0, 32.0])
    for rho, theta in lines:
        for seg in _line_segments(support, rho, theta, min_length, max_gap):
            _paint(out, seg, red, radius)
    return clip(round_half_up(out))
