"""Advanced tools: classic feature detectors, no ML or deep learning.

Line detection offers six methods, chosen with `method`:

  masks    Gonzalez & Woods line-detection masks (Sec. 10.2): four 3x3
           kernels, one per direction, thresholded response.
  sht      Standard Hough transform (G&W Sec. 10.2) on Canny edges:
           rho = x cos(theta) + y sin(theta), accumulator, peaks -> lines.
  pht      Progressive probabilistic Hough (Matas et al. 2000, OpenCV
           HoughLinesP): votes from random edge points, returns segments.
  lsd      Line Segment Detector (von Gioi et al. 2010, OpenCV): gradient
           regions validated by the number of false alarms.
  edlines  EDLines (Akinlar & Topal 2011, OpenCV contrib): Edge Drawing
           chains, least-squares segments, NFA validation.
  hessian  Steger's ridge detector: Hessian of the Gaussian-smoothed image,
           line points where the derivative along the normal vanishes.

Corner detection implements Harris & Stephens (1988): Sobel derivatives,
a Gaussian-windowed second-moment matrix M, the response
R = det(M) - k tr(M)^2, relative thresholding and non-maximum suppression.

Canny comes from OpenCV; the Harris response is computed here with NumPy
and SciPy. The methods work on the luma of the image; the lines/corners
are drawn over the colour image (dimmed) or over what the detector sees
(show="response"), or the Hough accumulator is shown (show="hough",
SHT only). Edge detection is left for teammates to add here.

Inputs and outputs are float64 RGB arrays of shape (H, W, 3) in [0, 255].
"""

import cv2
import numpy as np
from scipy.ndimage import convolve as _convolve
from scipy.ndimage import correlate as _correlate
from scipy.ndimage import gaussian_filter as _gaussian
from scipy.ndimage import maximum_filter as _max_filter

from .basic import L, luma
from .filters import gaussian_blur
from .geometry import OperationError

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

# G&W line-detection masks. The image y axis points down, so the "+45"
# mask (rising line, /) has its 2s on the anti-diagonal.
LINE_MASKS = {
    "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float64),
    "plus45": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float64),
    "vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float64),
    "minus45": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float64),
}
MASK_COLORS = {
    "horizontal": (255, 64, 64),
    "plus45": (80, 220, 100),
    "vertical": (70, 160, 255),
    "minus45": (255, 200, 40),
}
LINE_COLOR = (255, 40, 40)
PEAK_COLOR = (60, 230, 90)
CORNER_COLOR = (60, 230, 90)
DIM = 0.6  # the image under the drawn features is dimmed to 60%


# ---------- shared helpers ----------

def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical Sobel responses (reflection border)."""
    gx = _convolve(gray, _SOBEL_X, mode="mirror")
    gy = _convolve(gray, _SOBEL_Y, mode="mirror")
    return gx, gy


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """|gradient| from the two 3x3 Sobel kernels."""
    gx, gy = sobel_gradients(gray)
    return np.hypot(gx, gy)


def _to_u8(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(gray + 0.5), 0, L - 1).astype(np.uint8)


def canny_edges(gray: np.ndarray, sigma: float, low: float, high: float) -> np.ndarray:
    """Canny edge map (G&W Sec. 10.2): Gaussian smoothing with sigma (the
    project's own kernel), then OpenCV's Canny: Sobel gradient with the
    exact magnitude sqrt(gx^2 + gy^2), non-maximum suppression and
    hysteresis thresholding with T_L = low, T_H = high."""
    if sigma > 0:
        gray = gaussian_blur(gray[..., None], sigma)[..., 0]
    return cv2.Canny(_to_u8(gray), low, high, L2gradient=True) > 0


def harris_response(gray: np.ndarray, sigma: float = 1.0, k: float = 0.04) -> np.ndarray:
    """Harris-Stephens corner response (Harris & Stephens, 1988).

    Sobel gradients give I_x and I_y. A Gaussian window of scale sigma
    integrates their products into M = [[A, C], [C, B]], where
    A = G_sigma * I_x^2, B = G_sigma * I_y^2 and C = G_sigma * (I_x I_y).
    The response R = det(M) - k tr(M)^2 = AB - C^2 - k(A + B)^2
    avoids an explicit eigenvalue decomposition. Borders use the project's
    reflection-without-repetition rule throughout.
    """
    gx, gy = sobel_gradients(gray)
    a = _gaussian(gx * gx, sigma, mode="mirror")
    b = _gaussian(gy * gy, sigma, mode="mirror")
    c = _gaussian(gx * gy, sigma, mode="mirror")
    return a * b - c * c - k * (a + b) ** 2


def _normalised(x: np.ndarray) -> np.ndarray:
    top = float(x.max())
    return x * ((L - 1) / top) if top > 0 else np.zeros_like(x)


def _as_rgb_u8(gray: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.repeat(_to_u8(gray)[..., None], 3, axis=2))


def _thickness(h: int, w: int) -> int:
    # about 3 px on a 1000 px image, never thinner than 2 px
    return max(2, round(max(h, w) / 350))


def _draw_segments(canvas: np.ndarray, segments: np.ndarray, color, thickness: int) -> None:
    for x1, y1, x2, y2 in np.asarray(segments, dtype=np.float64).reshape(-1, 4):
        cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                 color, thickness, cv2.LINE_AA)


def _segment_lengths(segments: np.ndarray) -> np.ndarray:
    s = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    return np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])


# ---------- G&W line masks ----------

def line_mask_responses(gray: np.ndarray) -> dict[str, np.ndarray]:
    """R = sum_k w_k z_k for each of the four G&W masks (reflection border)."""
    return {name: _correlate(gray, w, mode="mirror") for name, w in LINE_MASKS.items()}


def _masks(gray, direction, polarity, threshold):
    """Line pixels and their direction. Bright lines need R >= T, dark
    lines -R >= T. With direction="all" each pixel takes the mask with
    the largest response, as G&W suggest for finding the direction."""
    sign = 1.0 if polarity == "bright" else -1.0
    responses = {k: sign * r for k, r in line_mask_responses(gray).items()}
    names = list(LINE_MASKS) if direction == "all" else [direction]
    stack = np.stack([responses[n] for n in names])
    best = stack.argmax(axis=0)
    value = stack.max(axis=0)
    return names, best, value, value >= threshold


# ---------- standard Hough transform ----------

def hough_accumulator(edges: np.ndarray):
    """A(rho, theta) of G&W's normal representation rho = x cos(theta) + y sin(theta).

    theta runs over -90 <= theta < 90 degrees in 1 degree steps, rho over
    -D..D in 1 pixel steps (D = image diagonal). Every edge pixel adds one
    vote to every (rho, theta) whose line passes through it.
    """
    h, w = edges.shape
    ys, xs = np.nonzero(edges)
    thetas = np.deg2rad(np.arange(-90, 90, 1.0))
    diag = int(np.ceil(np.hypot(h, w)))
    acc = np.zeros((2 * diag + 1, thetas.size), dtype=np.int64)
    if xs.size:
        for j, t in enumerate(thetas):
            rho = np.round(xs * np.cos(t) + ys * np.sin(t)).astype(np.int64) + diag
            acc[:, j] = np.bincount(rho, minlength=acc.shape[0])
    return acc, thetas, diag


def hough_peaks(acc, thetas, diag, min_votes: int, max_lines: int):
    """The max_lines strongest local maxima with at least min_votes votes,
    at least a few pixels and 5 degrees apart: (rho, theta, votes)."""
    min_rho_gap = max(2.0, 0.02 * diag)
    min_theta_gap = np.deg2rad(5)
    local = acc == _max_filter(acc, size=(2 * int(min_rho_gap) + 1, 5), mode="constant")
    rows, cols = np.nonzero(local & (acc >= max(1, min_votes)))
    order = np.argsort(-acc[rows, cols], kind="stable")
    chosen = []
    for i in order:
        rho, theta = float(rows[i] - diag), float(thetas[cols[i]])
        if any(abs(rho - r) < min_rho_gap and abs(theta - t) < min_theta_gap for r, t, _ in chosen):
            continue
        chosen.append((rho, theta, int(acc[rows[i], cols[i]])))
        if len(chosen) == max_lines:
            break
    return chosen


def _line_endpoints(rho: float, theta: float, h: int, w: int):
    """Two points far outside the image on the line rho = x cos + y sin."""
    x0, y0 = rho * np.cos(theta), rho * np.sin(theta)
    dx, dy = -np.sin(theta), np.cos(theta)
    far = 2.0 * np.hypot(h, w)
    return x0 - far * dx, y0 - far * dy, x0 + far * dx, y0 + far * dy


def _hough_image(acc, peaks, thetas, diag, h, w):
    """The accumulator as an H x W image: theta left to right (-90..90),
    rho top to bottom (-D..D), log(1 + A) for contrast, peaks circled."""
    img = cv2.resize(_to_u8(_normalised(np.log1p(acc.astype(np.float64)))), (w, h),
                     interpolation=cv2.INTER_AREA)
    canvas = np.ascontiguousarray(np.repeat(img[..., None], 3, axis=2))
    r = max(4, round(max(h, w) / 70))
    for rho, theta, _ in peaks:
        j = int(round(np.rad2deg(theta))) + 90
        cx = int((j + 0.5) * w / thetas.size)
        cy = int((rho + diag + 0.5) * h / acc.shape[0])
        cv2.circle(canvas, (cx, cy), r, PEAK_COLOR, max(1, r // 3), cv2.LINE_AA)
    return canvas


# ---------- Steger / Hessian ----------

def ridge_points(gray: np.ndarray, sigma: float, polarity: str, threshold: float):
    """Steger's line points. With r = G_sigma * f and its derivatives,
    the Hessian H = [[r_xx, r_xy], [r_xy, r_yy]] has its largest-magnitude
    eigenvalue lambda across the line, with unit eigenvector n (the line
    normal). Along n the profile peaks at offset

        t = -(r_x n_x + r_y n_y) / (r_xx n_x^2 + 2 r_xy n_x n_y + r_yy n_y^2),

    a pixel is a line point if that peak lies inside it (|t n_x|, |t n_y|
    <= 1/2) and the scale-normalised strength sigma^2 |lambda| reaches the
    threshold, with lambda < 0 for bright lines and > 0 for dark ones.
    Returns (points, strength).
    """
    d = lambda oy, ox: _gaussian(gray, sigma, order=(oy, ox), mode="mirror")  # noqa: E731
    rx, ry = d(0, 1), d(1, 0)
    rxx, ryy, rxy = d(0, 2), d(2, 0), d(1, 1)
    mean = (rxx + ryy) / 2
    disc = np.sqrt(((rxx - ryy) / 2) ** 2 + rxy ** 2)
    l1, l2 = mean + disc, mean - disc
    lam = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
    # eigenvector of lam: (r_xy, lam - r_xx) or (lam - r_yy, r_xy), whichever is not ~0
    ax, ay = rxy, lam - rxx
    bx, by = lam - ryy, rxy
    use_a = np.hypot(ax, ay) >= np.hypot(bx, by)
    nx, ny = np.where(use_a, ax, bx), np.where(use_a, ay, by)
    norm = np.hypot(nx, ny)
    flat = norm < 1e-12
    nx = np.where(flat, 1.0, nx / np.where(flat, 1.0, norm))
    ny = np.where(flat, 0.0, ny / np.where(flat, 1.0, norm))
    denom = rxx * nx * nx + 2 * rxy * nx * ny + ryy * ny * ny
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(rx * nx + ry * ny) / denom
    strength = (-lam if polarity == "bright" else lam) * sigma * sigma
    inside = (np.abs(t * nx) <= 0.5) & (np.abs(t * ny) <= 0.5)
    return inside & (strength >= threshold) & np.isfinite(t), np.maximum(strength, 0)


def _drop_short(points: np.ndarray, min_length: int) -> np.ndarray:
    """Keeps connected curves of at least min_length pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(points.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_length
    return keep[labels]


def _paint_points(canvas, points, color, thickness):
    if thickness > 1:
        points = cv2.dilate(points.astype(np.uint8), np.ones((thickness, thickness), np.uint8)) > 0
    canvas[points] = color


def harris_points(response: np.ndarray, quality: float = 0.01,
                  nms_size: int = 5, max_corners: int = 200) -> np.ndarray:
    """Threshold and non-maximum suppression for a Harris response.

    A corner must be a local maximum in an nms_size x nms_size window and
    have R >= quality * max(R). The relative threshold remains meaningful
    when image contrast changes. At most max_corners strongest peaks are
    returned, with row-major order breaking equal-response ties.
    """
    strongest = float(response.max())
    if strongest <= 0:
        return np.zeros(response.shape, dtype=bool)
    local_max = response == _max_filter(response, size=nms_size, mode="mirror")
    candidates = local_max & (response >= quality * strongest)
    ys, xs = np.nonzero(candidates)
    order = np.argsort(-response[ys, xs], kind="stable")[:max_corners]
    points = np.zeros(response.shape, dtype=bool)
    points[ys[order], xs[order]] = True
    return points


# ---------- the operations ----------

def line_detection(rgb: np.ndarray, method: str = "sht", show: str = "image",
                   sigma: float = 1.4, canny_low: float = 50, canny_high: float = 150,
                   vote_threshold: float = 0.15, max_lines: int = 10,
                   votes: int = 80, min_length: int = 40, max_gap: int = 10,
                   direction: str = "all", polarity: str = "bright", threshold: float = 300,
                   hessian_sigma: float = 2.0, ridge_threshold: float = 8.0) -> np.ndarray:
    """Detects straight lines (or, for hessian, thin curvilinear lines) and
    draws them. See the module docstring for the methods."""
    gray = luma(rgb)
    h, w = gray.shape
    thick = _thickness(h, w)
    if show == "hough" and method != "sht":
        raise OperationError("The Hough space view is only available for the standard Hough transform.")

    if method in ("sht", "pht"):
        edges = canny_edges(gray, sigma, canny_low, canny_high)
        response = edges * float(L - 1)
    elif method == "masks":
        names, best, value, hits = _masks(gray, direction, polarity, threshold)
        response = _normalised(np.maximum(value, 0))
    elif method == "hessian":
        hits, strength = ridge_points(gray, hessian_sigma, polarity, ridge_threshold)
        hits = _drop_short(hits, min_length)
        response = _normalised(strength)
    else:  # lsd, edlines: both start from the image gradient
        response = _normalised(sobel_magnitude(gray))

    if show == "response":
        canvas = _as_rgb_u8(response)
    elif show == "image":
        canvas = np.ascontiguousarray(np.clip(np.floor(rgb * DIM + 0.5), 0, L - 1).astype(np.uint8))

    if method == "sht":
        acc, thetas, diag = hough_accumulator(edges)
        peaks = hough_peaks(acc, thetas, diag, round(vote_threshold * max(h, w)), max_lines)
        if show == "hough":
            return _hough_image(acc, peaks, thetas, diag, h, w).astype(np.float64)
        for rho, theta, _ in peaks:
            _draw_segments(canvas, np.array(_line_endpoints(rho, theta, h, w)), LINE_COLOR, thick)
    elif method == "pht":
        segs = cv2.HoughLinesP(edges.astype(np.uint8) * 255, 1, np.pi / 180, int(votes),
                               minLineLength=int(min_length), maxLineGap=int(max_gap))
        if segs is not None:
            _draw_segments(canvas, segs, LINE_COLOR, thick)
    elif method in ("lsd", "edlines"):
        u8 = _to_u8(gray)
        if method == "lsd":
            segs = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(u8)[0]
        else:
            ed = cv2.ximgproc.createEdgeDrawing()
            ed.detectEdges(u8)
            segs = ed.detectLines()
        if segs is not None and len(segs):
            segs = np.asarray(segs).reshape(-1, 4)
            _draw_segments(canvas, segs[_segment_lengths(segs) >= min_length], LINE_COLOR, thick)
    elif method == "masks":
        if len(names) == 1:
            _paint_points(canvas, hits, LINE_COLOR, thick - 1)
        else:
            for i, name in enumerate(names):
                _paint_points(canvas, hits & (best == i), MASK_COLORS[name], thick - 1)
    elif method == "hessian":
        _paint_points(canvas, hits, LINE_COLOR, thick - 1)
    else:
        raise OperationError(f"Unknown line detection method: {method}")
    return canvas.astype(np.float64)


def corner_detection(rgb: np.ndarray, show: str = "image", sigma: float = 1.0,
                     k: float = 0.04, quality: float = 0.01,
                     nms_size: int = 5, max_corners: int = 200) -> np.ndarray:
    """Mark Harris-Stephens corners with green circles.

    show="image" draws over the dimmed colour image; show="response"
    draws over the positive Harris response, normalized for display.
    """
    gray = luma(rgb)
    response = harris_response(gray, sigma, k)
    points = harris_points(response, quality, nms_size, max_corners)
    if show == "response":
        canvas = _as_rgb_u8(_normalised(np.maximum(response, 0)))
    elif show == "image":
        canvas = np.ascontiguousarray(
            np.clip(np.floor(rgb * DIM + 0.5), 0, L - 1).astype(np.uint8)
        )
    else:
        raise OperationError(f"Unknown corner detection view: {show}")
    thick = _thickness(*gray.shape)
    radius = max(3, 2 * thick)
    for y, x in zip(*np.nonzero(points)):
        cv2.circle(canvas, (int(x), int(y)), radius, CORNER_COLOR, thick, cv2.LINE_AA)
    return canvas.astype(np.float64)
