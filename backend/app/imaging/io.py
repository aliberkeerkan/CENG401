"""Decoding uploads and encoding results.

Uploads (JPEG, PNG, TIFF) are converted to 8-bit RGB plus an optional
alpha channel. Results are always returned as PNG, which every browser
can display; this is how TIFF uploads become viewable.
"""

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 1600  # longest side kept after upload (risk table: large images)


class ImageDecodeError(ValueError):
    pass


def _to_8bit(img: Image.Image) -> Image.Image:
    """Scale 16-bit or 32-bit grayscale images (common in TIFF) to 8-bit."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(a)
        return Image.fromarray(np.floor(a + 0.5).astype(np.uint8), mode="L")
    return img


def decode(data: bytes) -> tuple[np.ndarray, np.ndarray | None, bool, str]:
    """Return (rgb uint8 HxWx3, alpha uint8 HxW or None, was_scaled, format)."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format or "unknown"
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ImageDecodeError("The file could not be read as an image.") from e

    img = ImageOps.exif_transpose(img)  # respect camera orientation
    img = _to_8bit(img)

    scaled = False
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        scaled = True

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if has_alpha:
        return arr[..., :3].copy(), arr[..., 3].copy(), scaled, fmt
    return arr, None, scaled, fmt


EXPORT_FORMATS = {
    "png": ("PNG", "image/png"),
    "jpeg": ("JPEG", "image/jpeg"),
    "tiff": ("TIFF", "image/tiff"),
}


def encode(rgb: np.ndarray, alpha: np.ndarray | None = None, fmt: str = "png", quality: int = 92) -> bytes:
    """Encode an RGB(+alpha) array. JPEG has no alpha channel, so transparent
    areas are flattened onto white."""
    pil_format, _ = EXPORT_FORMATS[fmt]
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    if fmt == "jpeg":
        if alpha is not None:
            a = alpha[..., None].astype(np.float64) / 255
            out = np.floor(out * a + 255 * (1 - a) + 0.5).astype(np.uint8)
        Image.fromarray(out).save(buf, format=pil_format, quality=quality)
    else:
        if alpha is not None:
            out = np.dstack([out, alpha])
        options = {"compress_level": 1} if fmt == "png" else {"compression": "tiff_deflate"}
        Image.fromarray(out).save(buf, format=pil_format, **options)
    return buf.getvalue()


def encode_png(rgb: np.ndarray, alpha: np.ndarray | None = None) -> bytes:
    return encode(rgb, alpha, "png")
